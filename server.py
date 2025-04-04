# -*- coding: utf-8 -*-
import asyncio
import json
import websockets
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM # Changed Gemma import
import numpy as np
import logging
import sys
import io
from PIL import Image
import time
# Assuming kokoro KPipeline is correctly installed and available
# from kokoro import KPipeline
from accelerate import Accelerator
import gc # Garbage Collector for explicit cleanup

# --- Mock Kokoro TTS for environments where it's not installed ---
class MockKokoroTTS:
    def __init__(self, *args, **kwargs):
        logger.warning("Using Mock Kokoro TTS. No actual audio will be generated.")
        self.sample_rate = 24000 # Typical TTS sample rate

    def __call__(self, text, *args, **kwargs):
        logger.info(f"Mock TTS called with text: '{text}'")
        # Generate silent audio matching roughly the text length
        # Estimate 5 words per second, 1 second = sample_rate samples
        num_words = len(text.split())
        duration_samples = int((num_words / 5.0) * self.sample_rate)
        if duration_samples == 0 and len(text) > 0:
             duration_samples = int(0.5 * self.sample_rate) # Min duration for non-empty text
        elif duration_samples == 0:
            return iter([]) # No audio for empty text

        # Yield a single chunk of silence
        silence = np.zeros(duration_samples, dtype=np.float32)
        # Kokoro yields tuples like (start_time, end_time, audio_chunk)
        yield (0, duration_samples / self.sample_rate, silence)

try:
    from kokoro import KPipeline
except ImportError:
    logger.warning("kokoro library not found. Using MockKokoroTTS.")
    KPipeline = MockKokoroTTS
# --- End Mock Kokoro TTS ---


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__) # Use __name__ for logger


class AudioSegmentDetector:
    """Detects speech segments based on energy and silence."""
    def __init__(self, sample_rate=16000, energy_threshold=0.015, silence_duration=0.5, min_speech_duration=0.5, max_speech_duration=10):
        self.sample_rate = sample_rate
        self.bytes_per_sample = 2 # For int16
        self.energy_threshold = energy_threshold
        self.silence_samples = int(silence_duration * sample_rate)
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.max_speech_samples = int(max_speech_duration * sample_rate)

        self.audio_buffer = bytearray()
        self.is_speech_active = False
        self.silence_counter_samples = 0 # Count silence in samples
        self.speech_start_byte_idx = 0
        self.lock = asyncio.Lock()
        self.segment_queue = asyncio.Queue(maxsize=5) # Allow a small buffer
        self.segments_detected = 0

        # --- Task Management for Interruption ---
        self.current_processing_task = None
        self.task_lock = asyncio.Lock()
        # --- TTS Playing State (remains useful for image handling) ---
        self.tts_playing = False
        self.tts_lock = asyncio.Lock()


    async def set_tts_playing(self, is_playing: bool):
        """Safely sets the TTS playing state."""
        async with self.tts_lock:
            if self.tts_playing != is_playing:
                logger.info(f"Setting TTS playing state to: {is_playing}")
                self.tts_playing = is_playing

    async def is_tts_currently_playing(self) -> bool:
        """Safely checks if TTS is playing."""
        async with self.tts_lock:
            return self.tts_playing

    async def cancel_current_processing(self):
        """Cancels the currently running processing task if it exists."""
        async with self.task_lock:
            task_cancelled = False
            if self.current_processing_task and not self.current_processing_task.done():
                logger.warning(">>> INTERRUPT: Cancelling current processing task. <<<")
                self.current_processing_task.cancel()
                try:
                    # Wait briefly for cancellation to propagate
                    await asyncio.wait_for(self.current_processing_task, timeout=0.1)
                except asyncio.CancelledError:
                    logger.info("Current processing task successfully cancelled.")
                    task_cancelled = True
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for task cancellation confirmation.")
                except Exception as e:
                    # Catch other potential exceptions during task cleanup
                    logger.error(f"Exception during task cancellation: {e}")

            # Clear the reference regardless of cancellation success
            self.current_processing_task = None
            # Clear any pending segments in the queue as they are now stale
            while not self.segment_queue.empty():
                try:
                    self.segment_queue.get_nowait()
                    logger.info("Removed stale segment from queue during interrupt.")
                except asyncio.QueueEmpty:
                    break
            # Reset TTS playing state as the cancelled task won't do it
            if task_cancelled:
                 await self.set_tts_playing(False)


    async def set_current_processing_task(self, task: asyncio.Task | None):
        """Sets the current processing task."""
        async with self.task_lock:
            # Cancel previous task if any (should ideally be done before calling this)
            if self.current_processing_task and not self.current_processing_task.done():
                 logger.warning("Overwriting an existing processing task reference without explicit cancellation.")
                 # Attempt cancellation just in case
                 self.current_processing_task.cancel()
            self.current_processing_task = task

    async def add_audio(self, audio_bytes: bytes):
        """Adds audio data and detects speech segments."""
        if not audio_bytes:
            return None

        async with self.lock:
            self.audio_buffer.extend(audio_bytes)
            # Process only the newly added bytes for energy calculation
            # Convert new bytes to numpy array
            if len(audio_bytes) % self.bytes_per_sample != 0:
                 logger.warning(f"Received partial audio sample ({len(audio_bytes)} bytes), skipping energy check for this chunk.")
                 # Still add to buffer, but don't process energy yet
                 return None

            try:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            except ValueError as e:
                logger.error(f"Error converting audio bytes to numpy array: {e}. Buffer length: {len(self.audio_buffer)}, new bytes: {len(audio_bytes)}")
                # If buffer gets corrupted, maybe reset it?
                # self.audio_buffer = bytearray()
                return None

            if len(audio_array) == 0:
                return None

            # Calculate energy of the new chunk
            energy = np.sqrt(np.mean(audio_array**2))
            num_new_samples = len(audio_array)

            # --- State Machine for Speech Detection ---
            if not self.is_speech_active:
                # Check if speech starts
                if energy > self.energy_threshold:
                    self.is_speech_active = True
                    # Start index is roughly where the new audio_bytes started in the buffer
                    self.speech_start_byte_idx = max(0, len(self.audio_buffer) - len(audio_bytes))
                    self.silence_counter_samples = 0
                    logger.info(f"Speech start detected (energy: {energy:.6f}) at buffer pos {self.speech_start_byte_idx}")
            else: # Speech is already active
                current_speech_duration_bytes = len(self.audio_buffer) - self.speech_start_byte_idx
                current_speech_duration_samples = current_speech_duration_bytes // self.bytes_per_sample

                # Check for silence ending the speech
                if energy <= self.energy_threshold:
                    self.silence_counter_samples += num_new_samples
                    if self.silence_counter_samples >= self.silence_samples:
                        # End of speech detected
                        speech_end_byte_idx = len(self.audio_buffer) - (self.silence_counter_samples * self.bytes_per_sample)
                        speech_segment_bytes = bytes(self.audio_buffer[self.speech_start_byte_idx : speech_end_byte_idx])
                        segment_duration_samples = len(speech_segment_bytes) // self.bytes_per_sample

                        # Reset state
                        self.is_speech_active = False
                        self.silence_counter_samples = 0
                        # Keep only the silent part in the buffer for potential future speech start
                        self.audio_buffer = self.audio_buffer[speech_end_byte_idx:]
                        self.speech_start_byte_idx = 0 # Reset start index

                        if segment_duration_samples >= self.min_speech_samples:
                            self.segments_detected += 1
                            logger.info(f"Speech segment [Silence End] detected: {segment_duration_samples / self.sample_rate:.2f}s ({len(speech_segment_bytes)} bytes)")
                            # --- INTERRUPT HANDLING ---
                            await self.cancel_current_processing() # Cancel previous before queuing new
                            try:
                                self.segment_queue.put_nowait(speech_segment_bytes)
                                logger.info(f"Segment added to queue. Queue size: {self.segment_queue.qsize()}")
                            except asyncio.QueueFull:
                                logger.warning("Segment queue full. Dropping segment.")
                            return speech_segment_bytes # Indicate detection
                        else:
                            logger.info(f"Speech segment too short ({segment_duration_samples / self.sample_rate:.2f}s), discarding.")

                else: # Still speech (energy above threshold)
                    self.silence_counter_samples = 0 # Reset silence counter

                    # Check for maximum speech duration
                    if current_speech_duration_samples > self.max_speech_samples:
                        logger.warning(f"Max speech duration exceeded ({current_speech_duration_samples / self.sample_rate:.2f}s). Segmenting.")
                        # Segment at max duration
                        segment_end_byte_idx = self.speech_start_byte_idx + (self.max_speech_samples * self.bytes_per_sample)
                        speech_segment_bytes = bytes(self.audio_buffer[self.speech_start_byte_idx : segment_end_byte_idx])

                        # Update state for the *next* potential segment
                        self.speech_start_byte_idx = segment_end_byte_idx # Start next segment from here
                        # Don't reset silence counter, as the audio continues

                        if len(speech_segment_bytes) > 0: # Ensure non-empty segment
                            self.segments_detected += 1
                            logger.info(f"Speech segment [Max Duration] detected: {self.max_speech_samples / self.sample_rate:.2f}s ({len(speech_segment_bytes)} bytes)")
                             # --- INTERRUPT HANDLING ---
                            await self.cancel_current_processing() # Cancel previous before queuing new
                            try:
                                self.segment_queue.put_nowait(speech_segment_bytes)
                                logger.info(f"Segment added to queue. Queue size: {self.segment_queue.qsize()}")
                            except asyncio.QueueFull:
                                logger.warning("Segment queue full. Dropping segment.")
                            return speech_segment_bytes # Indicate detection
                        else:
                             logger.warning("Max duration segment resulted in zero bytes, discarding.")


        # --- Buffer cleanup (optional, prevent excessive memory use) ---
        # max_buffer_size = self.sample_rate * self.bytes_per_sample * (self.max_speech_duration + 2 * self.silence_duration)
        # if len(self.audio_buffer) > max_buffer_size * 1.5: # Add some leeway
        #     keep_bytes = int(max_buffer_size)
        #     logger.warning(f"Audio buffer exceeding limit ({len(self.audio_buffer)} bytes). Truncating to {keep_bytes} bytes.")
        #     self.audio_buffer = self.audio_buffer[-keep_bytes:]
        #     # Adjust speech_start_byte_idx if necessary, though this might complicate things
        #     if self.is_speech_active:
        #        self.speech_start_byte_idx = max(0, self.speech_start_byte_idx - (len(self.audio_buffer) - keep_bytes))


        return None # No segment detected in this chunk

    async def get_next_segment(self) -> bytes | None:
        """Gets the next detected speech segment from the queue."""
        try:
            # Use a small timeout to avoid blocking indefinitely if queue becomes empty
            return await asyncio.wait_for(self.segment_queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            return None
        except asyncio.QueueEmpty: # Should be caught by timeout, but just in case
             return None


class WhisperTranscriber:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Creating WhisperTranscriber instance...")
            cls._instance = cls()
            logger.info("WhisperTranscriber instance created.")
        return cls._instance

    def __init__(self):
        if WhisperTranscriber._instance is not None:
             raise Exception("This class is a singleton!")
        else:
            WhisperTranscriber._instance = self

        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        # self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.torch_dtype = torch.bfloat16 # Force bfloat16 if desired, check compatibility
        logger.info(f"Using device: {self.device} with dtype: {self.torch_dtype}")

        # model_id = "openai/whisper-large-v3"
        model_id = "distil-whisper/distil-large-v3" # Use distilled version for faster inference
        logger.info(f"Loading Whisper model: {model_id}")
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                # attn_implementation="flash_attention_2" # FA2 can sometimes cause issues, enable if needed
                )
            # No need to explicitly move to device with Accelerator? No, pipeline needs it.
            self.model.to(self.device)

            self.processor = AutoProcessor.from_pretrained(model_id)

            logger.info("Creating Whisper pipeline...")
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
                # Chunking and batching can improve performance for longer audio
                chunk_length_s=30, # Process audio in 30s chunks
                stride_length_s=[6, 4], # Overlap chunks for better context (list for long/short strides)
                batch_size=4, # Adjust based on VRAM
                # Use Flash Attention 2 if available and compatible
                model_kwargs={"attn_implementation": "flash_attention_2"} if self.torch_dtype != torch.float32 else {}
                # model_kwargs={"use_flash_attention_2": True} # Old way?
            )
            # Accelerator might not be needed directly for pipeline, but good practice
            # self.pipe.model = self.accelerator.prepare(self.pipe.model) # Pipeline handles model placement

            self.transcription_count = 0
            logger.info(f"Whisper pipeline created successfully on device {self.device}.")
            logger.info(f"Whisper model memory footprint: {self.model.get_memory_footprint() / 1e9:.2f} GB")

        except Exception as e:
            logger.exception(f"FATAL: Failed to load Whisper model or pipeline: {e}")
            raise

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribes audio bytes to text."""
        self.transcription_count += 1
        start_time = time.time()
        try:
            if not audio_bytes or len(audio_bytes) < sample_rate * 0.1 * 2: # Need at least 0.1 seconds
                logger.warning(f"Audio too short for transcription ({len(audio_bytes)} bytes).")
                return ""

            # Convert bytes to float32 numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            logger.info(f"Transcribing audio segment: {len(audio_array)/sample_rate:.2f}s")

            # Run inference in executor thread
            result = await asyncio.get_event_loop().run_in_executor(
                None, # Use default executor
                lambda: self.pipe(
                    audio_array.copy(), # Send a copy to avoid issues if array is modified
                    generate_kwargs={"task": "transcribe", "language": "english"}
                    # return_timestamps=True # Could be useful later
                    )
            )

            # result = self.pipe(audio_array, generate_kwargs={"task": "transcribe", "language": "english"})

            text = result.get("text", "").strip()
            end_time = time.time()
            logger.info(f"Transcription #{self.transcription_count} ({(end_time - start_time)*1000:.1f}ms): '{text}'")
            # Explicitly clean up
            del audio_array
            del result
            gc.collect()
            if self.device.type == 'cuda':
                 torch.cuda.empty_cache()

            return text

        except RuntimeError as e:
             if "CUDA out of memory" in str(e):
                 logger.error(f"CUDA OOM Error during transcription: {e}. Attempting to clear cache.")
                 gc.collect()
                 if self.device.type == 'cuda':
                      torch.cuda.empty_cache()
                 return "[Transcription failed due to OOM]"
             else:
                  logger.exception(f"Runtime transcription error: {e}")
                  return "[Transcription runtime error]"
        except Exception as e:
            logger.exception(f"General transcription error: {e}")
            return "[Transcription error]"


class GemmaMultimodalProcessor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Creating GemmaMultimodalProcessor instance...")
            cls._instance = cls()
            logger.info("GemmaMultimodalProcessor instance created.")
        return cls._instance

    def __init__(self):
        if GemmaMultimodalProcessor._instance is not None:
            raise Exception("This class is a singleton!")
        else:
             GemmaMultimodalProcessor._instance = self

        self.accelerator = Accelerator()
        self.device = self.accelerator.device # Gemma should handle device mapping
        # self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.torch_dtype = torch.bfloat16 # Force bfloat16 if desired

        # --- Use Gemma 7B IT ---
        model_id = "google/gemma-7b-it"
        logger.info(f"Loading Gemma model: {model_id} with dtype: {self.torch_dtype}")

        try:
            # Use AutoModelForCausalLM for Gemma
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto", # Let accelerate handle device placement
                torch_dtype=self.torch_dtype,
                attn_implementation="flash_attention_2", # Use Flash Attention 2
                low_cpu_mem_usage=True # Add this for potentially large models
            )
            self.processor = AutoProcessor.from_pretrained(model_id)

            self.last_image = None
            self.last_image_timestamp = 0
            self.image_lock = asyncio.Lock() # Lock specifically for image access/update
            self.history_lock = asyncio.Lock() # Lock for message history access/update
            self.message_history = [] # Store as {"role": "user/model", "parts": ["text"]}
            self.generation_count = 0
            logger.info(f"Gemma model {model_id} loaded successfully.")
            logger.info(f"Gemma model memory footprint: {self.model.get_memory_footprint() / 1e9:.2f} GB")

        except Exception as e:
            logger.exception(f"FATAL: Failed to load Gemma model or processor: {e}")
            raise

    async def set_image(self, image_data: bytes):
        """Sets the image for multimodal context, clearing history."""
        if not image_data or len(image_data) < 100:
            logger.warning("Invalid or empty image data received, ignoring.")
            return False

        async with self.image_lock:
            try:
                logger.info(f"Processing new image ({len(image_data)} bytes)...")
                image = Image.open(io.BytesIO(image_data)).convert("RGB") # Ensure RGB

                # Optional: Resize if needed, but Gemma handles various sizes
                # max_size = (1024, 1024)
                # image.thumbnail(max_size, Image.Resampling.LANCZOS)

                # Clear message history when a new image is set
                async with self.history_lock:
                    logger.info("New image set, clearing conversation history.")
                    self.message_history = []

                self.last_image = image
                self.last_image_timestamp = time.time()
                logger.info("Image set successfully.")

                # Clean up (though PIL objects should be managed by GC)
                # del image # Not strictly necessary here

                return True
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                self.last_image = None # Ensure no stale image is kept on error
                return False

    def _build_prompt(self, text: str) -> str:
        """Builds the prompt string for the model, including history and image if available."""
        # Gemma uses a specific chat template format.
        # The processor should handle this via apply_chat_template.
        # We just need to construct the message list correctly.

        # Note: Gemma's multimodal capability might depend on specific versions/checkpoints.
        # The standard `google/gemma-7b-it` might not directly support image inputs
        # in the same way as dedicated vision models (like Llava or PaliGemma).
        # Let's assume text-only interaction for `gemma-7b-it` unless specifically documented otherwise.
        # If multimodal is needed, a different model like PaliGemma might be required.

        # --- Assuming Text-Only Interaction for gemma-7b-it ---
        chat = [
            # System prompt can be added if needed, but often omitted for simplicity
            # {"role": "system", "content": "You are a helpful assistant."}
        ]
        # Add history
        chat.extend(self.message_history)

        # Add current user turn
        chat.append({"role": "user", "content": text})

        # # --- Code block IF gemma-7b-it supported images like this (example, likely needs adjustment) ---
        # async with self.image_lock: # Access image safely
        #     current_image = self.last_image
        # chat = [
        #     # {"role": "system", "content": "..."} # System prompt if needed
        # ]
        # chat.extend(self.message_history) # Add past turns
        #
        # # Construct current user turn with optional image
        # user_parts = []
        # if current_image:
        #     # This is a placeholder - Gemma's image handling might differ.
        #     # It might expect the image data directly, or need specific formatting.
        #     # Check the specific model's documentation for image input format.
        #     user_parts.append({"type": "image"}) # Placeholder, actual data/format needed
        # user_parts.append({"type": "text", "text": text})
        # chat.append({"role": "user", "content": user_parts})
        # # --- End IF block ---

        return chat # Return the list of messages for the template processor


    async def _update_history(self, user_text: str, assistant_response: str):
        """Updates the conversation history (text only)."""
        async with self.history_lock:
            # Keep only the last interaction to prevent context overflow and focus on immediate conversation
            self.message_history = [
                 {"role": "user", "content": user_text},
                 {"role": "assistant", "content": assistant_response}
            ]
            # # To keep more history (example: last 3 turns = 6 messages)
            # max_history_turns = 3
            # self.message_history.append({"role": "user", "content": user_text})
            # self.message_history.append({"role": "assistant", "content": assistant_response})
            # self.message_history = self.message_history[-(max_history_turns * 2):]


    async def generate_streaming(self, text: str):
        """Generates a response stream using the Gemma model."""
        start_time = time.time()
        try:
            async with self.history_lock: # Lock history while building prompt
                 chat_messages = self._build_prompt(text)

            # --- Image Handling (Placeholder - adapt if model supports it) ---
            # async with self.image_lock:
            #     current_image = self.last_image
            # if current_image:
            #      logger.info("Including image in prompt (Format needs verification for Gemma-7B-IT)")
            #      # Modify inputs preparation to include image data correctly
            #      # This part is highly dependent on the exact model and processor requirements
            #      # Example: inputs = self.processor(text=None, images=current_image, chat_template=..., return_tensors="pt").to(self.device)
            #      pass # Requires specific implementation
            # else:
            #      # Prepare text-only input using the chat template
            #      inputs = self.processor.apply_chat_template(
            #           chat_messages,
            #           tokenize=True,
            #           add_generation_prompt=True, # Important for instruction-tuned models
            #           return_tensors="pt"
            #      ).to(self.device)

            # Prepare text-only input using the chat template
            # Run processor in executor? Probably not necessary as it's usually fast CPU work
            inputs = self.processor.apply_chat_template(
                    chat_messages,
                    tokenize=True,
                    add_generation_prompt=True, # Important for instruction-tuned models
                    return_tensors="pt"
            ).to(self.device)


            from transformers import TextIteratorStreamer
            # Run streamer creation in executor? No, it's lightweight
            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True, # Don't yield the input prompt
                skip_special_tokens=True
            )

            generation_kwargs = dict(
                inputs, # Contains input_ids and attention_mask
                streamer=streamer,
                # max_new_tokens=256, # Limit response length
                max_new_tokens=150, # Shorter limit for faster responses
                do_sample=False, # Use greedy decoding for more deterministic output
                # temperature=0.7, # Add some randomness if do_sample=True
                # top_k=50,
                # top_p=0.9,
                pad_token_id=self.processor.tokenizer.eos_token_id # Crucial for stopping generation
                #eos_token_id=self.processor.tokenizer.eos_token_id # Often set by default
            )

            # Run the generation in a separate thread to avoid blocking asyncio loop
            # Use run_in_executor for better asyncio integration
            loop = asyncio.get_event_loop()
            generation_future = loop.run_in_executor(
                 None, # Use default executor (thread pool)
                 lambda: self.model.generate(**generation_kwargs)
            )
            # generation_future is now an asyncio Future wrapping the thread execution

            # --- Process initial part of the stream ---
            initial_text = ""
            initial_chunk_count = 0
            min_initial_len = 10 # Characters
            max_initial_chunks = 5 # Limit how many chunks we wait for initially
            start_initial_time = time.time()

            try:
                 async for chunk in streamer: # Async iteration over the streamer
                      initial_text += chunk
                      initial_chunk_count += 1
                      # Break if we have enough text or hit a natural pause
                      if len(initial_text) >= min_initial_len or "." in chunk or "," in chunk or "!" in chunk or "?" in chunk or "\n" in chunk:
                           break
                      if initial_chunk_count >= max_initial_chunks:
                           logger.warning("Reached max initial chunks without sentence break.")
                           break
                      if time.time() - start_initial_time > 2.0: # Timeout for initial chunk
                           logger.warning("Timeout waiting for initial text chunk.")
                           break # Avoid waiting too long if generation stalls
            except Exception as e:
                 logger.error(f"Error consuming initial stream: {e}")
                 # generation_future might need cancellation here if the error is fatal
                 generation_future.cancel() # Attempt to cancel the background generation
                 return None, "Sorry, I encountered an issue generating the beginning of the response."


            self.generation_count += 1
            end_time = time.time()
            logger.info(f"LLM Gen #{self.generation_count} Initial: '{initial_text}' ({(end_time - start_time)*1000:.1f}ms)")

            # Return the streamer (which continues from where we left off) and the initial text
            # Also return the generation future so the caller can optionally wait for full generation if needed elsewhere
            # (though process_speech_segment currently consumes the streamer fully)
            return streamer, initial_text, generation_future

        except RuntimeError as e:
             if "CUDA out of memory" in str(e):
                 logger.error(f"CUDA OOM Error during Gemma generation: {e}. Attempting to clear cache.")
                 gc.collect()
                 if self.device.type == 'cuda':
                      torch.cuda.empty_cache()
                 return None, "[Generation failed due to OOM]", None
             else:
                  logger.exception(f"Runtime Gemma generation error: {e}")
                  return None, "[Generation runtime error]", None
        except Exception as e:
            logger.exception(f"Gemma streaming error: {e}")
            # Ensure potential background task is handled if it was started
            if 'generation_future' in locals() and generation_future:
                 generation_future.cancel()
            return None, f"Sorry, I couldn’t process that due to an error.", None


class KokoroTTSProcessor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Creating KokoroTTSProcessor instance...")
            cls._instance = cls()
            logger.info("KokoroTTSProcessor instance created.")
        return cls._instance

    def __init__(self):
        if KokoroTTSProcessor._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            KokoroTTSProcessor._instance = self
        try:
            # Use language 'en' or 'eng' typically for English
            self.pipeline = KPipeline(lang_code='en')
            self.default_voice = 'en_us_sarah' # Example US English voice, check available voices
            self.synthesis_count = 0
            self.sample_rate = 24000 # Kokoro default SR
            logger.info(f"Kokoro TTS loaded successfully. Default voice: {self.default_voice}")
        except NameError: # Handles case where KPipeline is the Mock class
            self.pipeline = MockKokoroTTS()
            self.default_voice = 'mock_voice'
            self.synthesis_count = 0
            self.sample_rate = self.pipeline.sample_rate
            logger.info(f"Using Mock Kokoro TTS. Sample rate: {self.sample_rate}")
        except Exception as e:
             logger.exception(f"FATAL: Failed to load Kokoro TTS pipeline: {e}")
             # Fallback to mock if real one fails?
             logger.warning("Falling back to Mock Kokoro TTS due to initialization error.")
             self.pipeline = MockKokoroTTS()
             self.default_voice = 'mock_voice_fallback'
             self.synthesis_count = 0
             self.sample_rate = self.pipeline.sample_rate
             # raise # Or re-raise the exception if TTS is critical

    async def synthesize_speech(self, text: str) -> np.ndarray | None:
        """Synthesizes speech from text using Kokoro TTS."""
        self.synthesis_count += 1
        start_time = time.time()
        if not text or not self.pipeline:
            logger.warning("TTS synthesis skipped: No text or pipeline unavailable.")
            return None
        try:
            logger.info(f"Synthesizing TTS #{self.synthesis_count} for text: '{text[:100]}...'")
            audio_segments = []

            # Run synchronous TTS in an executor thread
            generator = await asyncio.get_event_loop().run_in_executor(
                None, # Use default executor
                lambda: list(self.pipeline( # Consume the generator in the thread
                    text,
                    voice=self.default_voice,
                    speed=1.0,
                    # Use default splitting or provide a robust regex
                    split_pattern=r'[.!?]+' # Split on sentences
                    # split_pattern=r'[\n.!?]+' # Split on newlines and sentences
                 ))
            )

            # Generator yields tuples (start_time, end_time, audio_chunk)
            for _, _, audio_chunk in generator:
                if audio_chunk is not None and audio_chunk.size > 0:
                    # Ensure chunk is float32
                    if audio_chunk.dtype != np.float32:
                        audio_chunk = audio_chunk.astype(np.float32)
                    # Normalize if needed (Kokoro usually outputs floats in [-1, 1])
                    # max_val = np.max(np.abs(audio_chunk))
                    # if max_val > 1.0:
                    #     audio_chunk /= max_val
                    audio_segments.append(audio_chunk)

            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                end_time = time.time()
                duration_s = len(combined_audio) / self.sample_rate
                logger.info(f"TTS #{self.synthesis_count} synthesized: {duration_s:.2f}s ({(end_time - start_time)*1000:.1f}ms)")
                 # Clean up
                del audio_segments
                del generator
                # gc.collect() # Probably overkill here

                return combined_audio
            else:
                logger.warning(f"TTS #{self.synthesis_count} resulted in no audio segments for text: '{text}'")
                return None

        except Exception as e:
            logger.exception(f"TTS synthesis error for text '{text[:50]}...': {e}")
            return None


# ==============================================================================
# WebSocket Client Handler
# ==============================================================================
async def handle_client(websocket: websockets.WebSocketServerProtocol):
    """Handles a single client connection."""
    client_id = websocket.remote_address
    logger.info(f"Client connected: {client_id}")

    # Instantiate per-client components that manage state
    detector = AudioSegmentDetector()
    # Get singleton instances for models/processors
    transcriber = WhisperTranscriber.get_instance()
    gemma_processor = GemmaMultimodalProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance()

    tts_queue = asyncio.Queue() # Queue for audio chunks to send

    async def send_audio_chunks():
        """Sends audio chunks from the tts_queue to the client."""
        while True:
            try:
                audio_data = await tts_queue.get()
                if audio_data is None: # Signal to stop
                    break
                if not websocket.closed:
                     await websocket.send(json.dumps({"audio": base64.b64encode(audio_data).decode('utf-8')}))
                     logger.info(f"Sent audio chunk ({len(audio_data)} bytes) to {client_id}")
                     tts_queue.task_done()
                     await asyncio.sleep(0.01) # Small sleep to prevent tight loop burst
                else:
                     logger.warning(f"WebSocket closed while trying to send audio to {client_id}. Discarding chunk.")
                     tts_queue.task_done() # Mark task done even if not sent
                     # Consider breaking the loop if ws is closed
                     break
            except asyncio.CancelledError:
                 logger.info("Audio sending task cancelled.")
                 break
            except Exception as e:
                 logger.error(f"Error in send_audio_chunks for {client_id}: {e}")
                 # Continue processing queue if possible, or break if fatal
                 if isinstance(e, websockets.ConnectionClosed):
                      break

    async def process_speech_segment(speech_segment: bytes):
        """Processes a single speech segment: Transcribe -> Generate -> Synthesize -> Queue TTS audio."""
        processing_start_time = time.time()
        logger.info(f"Processing segment (len: {len(speech_segment)} bytes)...")
        try:
            await detector.set_tts_playing(True) # Indicate processing/TTS might start soon

            # 1. Transcribe
            transcription = await transcriber.transcribe(speech_segment, detector.sample_rate)
            # Explicitly delete segment after use
            del speech_segment
            gc.collect()
            if self.device.type == 'cuda':
                 torch.cuda.empty_cache()


            if not transcription or not any(c.isalnum() for c in transcription):
                logger.info(f"Skipping empty or non-alphanumeric transcription: '{transcription}'")
                await detector.set_tts_playing(False) # Reset state as we are aborting
                return # Don't proceed further

            logger.info(f"Transcription result: '{transcription}'")

            # 2. Generate Response Stream
            streamer, initial_text, generation_future = await gemma_processor.generate_streaming(transcription)

            if not streamer:
                logger.error(f"LLM generation failed for transcription: '{transcription}'. Sending error TTS.")
                error_audio = await tts_processor.synthesize_speech(initial_text or "Sorry, I couldn't generate a response.") # Use error text if provided
                if error_audio is not None:
                    audio_bytes = (error_audio * 32767).astype(np.int16).tobytes()
                    await tts_queue.put(audio_bytes)
                await detector.set_tts_playing(False)
                return # Abort processing this segment

            # 3. Synthesize and Queue Initial TTS
            logger.info(f"Synthesizing initial TTS for: '{initial_text}'")
            initial_audio = await tts_processor.synthesize_speech(initial_text)
            if initial_audio is not None and initial_audio.size > 0:
                audio_bytes = (initial_audio * 32767).astype(np.int16).tobytes()
                await tts_queue.put(audio_bytes)
                del initial_audio # Clean up array
            else:
                logger.warning(f"Initial TTS synthesis failed or produced empty audio for: '{initial_text}'")

            # 4. Process Remaining Stream and Synthesize/Queue Remaining TTS
            remaining_text = ""
            full_response = initial_text # Start building full response
            async for chunk in streamer:
                remaining_text += chunk
                full_response += chunk
                # Synthesize in chunks based on sentence boundaries for lower latency
                sentence_end_chars = ".!?\n"
                if any(char in chunk for char in sentence_end_chars):
                    # Check if the chunk actually ends with the boundary (or very close)
                    trimmed_chunk = chunk.strip()
                    if trimmed_chunk and trimmed_chunk[-1] in sentence_end_chars:
                         logger.info(f"Synthesizing TTS for sentence chunk: '{remaining_text}'")
                         chunk_audio = await tts_processor.synthesize_speech(remaining_text)
                         if chunk_audio is not None and chunk_audio.size > 0:
                             audio_bytes = (chunk_audio * 32767).astype(np.int16).tobytes()
                             await tts_queue.put(audio_bytes)
                             del chunk_audio
                         else:
                              logger.warning(f"Mid-stream TTS failed for chunk: '{remaining_text}'")
                         remaining_text = "" # Reset for next sentence

            # Synthesize any leftover text after the stream finishes
            if remaining_text:
                 logger.info(f"Synthesizing final TTS chunk: '{remaining_text}'")
                 final_audio = await tts_processor.synthesize_speech(remaining_text)
                 if final_audio is not None and final_audio.size > 0:
                     audio_bytes = (final_audio * 32767).astype(np.int16).tobytes()
                     await tts_queue.put(audio_bytes)
                     del final_audio
                 else:
                    logger.warning(f"Final TTS failed for chunk: '{remaining_text}'")

            # Wait for the background generation task to fully complete (optional, good for resource cleanup)
            if generation_future:
                 try:
                      await generation_future # Wait for model.generate() thread to finish
                      logger.info("LLM generation background task completed.")
                 except asyncio.CancelledError:
                      logger.warning("LLM generation background task was cancelled.")
                 except Exception as e:
                      logger.error(f"Error waiting for LLM generation background task: {e}")


            # 5. Update History
            await gemma_processor._update_history(transcription, full_response)
            logger.info("Conversation history updated.")

            # Explicit cleanup
            del streamer
            del transcription
            del full_response
            gc.collect()
            if self.device.type == 'cuda':
                 torch.cuda.empty_cache()

        except asyncio.CancelledError:
            # This happens when a new segment interrupts the current one via cancel_current_processing
            logger.info(f">>> Processing task for segment (started at {processing_start_time}) was cancelled by interrupt. <<<")
            # No need to set tts_playing=False here, cancel_current_processing handles it.
            # Also, don't send error TTS, as it was intentionally interrupted.
        except Exception as e:
            logger.exception(f"Error during speech segment processing: {e}")
            # Attempt to send an error message via TTS
            try:
                error_audio = await tts_processor.synthesize_speech("Sorry, an internal error occurred.")
                if error_audio is not None:
                    audio_bytes = (error_audio * 32767).astype(np.int16).tobytes()
                    await tts_queue.put(audio_bytes) # Use the queue
            except Exception as tts_err:
                logger.error(f"Failed to send error TTS: {tts_err}")
            # Ensure TTS state is reset after error
            await detector.set_tts_playing(False)
        finally:
            # Ensure TTS state is reset if processing finishes normally (not cancelled)
            # Cancellation handles its own state reset.
            if not asyncio.current_task().cancelled():
                 await detector.set_tts_playing(False)
            processing_end_time = time.time()
            logger.info(f"Segment processing finished in {(processing_end_time - processing_start_time):.2f}s")


    async def detect_speech_segments(detector: AudioSegmentDetector):
        """Continuously checks the detector queue for new segments and starts processing."""
        while True:
            try:
                speech_segment = await detector.get_next_segment()
                if speech_segment:
                    logger.info(f"Dequeued segment (size: {len(speech_segment)}), starting processing task...")
                    # NOTE: Cancellation of previous task is handled in detector.add_audio
                    # Create the processing task
                    task = asyncio.create_task(process_speech_segment(speech_segment))
                    # Store the reference to the *new* task so it can be cancelled
                    await detector.set_current_processing_task(task)
                else:
                     # No segment, sleep briefly to avoid busy-waiting
                     await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                logger.info("Speech detection loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in speech detection loop: {e}")
                await asyncio.sleep(1) # Sleep longer after an error

    async def receive_data(websocket, detector: AudioSegmentDetector, gemma_processor: GemmaMultimodalProcessor):
        """Receives audio and image data from the client."""
        async for message in websocket:
            try:
                data = json.loads(message)

                # Handle realtime media chunks (audio/image)
                if "realtime_input" in data and "media_chunks" in data["realtime_input"]:
                    # Check if TTS is playing before processing new input that might interrupt
                    is_tts_active = await detector.is_tts_currently_playing()

                    for chunk in data["realtime_input"]["media_chunks"]:
                        mime_type = chunk.get("mime_type")
                        chunk_data_b64 = chunk.get("data")
                        if not mime_type or not chunk_data_b64:
                            continue

                        try:
                             decoded_data = base64.b64decode(chunk_data_b64)
                        except Exception as e:
                             logger.warning(f"Failed to decode base64 data for mime_type {mime_type}: {e}")
                             continue

                        if mime_type == "audio/pcm":
                            await detector.add_audio(decoded_data)
                        elif mime_type == "image/jpeg" or mime_type == "image/png":
                             if not is_tts_active: # Only process images if TTS is not playing
                                  logger.info(f"Received image chunk ({mime_type}, {len(decoded_data)} bytes)")
                                  await gemma_processor.set_image(decoded_data)
                             else:
                                  logger.info("Ignoring received image because TTS is active.")

                # Handle standalone image message
                elif "image" in data:
                    is_tts_active = await detector.is_tts_currently_playing()
                    if not is_tts_active:
                         image_data_b64 = data["image"]
                         logger.info(f"Received standalone image message.")
                         try:
                              decoded_data = base64.b64decode(image_data_b64)
                              await gemma_processor.set_image(decoded_data)
                         except Exception as e:
                              logger.warning(f"Failed to decode base64 image data: {e}")
                    else:
                         logger.info("Ignoring received image message because TTS is active.")

                elif "setup" in data:
                     logger.info(f"Received setup message from {client_id}: {data['setup']}")
                     # Handle any setup config here if needed

                # Handle other message types if necessary
                # elif "text_input" in data:
                #    text = data["text_input"]
                #    # Simulate a segment for text input? Or handle differently?
                #    # This requires deciding how text-only input integrates with the VAD flow.

            except json.JSONDecodeError:
                 logger.warning(f"Received non-JSON message from {client_id}: {message[:100]}...")
            except asyncio.CancelledError:
                 logger.info(f"Receive data task for {client_id} cancelled.")
                 raise # Re-raise cancellation
            except Exception as e:
                 logger.error(f"Error processing message from {client_id}: {e}")
                 # Don't crash the handler, log and continue listening

    async def send_keepalive(websocket):
        """Sends periodic pings to keep the connection alive."""
        while True:
            try:
                await websocket.ping()
                await asyncio.sleep(20) # Send ping every 20 seconds
            except asyncio.CancelledError:
                logger.info("Keepalive task cancelled.")
                break
            except websockets.ConnectionClosed:
                logger.info("Keepalive task stopping: Connection closed.")
                break
            except Exception as e:
                 logger.error(f"Keepalive error: {e}")
                 break # Stop keepalive on other errors


    # --- Main client handling logic ---
    receive_task = asyncio.create_task(receive_data(websocket, detector, gemma_processor))
    detect_task = asyncio.create_task(detect_speech_segments(detector))
    keepalive_task = asyncio.create_task(send_keepalive(websocket))
    audio_send_task = asyncio.create_task(send_audio_chunks())

    # Wait for any task to complete (normally receive_task on disconnect)
    done, pending = await asyncio.wait(
        [receive_task, detect_task, keepalive_task, audio_send_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    logger.info(f"Client {client_id} disconnected or task finished. Cleaning up...")

    # Cancel all other pending tasks for this client
    for task in pending:
        task.cancel()
        try:
             # Wait briefly for tasks to acknowledge cancellation
             await asyncio.wait_for(task, timeout=0.5)
        except asyncio.CancelledError:
             pass # Expected
        except asyncio.TimeoutError:
             logger.warning(f"Task {task.get_name()} did not cancel quickly.")
        except Exception as e:
             logger.error(f"Error during task cleanup: {e}")


    # Explicitly cancel any ongoing processing task managed by the detector
    await detector.cancel_current_processing()

    # Signal the audio sending queue to stop
    await tts_queue.put(None)
    try:
        # Wait for the audio send task to finish processing remaining items + the None signal
        await asyncio.wait_for(audio_send_task, timeout=1.0)
    except asyncio.TimeoutError:
        logger.warning("Timeout waiting for audio send task to finish cleanup.")
    except Exception as e:
        logger.error(f"Error waiting for audio_send_task completion: {e}")


    # Attempt final cleanup of GPU memory if applicable
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Performed final GPU cache clearing.")

    logger.info(f"Finished cleanup for client {client_id}.")


# ==============================================================================
# Main Server Start
# ==============================================================================
async def main():
    # Initialize singletons eagerly before starting the server
    logger.info("Pre-loading models...")
    try:
        WhisperTranscriber.get_instance()
        GemmaMultimodalProcessor.get_instance()
        KokoroTTSProcessor.get_instance()
        logger.info("Models pre-loaded successfully.")
    except Exception as e:
         logger.exception("FATAL: Failed to initialize models during startup. Exiting.")
         sys.exit(1)

    # Clear cache after loading
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Cleared GPU cache after model loading.")


    host = "0.0.0.0"
    port = 9073
    logger.info(f"Starting WebSocket server on {host}:{port}")

    # Adjust serve parameters for robustness
    async with websockets.serve(
        handle_client,
        host,
        port,
        ping_interval=20,      # Send pings every 20s
        ping_timeout=60,       # Disconnect if no pong received within 60s
        close_timeout=10,      # Time to wait for graceful close handshake
        # Increase queue size if many clients connect rapidly (default is 100)
        # backlog=100,
        # Set max message size if needed (default is 1MB)
        max_size=2*1024*1024, # 2MB limit
        # Increase read/write buffer limits if dealing with large chunks often
        # read_limit=2**20, # 1MB
        # write_limit=2**20 # 1MB
        ):
        await asyncio.Future() # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt).")
    except Exception as e:
         logger.exception(f"Server encountered critical error: {e}")
