import asyncio
import json
import websockets
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, Gemma3ForConditionalGeneration, TextIteratorStreamer
from threading import Thread
import numpy as np
import logging
import sys
import io
from PIL import Image
import time
import os
from datetime import datetime
import re

# Import Kokoro TTS library
# Ensure Kokoro is installed: pip install kokoro-tts
try:
    from kokoro import KPipeline
except ImportError:
    print("Error: Kokoro TTS library not found. Please install it: pip install kokoro-tts")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__) # Use __name__ for logger

class AudioSegmentDetector:
    """Detects speech segments based on audio energy levels"""

    def __init__(self,
                 sample_rate=16000,
                 # --- Tunable Parameters ---
                 energy_threshold=0.015,  # Adjust based on mic sensitivity and background noise
                 silence_duration=0.7,    # Shorter silence might make it more responsive
                 min_speech_duration=0.6, # Shorter duration detects quicker utterances
                 max_speech_duration=15): # Max duration before forced segmentation

        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        # Convert durations to samples (and bytes for silence comparison)
        self.silence_samples_threshold_bytes = int(silence_duration * sample_rate * 2) # * 2 for 16-bit bytes
        self.min_speech_samples_bytes = int(min_speech_duration * sample_rate * 2)
        self.max_speech_samples_bytes = int(max_speech_duration * sample_rate * 2)

        # Internal state
        self.audio_buffer = bytearray()
        self.is_speech_active = False
        self.silence_counter_bytes = 0 # Track silence in bytes
        self.speech_start_idx_bytes = 0 # Index in bytes
        self.lock = asyncio.Lock()
        self.segment_queue = asyncio.Queue()

        # Counters
        self.segments_detected = 0

        # TTS playback and generation control
        self._tts_playing = False # Use property for thread safety if needed, but locks suffice here
        self._tts_lock = asyncio.Lock()
        self._current_generation_task = None
        self._current_tts_task = None
        self._task_lock = asyncio.Lock()
        logger.info(f"AudioSegmentDetector initialized: energy_threshold={energy_threshold}, silence_duration={silence_duration}s, min_speech={min_speech_duration}s")


    async def is_tts_playing(self):
        """Check TTS playback state"""
        async with self._tts_lock:
            return self._tts_playing

    async def set_tts_playing(self, is_playing: bool):
        """Set TTS playback state"""
        async with self._tts_lock:
            if self._tts_playing != is_playing:
                logger.info(f"Setting TTS playing state to: {is_playing}")
                self._tts_playing = is_playing

    async def cancel_current_tasks(self):
        """Cancel any ongoing generation and TTS tasks"""
        async with self._task_lock:
            cancelled_tasks = []
            if self._current_generation_task and not self._current_generation_task.done():
                logger.info("Attempting to cancel current generation task...")
                self._current_generation_task.cancel()
                cancelled_tasks.append(self._current_generation_task)
                self._current_generation_task = None # Clear immediately

            if self._current_tts_task and not self._current_tts_task.done():
                logger.info("Attempting to cancel current TTS task...")
                self._current_tts_task.cancel()
                cancelled_tasks.append(self._current_tts_task)
                self._current_tts_task = None # Clear immediately

            if cancelled_tasks:
                try:
                    # Wait briefly for tasks to acknowledge cancellation
                    await asyncio.wait(cancelled_tasks, timeout=0.5)
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for tasks to cancel.")
                except asyncio.CancelledError:
                    logger.info("Cancellation task itself was cancelled.") # Should not happen often here
                finally:
                     # Ensure flags are reset even if waiting fails/is cancelled
                    await self.set_tts_playing(False)
            else:
                 # If no tasks were running, ensure state is consistent
                 await self.set_tts_playing(False)

            # Reset task references *after* cancellation attempt
            self._current_generation_task = None
            self._current_tts_task = None
            logger.info("Cleared current generation and TTS task references.")


    async def set_current_tasks(self, generation_task=None, tts_task=None):
        """Set current generation and TTS tasks safely."""
        async with self._task_lock:
            # Only update if the provided task is not None
            if generation_task is not None:
                # If there's an old task still tracked, cancel it first
                if self._current_generation_task and not self._current_generation_task.done():
                    logger.warning("Setting new generation task while old one existed. Cancelling old.")
                    self._current_generation_task.cancel()
                self._current_generation_task = generation_task
            if tts_task is not None:
                 # If there's an old task still tracked, cancel it first
                if self._current_tts_task and not self._current_tts_task.done():
                    logger.warning("Setting new TTS task while old one existed. Cancelling old.")
                    self._current_tts_task.cancel()
                self._current_tts_task = tts_task

    # --- CORRECTED add_audio Method ---
    async def add_audio(self, audio_bytes: bytes):
        """Add audio data to the buffer and check for speech segments"""
        if not audio_bytes:
            return None

        async with self.lock:
            self.audio_buffer.extend(audio_bytes)
            buffer_len_bytes = len(self.audio_buffer)
            added_bytes_len = len(audio_bytes)

            # Convert recent audio to numpy for energy analysis
            try:
                # Ensure we have enough bytes for a valid float conversion (at least one 16-bit sample)
                if added_bytes_len < 2:
                     return None
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            except ValueError as e:
                 logger.error(f"Error converting audio bytes to numpy array: {e}. Length: {added_bytes_len}")
                 return None


            # Calculate audio energy (root mean square)
            if len(audio_array) > 0:
                energy = np.sqrt(np.mean(audio_array**2))

                # --- Speech detection logic ---
                if not self.is_speech_active and energy > self.energy_threshold:
                    # Speech start detected
                    self.is_speech_active = True
                    # Start index is roughly where the energetic audio *began* in the buffer
                    self.speech_start_idx_bytes = max(0, buffer_len_bytes - added_bytes_len)
                    self.silence_counter_bytes = 0
                    # logger.debug(f"Speech start detected (energy: {energy:.6f}) at index {self.speech_start_idx_bytes}")

                elif self.is_speech_active:
                    current_speech_len_bytes = buffer_len_bytes - self.speech_start_idx_bytes

                    # First, check if silence ends the segment
                    if energy <= self.energy_threshold:
                        # Potential end of speech - accumulating silence
                        self.silence_counter_bytes += added_bytes_len

                        # Check if enough silence (in bytes) to end speech segment
                        if self.silence_counter_bytes >= self.silence_samples_threshold_bytes:
                            speech_end_idx_bytes = buffer_len_bytes - self.silence_counter_bytes
                            # Ensure start index is validly before end index
                            if self.speech_start_idx_bytes < speech_end_idx_bytes:
                                speech_segment_bytes = bytes(self.audio_buffer[self.speech_start_idx_bytes : speech_end_idx_bytes])
                                segment_duration_s = len(speech_segment_bytes) / 2 / self.sample_rate

                                # Reset state *before* processing segment
                                is_speech_active_before_reset = self.is_speech_active
                                self.is_speech_active = False # End of speech detected
                                self.silence_counter_bytes = 0
                                # Trim buffer efficiently: keep only audio *after* the detected silence started
                                self.audio_buffer = self.audio_buffer[speech_end_idx_bytes:]
                                self.speech_start_idx_bytes = 0 # Reset relative start index

                                # Only process if speech segment is long enough and speech *was* active
                                if is_speech_active_before_reset and len(speech_segment_bytes) >= self.min_speech_samples_bytes:
                                    self.segments_detected += 1
                                    logger.info(f"Speech segment detected (silence): {segment_duration_s:.2f}s")
                                    await self.segment_queue.put(speech_segment_bytes)
                                    # Allow buffer maintenance to run below
                                elif is_speech_active_before_reset:
                                    logger.info(f"Speech segment too short ({segment_duration_s:.2f}s), discarding.")

                            else:
                                # Edge case: silence detected but indices seem invalid, reset defensively
                                logger.warning("Silence end detected with invalid indices, resetting state.")
                                self.is_speech_active = False
                                self.silence_counter_bytes = 0
                                self.speech_start_idx_bytes = 0
                                # Keep only the recent silence part in buffer as a guess
                                self.audio_buffer = self.audio_buffer[max(0, buffer_len_bytes - self.silence_counter_bytes):]

                    else: # energy > self.energy_threshold
                        # Continued speech, reset silence counter
                        self.silence_counter_bytes = 0

                    # --- Check for Max Duration ---
                    # This check happens *after* the energy/silence check for the current chunk.
                    # It should only trigger if speech is *still* considered active.
                    if self.is_speech_active and current_speech_len_bytes >= self.max_speech_samples_bytes:
                        logger.info(f"Max duration check: current_len={current_speech_len_bytes}, max_len={self.max_speech_samples_bytes}")
                        # Extract the max duration segment
                        segment_end_idx = self.speech_start_idx_bytes + self.max_speech_samples_bytes
                        speech_segment_bytes = bytes(self.audio_buffer[self.speech_start_idx_bytes : segment_end_idx])
                        segment_duration_s = len(speech_segment_bytes) / 2 / self.sample_rate

                        # Update buffer: keep audio *after* the extracted segment
                        self.audio_buffer = self.audio_buffer[segment_end_idx:]

                        # Reset relative start index for the remaining buffer content
                        self.speech_start_idx_bytes = 0
                        # Reset silence counter as we forced a segment end
                        self.silence_counter_bytes = 0
                        # Keep is_speech_active = True, as speech likely continues immediately

                        self.segments_detected += 1
                        logger.info(f"Max duration speech segment extracted: {segment_duration_s:.2f}s")
                        await self.segment_queue.put(speech_segment_bytes)
                        # Allow buffer maintenance to run below


            # --- Buffer Maintenance ---
            # Limit buffer size to avoid excessive memory usage
            # Keep roughly max_speech_duration + silence_duration worth of audio (in bytes)
            max_buffer_len_bytes = self.max_speech_samples_bytes + self.silence_samples_threshold_bytes
            current_buffer_len_bytes = len(self.audio_buffer) # Get length again after potential trimming
            if current_buffer_len_bytes > max_buffer_len_bytes:
                over = current_buffer_len_bytes - max_buffer_len_bytes
                self.audio_buffer = self.audio_buffer[over:]
                # Adjust speech_start_idx relative to the truncated buffer ONLY if speech is active
                if self.is_speech_active:
                    self.speech_start_idx_bytes = max(0, self.speech_start_idx_bytes - over)


        return None # Indicate no segment completed *and returned* in this specific call
    # --- END of corrected add_audio Method ---

    async def get_next_segment(self):
        """Get the next available speech segment from the queue"""
        try:
            # Use a short timeout to prevent blocking indefinitely if queue is empty
            return await asyncio.wait_for(self.segment_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None

class WhisperTranscriber:
    """Handles speech transcription using Whisper large-v3 model with pipeline"""
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Creating WhisperTranscriber instance...")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if WhisperTranscriber._instance is not None:
             raise Exception("Singleton already instantiated")
        # Use GPU for transcription
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Whisper using device: {self.device}")

        # Set torch dtype based on device
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
        logger.info(f"Whisper using torch dtype: {self.torch_dtype}")

        # Load model and processor
        model_id = "openai/whisper-large-v3" # Using large-v3, remove -turbo if not needed/available
        logger.info(f"Loading Whisper model: {model_id}...")
        load_start_time = time.time()

        try:
            # Load model
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            self.model.to(self.device)
            self.model.eval() # Set model to evaluation mode

            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)

            # Create pipeline with optimizations
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
                # Pipeline arguments for potential speedup/efficiency
                chunk_length_s=20,       # Process audio in 20s chunks
                stride_length_s=5,      # Overlap chunks by 5s for context
                batch_size=8             # Adjust based on GPU memory (e.g., 4, 8, 16)
            )
            load_end_time = time.time()
            logger.info(f"Whisper model loaded in {load_end_time - load_start_time:.2f} seconds.")

        except Exception as e:
            logger.error(f"FATAL: Failed to load Whisper model '{model_id}'. Error: {e}", exc_info=True)
            # Depending on desired behavior, either raise e, exit, or set self.pipe = None
            raise e # Re-raise to stop the application if model loading fails

        # Counter
        self.transcription_count = 0


    async def transcribe(self, audio_bytes: bytes, sample_rate=16000):
        """Transcribe audio bytes to text using the pipeline"""
        if not audio_bytes or len(audio_bytes) < sample_rate * 0.1 * 2: # Ignore very short audio (e.g., < 0.1s)
            logger.warning("Skipping transcription for very short audio segment.")
            return ""

        start_time = time.perf_counter()
        try:
            # Convert PCM bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Run transcription in executor thread
            result = await asyncio.get_event_loop().run_in_executor(
                None, # Use default executor
                lambda: self.pipe(
                    {"array": audio_array, "sampling_rate": sample_rate},
                    # Generation arguments for Whisper
                    generate_kwargs={
                        "task": "transcribe",
                        "language": "english", # Force English
                        "temperature": 0.0,    # Deterministic output
                        # "do_sample": False, # Implied by temp 0.0
                    }
                )
            )

            # Extract the text from the result
            text = result.get("text", "").strip() if result else ""

            end_time = time.perf_counter()
            duration = end_time - start_time
            self.transcription_count += 1
            logger.info(f"Transcription #{self.transcription_count} completed in {duration:.2f}s: '{text}'")

            return text

        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.error(f"Transcription error after {duration:.2f}s: {e}", exc_info=True)
            return ""


class GemmaMultimodalProcessor:
    """Handles multimodal generation using Gemma 3 model"""
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Creating GemmaMultimodalProcessor instance...")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if GemmaMultimodalProcessor._instance is not None:
             raise Exception("Singleton already instantiated")
        # Use GPU for generation
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Gemma using device: {self.device}")

        # Load model and processor
        model_id = "google/gemma-3-4b-it"
        logger.info(f"Loading Gemma model: {model_id}...")
        load_start_time = time.time()

        try:
            # Load model with 8-bit quantization for memory efficiency
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto", # Let HF decide device placement
                # load_in_8bit=True,  # 8-bit can sometimes be slower, try without first if speed is main issue
                # torch_dtype=torch.bfloat16 # Use bfloat16 if supported and not using 8bit
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32 # float16 often better balance
            )
            self.model.eval() # Set model to evaluation mode

            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)

            load_end_time = time.time()
            logger.info(f"Gemma model loaded in {load_end_time - load_start_time:.2f} seconds.")

        except Exception as e:
            logger.error(f"FATAL: Failed to load Gemma model '{model_id}'. Error: {e}", exc_info=True)
            raise e # Re-raise

        # Cache for most recent image
        self.last_image = None
        self.last_image_timestamp = 0
        self.lock = asyncio.Lock() # Lock for accessing image and history

        # Message history management
        self.message_history = []
        self.max_history_len = 4 # Keep last N turns (1 turn = 1 user + 1 assistant)

        # Counter
        self.generation_count = 0

        # Store pending user message for history update after streaming
        self._pending_user_message = None


    async def set_image(self, image_data: bytes):
        """Cache the most recent image received, resizing it, and clearing history."""
        if not image_data:
            return False
        start_time = time.perf_counter()
        async with self.lock:
            try:
                # Convert image data to PIL Image
                image = Image.open(io.BytesIO(image_data)).convert("RGB") # Ensure RGB

                # Resize to 75% of original size for potentially faster processing
                if image.size[0] > 1024 or image.size[1] > 1024: # Only resize larger images
                    new_size = (int(image.size[0] * 0.75), int(image.size[1] * 0.75))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                    logger.info(f"Resized image from {image.size} to {new_size}")

                # Clear message history when new image context is set
                if self.message_history:
                    logger.info("New image received, clearing conversation history.")
                    self.message_history = []

                self.last_image = image
                self.last_image_timestamp = time.time()
                end_time = time.perf_counter()
                logger.info(f"Processed and cached new image in {end_time - start_time:.3f}s.")
                return True
            except Exception as e:
                end_time = time.perf_counter()
                logger.error(f"Error processing image in {end_time - start_time:.3f}s: {e}", exc_info=True)
                self.last_image = None # Ensure last image is cleared on error
                return False

    def _build_prompt_and_messages(self, text: str):
        """Build messages array for the model, including system prompt and history."""
        # System Prompt - Concise and focused
        system_message = {
            "role": "system",
            "content": """You are a concise and conversational AI assistant.
Respond naturally based on the user's input and the provided image.
Keep responses short (1-3 sentences) and suitable for spoken delivery.
If the user's query isn't about the image, respond conversationally without forcing image descriptions.
If unsure, ask for clarification politely."""
        }

        # Start with system prompt + history
        messages = [system_message] + self.message_history

        # Add current user message with image (if available)
        current_user_content = []
        # if self.last_image: # Image is handled by processor call, not directly in template for Gemma 3
        #     current_user_content.append({"type": "image"}) # Placeholder for processor
        current_user_content.append({"type": "text", "text": text})

        messages.append({"role": "user", "content": current_user_content})

        # Prepare prompt string for the processor
        try:
             # Let processor handle image and text separately
             prompt = self.processor.apply_chat_template(
                 messages,
                 add_generation_prompt=True, # Important for instruct models
                 tokenize=False # Get the prompt string
             )

             inputs = self.processor(
                 text=prompt,
                 images=self.last_image, # Pass the actual image here
                 return_tensors="pt"
             )

             return inputs.to(self.model.device), text # Return processed inputs and original text

        except Exception as e:
             logger.error(f"Error applying chat template or processing inputs: {e}", exc_info=True)
             return None, text # Indicate error


    def _update_history(self, user_text: str, assistant_response: str):
        """Update message history, trimming old messages."""
        if not user_text or not assistant_response: # Don't add empty exchanges
             return

        # Add user message (text only for history)
        self.message_history.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
        # Add assistant response
        self.message_history.append({"role": "assistant", "content": [{"type": "text", "text": assistant_response}]})

        # Trim history: keep last `max_history_len` turns (user + assistant = 1 turn)
        max_messages = self.max_history_len * 2
        if len(self.message_history) > max_messages:
            self.message_history = self.message_history[-max_messages:]
            # logger.debug(f"Trimmed history to {len(self.message_history)} messages.")


    async def generate_streaming(self, text: str):
        """Generate response stream using the latest image and text input."""
        start_time = time.perf_counter()
        async with self.lock: # Lock to prevent concurrent access to image/history
            image_available = self.last_image is not None
            if not image_available:
                logger.warning("No image available for multimodal generation. Proceeding with text-only.")
                # Note: The prompt building will just not include the image if self.last_image is None

            # Build messages and process inputs
            inputs, original_user_text = self._build_prompt_and_messages(text)
            if inputs is None:
                 logger.error("Failed to build prompt/messages.")
                 # Return a streamer that yields nothing and the start time
                 empty_streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
                 empty_streamer.put(None) # Signal end immediately
                 return empty_streamer, start_time


            # Store user text for history update later
            self._pending_user_message = original_user_text

            input_len = inputs["input_ids"].shape[-1]

            # Create a streamer for token-by-token generation
            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_special_tokens=True,
                skip_prompt=True # Don't stream the input prompt back
            )

            # Generation arguments
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=100, # Reduced for faster, concise responses
                do_sample=True,
                temperature=0.6,    # Slightly lower temp for less randomness
                use_cache=True,     # Important for generation speed
                streamer=streamer,
            )

            # Run generation in a separate thread to avoid blocking asyncio loop
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True)
            thread.start()

            self.generation_count += 1
            logger.info(f"Gemma generation #{self.generation_count} started (Image available: {image_available})...")
            # Return the streamer immediately for processing in the main loop
            # The calling function will handle iterating through it.
            return streamer, start_time # Return streamer and start time for duration calc

    async def complete_generation_and_update_history(self, streamer: TextIteratorStreamer, start_time: float):
        """Consume the rest of the streamer, update history, and return full text."""
        full_response = ""
        if not streamer:
            return ""

        logger.info("Collecting remaining generation text...")
        stream_start_time = time.perf_counter()
        try:
            # This loop consumes the generator
            for text_chunk in streamer:
                if text_chunk is None: # Handle potential early end signal
                    break
                full_response += text_chunk
                # Optional: yield chunks if further processing needs streaming output

            end_time = time.perf_counter()
            # Duration calculation should use the overall start_time passed in
            total_duration = end_time - start_time
            stream_duration = end_time - stream_start_time
            logger.info(f"Gemma generation #{self.generation_count} finished.")
            logger.info(f"  - Stream consumption: {stream_duration:.2f}s")
            logger.info(f"  - Total generation time (incl. setup): {total_duration:.2f}s")
            logger.info(f"  - Full response length: {len(full_response)} chars.")


        except Exception as e:
            logger.error(f"Error consuming generation streamer: {e}", exc_info=True)
            # History might be incomplete if error occurs mid-stream

        # Update history needs the lock
        async with self.lock:
             if self._pending_user_message:
                 self._update_history(self._pending_user_message, full_response)
                 self._pending_user_message = None # Clear pending message
             else:
                  logger.warning("No pending user message found for history update.")

        return full_response


class KokoroTTSProcessor:
    """Handles text-to-speech conversion using Kokoro model"""
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Creating KokoroTTSProcessor instance...")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if KokoroTTSProcessor._instance is not None:
             raise Exception("Singleton already instantiated")
        logger.info("Initializing Kokoro TTS processor...")
        self.pipeline = None
        self.sampling_rate = None # Will be set after loading
        try:
            # Initialize Kokoro TTS pipeline
            # 'en' for English, 'zh' for Chinese, 'ja' for Japanese, 'a' for auto-detect (experimental)
            lang_code = 'en'
            self.pipeline = KPipeline(lang_code=lang_code) # Use 'en' for primarily English

            # Set a default English voice (check available voices in Kokoro docs/source)
            self.default_voice = 'en_sarah' # Example English voice, replace if needed
            # self.default_voice = 'af_sarah' # Seems african accent
            self.sampling_rate = self.pipeline.hps.data.sampling_rate


            logger.info(f"Kokoro TTS processor initialized successfully with lang='{lang_code}', voice='{self.default_voice}', sample_rate={self.sampling_rate}.")
            # Counter
            self.synthesis_count = 0
        except Exception as e:
            logger.error(f"FATAL: Error initializing Kokoro TTS: {e}", exc_info=True)
            self.pipeline = None # Ensure pipeline is None on failure
            raise e # Re-raise

    async def _synthesize_internal(self, text: str, split_pattern: str | None):
        """Internal synthesis logic"""
        if not text or not self.pipeline:
            logger.warning(f"Skipping TTS: Text empty ({not text}), Pipeline not ready ({not self.pipeline})")
            return None

        start_time = time.perf_counter()
        try:
            logger.info(f"Synthesizing TTS for text (approx {len(text)} chars)... Pattern: '{split_pattern}'")

            # Run TTS in a thread pool executor to avoid blocking async loop
            audio_segments = []

            # Use the executor to run the TTS pipeline
            # Ensure lambda captures current values if needed, though here it's direct
            generator = await asyncio.get_event_loop().run_in_executor(
                None, # Default executor
                lambda: self.pipeline(
                    text=text,
                    voice=self.default_voice,
                    speed=1.0, # Normal speed
                    split_pattern=split_pattern # Use provided split pattern
                )
            )

            # Process all generated segments from the generator
            segment_count = 0
            total_samples = 0
            for _generated_sentence, _phonemes, audio_chunk in generator:
                if audio_chunk is not None and len(audio_chunk) > 0:
                    audio_segments.append(audio_chunk)
                    segment_count += 1
                    total_samples += len(audio_chunk)
                else:
                    # This can happen normally if a "split" part is just punctuation etc.
                    # logger.debug("TTS generator yielded empty/None audio chunk.")
                    pass


            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                end_time = time.perf_counter()
                duration = end_time - start_time
                self.synthesis_count += 1
                audio_duration_s = total_samples / self.sampling_rate if self.sampling_rate else 0
                logger.info(f"TTS synthesis #{self.synthesis_count} complete in {duration:.2f}s ({segment_count} segments, {total_samples} samples, ~{audio_duration_s:.2f}s audio).")
                return combined_audio
            else:
                logger.warning("TTS synthesis resulted in no audio segments for the given text.")
                return None

        except asyncio.CancelledError:
             logger.info("TTS synthesis task was cancelled.")
             raise # Re-raise cancellation
        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.error(f"TTS synthesis error after {duration:.2f}s: {e}", exc_info=True)
            return None


    async def synthesize_initial_speech(self, text: str):
        """Synthesize the initial part of the text quickly (less splitting)."""
        # Split only on major punctuation for faster initial response
        # Adjust pattern for robustness - ensure it handles various sentence endings
        # Use raw string r'' for regex patterns
        split_pattern = r'[\n.!?]+' # Split on newline or major terminators
        return await self._synthesize_internal(text, split_pattern)


    async def synthesize_remaining_speech(self, text: str):
        """Synthesize the remaining text with more natural splitting."""
        # Split on more punctuation types for potentially better prosody on longer text
        split_pattern = r'[.!?,;:]+' # Split on common punctuation
        return await self._synthesize_internal(text, split_pattern)


# --- WebSocket Handler ---

async def handle_client(websocket, path):
    """Handles WebSocket client connection, processing audio and images."""
    client_ip = websocket.remote_address
    logger.info(f"Client connected from {client_ip}")

    # Initialize components for this connection
    # Models are singletons, detector is per-connection
    detector = AudioSegmentDetector()
    transcriber = WhisperTranscriber.get_instance()
    gemma_processor = GemmaMultimodalProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance()

    if not all([transcriber, gemma_processor, tts_processor, tts_processor.pipeline]):
         logger.error("One or more processing components failed to initialize. Closing connection.")
         try:
             await websocket.close(code=1011, reason="Server component initialization failed")
         except websockets.exceptions.ConnectionClosed:
             pass # Already closed
         return

    # Task for processing detected speech segments
    speech_processing_task = None # Track the current processing workflow

    async def detect_and_process_speech():
        """Coroutine to detect speech and manage the processing workflow."""
        nonlocal speech_processing_task # Allow modification of the outer task variable
        while True:
            try:
                # Check for next segment without blocking indefinitely
                speech_segment = await detector.get_next_segment()

                if speech_segment:
                    # --- New Speech Detected ---
                    segment_duration = len(speech_segment) / 2 / detector.sample_rate
                    logger.info(f"--- New speech segment ({segment_duration:.2f}s) ready for processing ---")

                    # 1. Cancel any ongoing TTS/Generation immediately
                    # Check if a workflow is actually running before cancelling
                    if speech_processing_task and not speech_processing_task.done():
                        logger.info("New speech detected, cancelling previous processing workflow...")
                        speech_processing_task.cancel()
                        try:
                            await speech_processing_task # Allow cleanup within the task
                        except asyncio.CancelledError:
                            logger.info("Previous workflow cancelled successfully.")
                        except Exception as e:
                            logger.error(f"Error awaiting cancelled workflow task: {e}")
                    # Also explicitly cancel tasks managed by detector state
                    await detector.cancel_current_tasks() # Ensures tts_playing is False

                    # 2. Send interrupt signal to client (optional, client needs to handle it)
                    logger.info("Sending interrupt signal to client.")
                    try:
                        await websocket.send(json.dumps({"interrupt": True}))
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("Connection closed while sending interrupt.")
                        break # Exit loop if connection is closed

                    # 3. Start processing the new segment
                    # Set TTS playing flag early to block incoming images during processing
                    await detector.set_tts_playing(True)

                    # Transcribe the audio segment
                    transcription = await transcriber.transcribe(speech_segment)

                    # Filter empty or noise-like transcriptions
                    if not transcription or not any(c.isalnum() for c in transcription):
                        logger.info(f"Skipping empty/non-alphanumeric transcription: '{transcription}'")
                        await detector.set_tts_playing(False) # Reset flag if skipping
                        continue # Get next segment

                    words = transcription.split()
                    if len(words) <= 1 and len(transcription) < 10 : # Filter very short utterances
                         logger.info(f"Skipping short transcription: '{transcription}'")
                         await detector.set_tts_playing(False)
                         continue

                    # Use re.fullmatch for exact pattern matching from start to end
                    filler_patterns = [ r'(um+|uh+|ah+|oh+|hm+|mhm+|hmm+)', r'(okay|yes|no|yeah|nah)', r'(bye+)' ]
                    # Check if the lowercased transcription exactly matches any filler pattern
                    if any(re.fullmatch(pattern, transcription.lower()) for pattern in filler_patterns):
                        logger.info(f"Skipping filler sound/word: '{transcription}'")
                        await detector.set_tts_playing(False)
                        continue

                    logger.info(f"User said: '{transcription}'")

                    # --- Start Generation and TTS Workflow ---
                    # Define the workflow as an inner async function
                    async def generation_tts_workflow_inner(user_input: str):
                        generation_streamer = None
                        initial_text = ""
                        remaining_text = ""
                        gen_task = None
                        tts_task = None
                        collect_task = None
                        initial_audio_sent = False
                        remaining_audio_sent = False

                        try:
                            # --- Generation ---
                            logger.info("Starting Gemma generation...")
                            streamer_future = asyncio.create_task(
                                gemma_processor.generate_streaming(user_input)
                            )
                            await detector.set_current_tasks(generation_task=streamer_future)

                            generation_streamer, gen_start_time = await streamer_future
                            # Don't clear task yet, need streamer for collection

                            if not generation_streamer:
                                 logger.error("Generation failed to return a streamer.")
                                 return # Exit workflow

                            # --- Initial Text Collection & TTS ---
                            logger.info("Collecting initial text from streamer...")
                            initial_text_list = []
                            min_initial_chars = 40
                            sentence_end_pattern = re.compile(r'[.!?]')
                            collected_chars = 0
                            try:
                                async for chunk in generation_streamer: # Iterate directly
                                     if chunk is None: break # End signal
                                     initial_text_list.append(chunk)
                                     collected_chars += len(chunk)
                                     if collected_chars >= min_initial_chars and (sentence_end_pattern.search(chunk) or ',' in chunk):
                                          break
                                     if collected_chars >= min_initial_chars * 2: break # Safety break
                            except Exception as e:
                                logger.error(f"Error during initial text collection: {e}")
                                # Continue with what was collected

                            initial_text = "".join(initial_text_list)
                            logger.info(f"Initial text for TTS: '{initial_text}'")

                            if initial_text:
                                logger.info("Starting initial TTS...")
                                tts_task = asyncio.create_task(
                                    tts_processor.synthesize_initial_speech(initial_text)
                                )
                                await detector.set_current_tasks(tts_task=tts_task)
                                initial_audio = await tts_task
                                await detector.set_current_tasks(tts_task=None) # Clear completed task

                                if initial_audio is not None:
                                    logger.info("Sending initial audio chunk...")
                                    audio_bytes = (initial_audio * 32767).astype(np.int16).tobytes()
                                    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                                    await websocket.send(json.dumps({"audio": base64_audio}))
                                    initial_audio_sent = True
                                else:
                                     logger.warning("Initial TTS synthesis failed or produced no audio.")
                            else:
                                 logger.warning("No initial text generated or collected.")


                            # --- Remaining Text Collection ---
                            logger.info("Collecting remaining text...")
                            # Task to collect the rest of the text from the streamer
                            async def collect_remaining_inner(streamer):
                                rem_text_list = []
                                try:
                                    async for chunk in streamer:
                                        if chunk is None: break
                                        rem_text_list.append(chunk)
                                except Exception as e:
                                    logger.error(f"Error collecting remaining text: {e}", exc_info=True)
                                return "".join(rem_text_list)

                            collect_task = asyncio.create_task(collect_remaining_inner(generation_streamer))
                            # Allow other tasks while collecting
                            await detector.set_current_tasks(generation_task=collect_task) # Track collection task
                            remaining_text = await collect_task
                            await detector.set_current_tasks(generation_task=None) # Clear completed task
                            logger.info(f"Remaining text collected ({len(remaining_text)} chars).")

                            # --- Update History ---
                            full_response = initial_text + remaining_text
                            # Use the method which handles locking and pending message
                            await gemma_processor.complete_generation_and_update_history(None, gen_start_time) # Pass None streamer as it's consumed
                            # Note: This assumes complete_generation... mainly updates history now


                            # --- Remaining TTS ---
                            if remaining_text:
                                logger.info("Starting remaining TTS...")
                                tts_task = asyncio.create_task(
                                    tts_processor.synthesize_remaining_speech(remaining_text)
                                )
                                await detector.set_current_tasks(tts_task=tts_task)
                                remaining_audio = await tts_task
                                await detector.set_current_tasks(tts_task=None) # Clear completed task

                                if remaining_audio is not None:
                                    logger.info("Sending remaining audio chunk...")
                                    audio_bytes = (remaining_audio * 32767).astype(np.int16).tobytes()
                                    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                                    await websocket.send(json.dumps({"audio": base64_audio}))
                                    remaining_audio_sent = True
                                else:
                                    logger.warning("Remaining TTS synthesis failed or produced no audio.")
                            elif initial_audio_sent:
                                logger.info("No remaining text to synthesize.")
                            else:
                                # If neither initial nor remaining audio was sent, maybe send a silent marker?
                                # Or just log that no audio was produced for this response.
                                logger.warning("No audio generated for this response (initial text empty/failed, remaining text empty/failed).")


                        except asyncio.CancelledError:
                            logger.info("Generation/TTS workflow cancelled.")
                            # Attempt to clean up any potentially running sub-tasks
                            if tts_task and not tts_task.done(): tts_task.cancel()
                            if collect_task and not collect_task.done(): collect_task.cancel()
                            # History might not be updated if cancelled early
                            async with gemma_processor.lock: gemma_processor._pending_user_message = None
                            raise # Re-raise cancellation to be caught by the outer handler
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("Connection closed during generation/TTS workflow.")
                            # Propagate error if needed, or just log
                        except Exception as e:
                            logger.error(f"Error in generation/TTS workflow: {e}", exc_info=True)
                            # Fallthrough to finally
                        finally:
                            # Workflow finished (normally, cancelled, or errored)
                            logger.info("Generation/TTS workflow concluding.")
                            # Ensure TTS playing flag is reset *unless* cancellation came from *outside* this workflow
                            # If cancelled from outside, the outer loop handles the flag reset.
                            # If completed normally or errored within, reset the flag here.
                            # Check if the task itself was cancelled.
                            if not asyncio.current_task().cancelled():
                                 await detector.set_tts_playing(False)
                            # Explicitly clear tasks managed by detector state, just in case
                            await detector.set_current_tasks(generation_task=None, tts_task=None)


                    # Launch the workflow task for the current transcription
                    speech_processing_task = asyncio.create_task(generation_tts_workflow_inner(transcription))
                    # Don't await it here; the loop continues checking for new segments.

                else:
                    # No new segment, brief sleep to prevent busy-waiting
                    await asyncio.sleep(0.02) # Check queue ~50 times/sec

            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed while detecting speech.")
                break # Exit loop
            except asyncio.CancelledError:
                 logger.info("Speech detection task cancelled.")
                 break
            except Exception as e:
                # Log unexpected errors in the detection loop
                logger.error(f"Error in speech detection loop: {e}", exc_info=True)
                # Reset state cautiously
                await detector.set_tts_playing(False)
                await detector.cancel_current_tasks()
                if speech_processing_task and not speech_processing_task.done():
                    speech_processing_task.cancel()
                await asyncio.sleep(0.1) # Short pause before retrying


    async def receive_media():
        """Coroutine to receive audio and image data from the client."""
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                tts_is_playing = await detector.is_tts_playing() # Check state once per message

                # Handle audio data for VAD
                if "realtime_input" in data and "media_chunks" in data["realtime_input"]:
                    for chunk in data["realtime_input"]["media_chunks"]:
                        mime_type = chunk.get("mime_type")
                        chunk_data_b64 = chunk.get("data", "")

                        if mime_type == "audio/pcm":
                            try:
                                if chunk_data_b64:
                                    audio_data = base64.b64decode(chunk_data_b64)
                                    if audio_data:
                                         await detector.add_audio(audio_data)
                            except (TypeError, ValueError) as e:
                                logger.warning(f"Error decoding audio chunk: {e}")
                        # Handle images ONLY if TTS is not active
                        elif mime_type == "image/jpeg":
                             if not tts_is_playing:
                                 try:
                                     if chunk_data_b64:
                                         image_data = base64.b64decode(chunk_data_b64)
                                         if image_data:
                                             await gemma_processor.set_image(image_data)
                                 except (TypeError, ValueError) as e:
                                     logger.warning(f"Error decoding image chunk: {e}")
                             # else: logger.debug("Ignoring image chunk because TTS is playing.")

                # Handle standalone image messages ONLY if TTS is not active
                elif "image" in data:
                    if not tts_is_playing:
                         try:
                            image_data_b64 = data.get("image", "")
                            if image_data_b64:
                                image_data = base64.b64decode(image_data_b64)
                                if image_data:
                                     await gemma_processor.set_image(image_data)
                         except (TypeError, ValueError) as e:
                             logger.warning(f"Error decoding standalone image data: {e}")
                    # else: logger.debug("Ignoring standalone image because TTS is playing.")


            except websockets.exceptions.ConnectionClosed as e:
                logger.info(f"Connection closed by client ({e.code} {e.reason}).")
                break # Exit loop
            except asyncio.CancelledError:
                 logger.info("Media receiving task cancelled.")
                 break
            except json.JSONDecodeError:
                logger.warning(f"Received invalid JSON message.")
            except Exception as e:
                logger.error(f"Error receiving/processing client message: {e}", exc_info=True)
                # Decide whether to break or continue based on error type
                # Continue unless it's a connection error to be robust
                if isinstance(e, websockets.exceptions.WebSocketException):
                    break


    # --- Main client handling logic ---
    detection_task = asyncio.create_task(detect_and_process_speech())
    receiver_task = asyncio.create_task(receive_media())

    # Wait for either task to complete (e.g., due to connection close or fatal error)
    done, pending = await asyncio.wait(
        [detection_task, receiver_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # --- Cleanup on disconnect ---
    logger.info(f"Client handler for {client_ip} finishing, cancelling pending tasks...")
    for task in pending:
        if not task.done():
            task.cancel()
            try:
                await task # Allow task to handle cancellation gracefully
            except asyncio.CancelledError:
                pass # Expected
            except Exception as e:
                 logger.error(f"Error during pending task cleanup: {e}", exc_info=True)

    # Ensure any active workflow is explicitly cancelled on disconnect
    if speech_processing_task and not speech_processing_task.done():
         logger.info("Cancelling active speech processing workflow due to disconnect.")
         speech_processing_task.cancel()
         try:
             await speech_processing_task # Wait for it to finish cancelling
         except asyncio.CancelledError:
             pass # Expected
         except Exception as e:
              logger.error(f"Error awaiting final cancellation of workflow task: {e}")


    # Final cleanup of detector state (redundant but safe)
    await detector.cancel_current_tasks()
    logger.info(f"Client {client_ip} disconnected.")


async def main():
    """Main function to initialize models and start the WebSocket server."""
    # Initialize models/processors once at startup
    logger.info("--- Initializing Models ---")
    try:
        WhisperTranscriber.get_instance()
        GemmaMultimodalProcessor.get_instance()
        KokoroTTSProcessor.get_instance()
        logger.info("--- All Models Initialized ---")
    except Exception as e:
        logger.error(f"Fatal error during model initialization: {e}", exc_info=True)
        sys.exit(1) # Exit if core components fail to load

    host = "0.0.0.0"
    port = 9073

    logger.info(f"Starting WebSocket server on {host}:{port}")
    try:
        # Start the server with keepalive pings
        server = await websockets.serve(
            handle_client,
            host,
            port,
            ping_interval=20,    # Send ping every 20 seconds
            ping_timeout=30,     # Wait up to 30 seconds for pong response
            close_timeout=10,    # Wait up to 10 seconds for close handshake
            # Increase max_size if expecting very large image messages
            # max_size=2**22 # Example: 4MB limit (default is 1MB)
        )
        logger.info(f"WebSocket server running. Access via ws://<your-server-ip>:{port}")
        await asyncio.Future() # Run forever (or until server stops)
    except OSError as e:
         logger.error(f"Failed to start server on {host}:{port}: {e}. Is the port already in use?")
    except Exception as e:
        logger.error(f"Server encountered an unhandled error: {e}", exc_info=True)
    finally:
         logger.info("WebSocket server shutting down.")
         # Optional: Add cleanup for server object if necessary, though serve usually handles it.


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested (KeyboardInterrupt).")
    except Exception as e:
        logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True)
