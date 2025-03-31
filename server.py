import asyncio
import json
import websockets
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, Gemma3ForConditionalGeneration # <--- Import Gemma3
import numpy as np
import logging
import sys
import io
from PIL import Image
import time
import os
from datetime import datetime
# Import Kokoro TTS library
from kokoro import KPipeline
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', # Added filename/lineno
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Constants ---
SAMPLE_RATE = 16000
ENERGY_THRESHOLD = 0.018 # Slightly higher threshold
SILENCE_DURATION = 1.0    # Increased silence duration
MIN_SPEECH_DURATION = 1.0 # Increased min speech duration
MAX_SPEECH_DURATION = 15

WHISPER_MODEL_ID = "openai/whisper-large-v3" # Or choose a faster variant like "distil-whisper/distil-large-v2"
GEMMA_MODEL_ID = "google/gemma-3-4b-it" # <--- Using Gemma 3 4B IT model
KOKORO_VOICE = 'en_us_sarah' # Assuming English voice is desired

# --- Singleton Classes (Modified/Simplified) ---

class AudioSegmentDetector:
    """Detects speech segments and manages interruption"""

    def __init__(self,
                 websocket, # Pass websocket for sending interrupts
                 sample_rate=SAMPLE_RATE,
                 energy_threshold=ENERGY_THRESHOLD,
                 silence_duration=SILENCE_DURATION,
                 min_speech_duration=MIN_SPEECH_DURATION,
                 max_speech_duration=MAX_SPEECH_DURATION):

        self.websocket = websocket # Store websocket reference
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_samples = int(silence_duration * sample_rate)
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.max_speech_samples = int(max_speech_duration * sample_rate)

        # Internal state
        self.audio_buffer = bytearray()
        self.is_speech_active = False
        self.silence_counter = 0
        self.speech_start_idx = 0
        self.lock = asyncio.Lock()
        self.segment_queue = asyncio.Queue()

        # Counters
        self.segments_detected = 0

        # TTS playback and generation control
        self.tts_playing = False
        self.tts_lock = asyncio.Lock() # Lock for tts_playing state
        self.task_lock = asyncio.Lock() # Lock for accessing tasks
        self.current_generation_task = None
        self.current_tts_task = None
        self.interrupt_pending = False # Flag to prevent multiple interrupts

    async def set_tts_playing(self, is_playing):
        """Set TTS playback state safely"""
        async with self.tts_lock:
            if self.tts_playing != is_playing:
                logger.info(f"Setting TTS playing state to: {is_playing}")
                self.tts_playing = is_playing
                if not is_playing:
                    # Reset interrupt flag when TTS stops
                    self.interrupt_pending = False
            # Always ensure tasks are cleared if we are setting to not playing
            if not is_playing:
                 await self._clear_tasks_unsafe() # Clear tasks without acquiring task lock again


    async def _clear_tasks_unsafe(self):
        """Clears tasks without acquiring the task lock (must be called within task_lock context or when safe)"""
        self.current_generation_task = None
        self.current_tts_task = None
        logger.debug("Cleared current generation and TTS tasks.")


    async def cancel_current_tasks(self):
        """Cancel any ongoing generation and TTS tasks safely"""
        async with self.task_lock:
            cancelled_gen = False
            if self.current_generation_task and not self.current_generation_task.done():
                logger.info("Attempting to cancel generation task...")
                self.current_generation_task.cancel()
                try:
                    await asyncio.wait_for(self.current_generation_task, timeout=1.0) # Short timeout
                except asyncio.CancelledError:
                    logger.info("Generation task cancelled successfully.")
                except asyncio.TimeoutError:
                     logger.warning("Timeout waiting for generation task cancellation.")
                except Exception as e:
                    logger.error(f"Error awaiting cancelled generation task: {e}")
                cancelled_gen = True
                self.current_generation_task = None # Clear reference immediately

            cancelled_tts = False
            if self.current_tts_task and not self.current_tts_task.done():
                logger.info("Attempting to cancel TTS task...")
                self.current_tts_task.cancel()
                try:
                    await asyncio.wait_for(self.current_tts_task, timeout=1.0) # Short timeout
                except asyncio.CancelledError:
                    logger.info("TTS task cancelled successfully.")
                except asyncio.TimeoutError:
                     logger.warning("Timeout waiting for TTS task cancellation.")
                except Exception as e:
                    logger.error(f"Error awaiting cancelled TTS task: {e}")
                cancelled_tts = True
                self.current_tts_task = None # Clear reference immediately

            if cancelled_gen or cancelled_tts:
                 logger.info("Ongoing task cancellation process finished.")
            # TTS playing state will be reset by the caller or in the main loop finally block

    async def set_current_tasks(self, generation_task=None, tts_task=None):
        """Set current generation and TTS tasks safely"""
        async with self.task_lock:
            self.current_generation_task = generation_task
            self.current_tts_task = tts_task
            logger.debug(f"Set current tasks: Gen={generation_task is not None}, TTS={tts_task is not None}")

    async def add_audio(self, audio_bytes):
        """Add audio data, detect speech, and handle interruptions."""
        async with self.lock: # Lock for buffer manipulation and VAD state
            self.audio_buffer.extend(audio_bytes)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_array) == 0:
                return None

            energy = np.sqrt(np.mean(audio_array**2))

            # --- Interruption Logic ---
            # Check TTS state without blocking buffer processing for too long
            async with self.tts_lock:
                is_currently_playing = self.tts_playing

            if is_currently_playing and energy > self.energy_threshold / 2 and not self.interrupt_pending: # More sensitive threshold for interruption
                logger.info(f"Interrupt detected! Energy {energy:.4f} > threshold/2 while TTS playing.")
                self.interrupt_pending = True # Set flag to prevent spamming

                # Perform cancellation and sending interrupt in a separate task
                # to avoid holding the buffer lock
                asyncio.create_task(self.handle_interruption())

                # Don't process this audio chunk further for speech detection if interrupting
                return None
            # --- End Interruption Logic ---

            # --- Speech Detection Logic (only if not interrupting) ---
            if not self.is_speech_active and energy > self.energy_threshold:
                self.is_speech_active = True
                self.speech_start_idx = max(0, len(self.audio_buffer) - len(audio_bytes))
                self.silence_counter = 0
                logger.info(f"Speech start detected (energy: {energy:.6f})")

            elif self.is_speech_active:
                if energy > self.energy_threshold:
                    self.silence_counter = 0
                else:
                    self.silence_counter += len(audio_array)

                    if self.silence_counter >= self.silence_samples:
                        speech_end_idx = len(self.audio_buffer) - self.silence_counter
                        if self.speech_start_idx < speech_end_idx: # Ensure valid segment
                            speech_segment = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])
                            segment_len_samples = len(speech_segment) // 2

                            # Reset VAD state *before* queueing
                            current_buffer_tail = self.audio_buffer[speech_end_idx:]
                            self.audio_buffer = current_buffer_tail
                            self.is_speech_active = False
                            self.silence_counter = 0
                            self.speech_start_idx = 0 # Reset start index relative to new buffer

                            if segment_len_samples >= self.min_speech_samples:
                                self.segments_detected += 1
                                logger.info(f"Speech segment detected: {segment_len_samples / self.sample_rate:.2f}s")
                                await self.segment_queue.put(speech_segment)
                                # Return value not used now, action happens on queue read
                            else:
                                logger.info(f"Speech segment too short ({segment_len_samples / self.sample_rate:.2f}s), discarding.")
                        else:
                             # Reset state even if segment is invalid
                            logger.warning(f"Invalid segment indices on silence end: start={self.speech_start_idx}, end={speech_end_idx}. Resetting VAD.")
                            self.audio_buffer = bytearray() # Clear buffer on error
                            self.is_speech_active = False
                            self.silence_counter = 0
                            self.speech_start_idx = 0


                # Check max duration (only if speech is still active)
                current_segment_len_bytes = len(self.audio_buffer) - self.speech_start_idx
                if self.is_speech_active and current_segment_len_bytes >= self.max_speech_samples * 2:
                    # Extract max duration segment
                    speech_segment = bytes(self.audio_buffer[self.speech_start_idx : self.speech_start_idx + self.max_speech_samples * 2])
                    segment_len_samples = len(speech_segment) // 2

                    # Reset VAD state, but keep the buffer overlap
                    new_start_idx = self.speech_start_idx + self.max_speech_samples * 2
                    current_buffer_tail = self.audio_buffer[new_start_idx:]
                    self.audio_buffer = current_buffer_tail # Keep remaining part
                    # self.is_speech_active remains True (potentially)
                    self.silence_counter = 0 # Reset silence as we forced a split
                    self.speech_start_idx = 0 # Reset start index relative to new buffer

                    self.segments_detected += 1
                    logger.info(f"Max duration speech segment extracted: {segment_len_samples / self.sample_rate:.2f}s")
                    await self.segment_queue.put(speech_segment)
                    # Return value not used

            # Clean up buffer if it gets too large without speech detection
            max_buffer_samples = (self.max_speech_duration + self.silence_duration) * self.sample_rate * 5 # Allow larger buffer
            if not self.is_speech_active and len(self.audio_buffer) > max_buffer_samples * 2:
                 logger.warning(f"Audio buffer trimming (idle): {len(self.audio_buffer)} bytes")
                 keep_bytes = int(self.silence_samples * 2 * 1.5) # Keep a bit more than silence duration
                 self.audio_buffer = self.audio_buffer[-keep_bytes:]
                 self.speech_start_idx = 0 # Reset relative start index

            return None # Return value not used

    async def handle_interruption(self):
        """Handles cancelling tasks and sending interrupt signal."""
        logger.debug("Executing handle_interruption task...")
        await self.cancel_current_tasks()
        try:
            interrupt_message = json.dumps({"interrupt": True})
            await self.websocket.send(interrupt_message)
            logger.info("Sent interrupt signal to client.")
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed, cannot send interrupt signal.")
        except Exception as e:
            logger.error(f"Failed to send interrupt signal: {e}")
        # Important: Resetting tts_playing to False should happen
        # in the main loop's finally block after processing is confirmed stopped.
        # Resetting interrupt_pending is handled when set_tts_playing(False) is called.


    async def get_next_segment(self):
        """Get the next available speech segment"""
        try:
            # Use a small timeout to prevent blocking indefinitely if queue remains empty
            return await asyncio.wait_for(self.segment_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error getting segment from queue: {e}")
            return None

class WhisperTranscriber:
    _instance = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls):
        async with cls._lock:
            if cls._instance is None:
                logger.info("Initializing WhisperTranscriber instance...")
                cls._instance = cls()
                logger.info("WhisperTranscriber instance initialized.")
        return cls._instance

    def __init__(self):
        if WhisperTranscriber._instance is not None:
             raise Exception("This class is a singleton!")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Whisper using device: {self.device}")
        self.torch_dtype = torch.float16 if self.device != "cpu" else torch.float32

        logger.info(f"Loading Whisper model: {WHISPER_MODEL_ID}...")
        start_time = time.time()
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                WHISPER_MODEL_ID,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                # attn_implementation="flash_attention_2" # Optional: if flash attention is installed and compatible
            )
            self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
                chunk_length_s=30, # Process in 30s chunks
                batch_size=8       # Batch inference if possible (pipeline handles this)
            )
            logger.info(f"Whisper model loaded in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            raise

        self.transcription_count = 0

    async def transcribe(self, audio_bytes, sample_rate=SAMPLE_RATE):
        """Transcribe audio bytes to text using the pipeline"""
        if not audio_bytes:
            logger.warning("Transcription attempted with empty audio bytes.")
            return ""
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_array) < sample_rate * 0.1: # Check minimum length (e.g., 0.1 seconds)
                logger.info(f"Audio segment too short for transcription ({len(audio_array)/sample_rate:.2f}s).")
                return ""

            start_time = time.time()
            # Use run_in_executor for the blocking pipeline call
            result = await asyncio.get_event_loop().run_in_executor(
                None, # Use default executor (ThreadPoolExecutor)
                lambda: self.pipe(
                    {"array": audio_array, "sampling_rate": sample_rate},
                     generate_kwargs={
                        "task": "transcribe",
                        "language": "english", # Force English
                        "temperature": 0.0 # Deterministic transcription
                    }
                )
            )
            elapsed = time.time() - start_time

            text = result.get("text", "").strip()
            if text: # Only count non-empty results
                 self.transcription_count += 1
                 logger.info(f"Transcription #{self.transcription_count} ({(len(audio_array)/sample_rate):.2f}s audio -> {elapsed:.2f}s): '{text}'")
            else:
                 logger.info(f"Transcription resulted in empty text. ({(len(audio_array)/sample_rate):.2f}s audio -> {elapsed:.2f}s)")


            return text

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return ""

class GemmaMultimodalProcessor:
    _instance = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls):
        async with cls._lock:
            if cls._instance is None:
                logger.info("Initializing GemmaMultimodalProcessor instance...")
                cls._instance = cls()
                logger.info("GemmaMultimodalProcessor instance initialized.")
        return cls._instance

    def __init__(self):
        if GemmaMultimodalProcessor._instance is not None:
            raise Exception("This class is a singleton!")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Gemma using device: {self.device}")

        logger.info(f"Loading Gemma model: {GEMMA_MODEL_ID}...")
        start_time = time.time()
        try:
            # Use bfloat16 for mixed-precision (good balance of speed/memory on compatible GPUs)
            # Keep 8-bit loading for memory savings
            model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
            logger.info(f"Using dtype: {model_dtype} and 8-bit loading for Gemma 3")

            self.model = Gemma3ForConditionalGeneration.from_pretrained( # <--- Use Gemma3ForConditionalGeneration
                GEMMA_MODEL_ID,
                device_map="auto",       # Distribute across available devices
                load_in_8bit=True,       # Enable 8-bit quantization for memory saving
                torch_dtype=model_dtype, # Use bfloat16 or float32
                # Optional: Add if flash attention is installed and compatible
                # attn_implementation="flash_attention_2"
            )
            # Processor is usually fine with AutoProcessor for the same model family
            self.processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)

            logger.info(f"Gemma 3 model loaded in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to load Gemma 3 model: {e}", exc_info=True)
            raise

        self.last_image = None
        self.last_image_timestamp = 0
        self.image_lock = asyncio.Lock()

        self.message_history = []
        self.max_history_len_tokens = 1536 # Gemma 3 might handle slightly longer context
        self.history_lock = asyncio.Lock()

        self.generation_count = 0

    async def set_image(self, image_data):
        """Cache the most recent image received, resizing it."""
        async with self.image_lock:
            try:
                image = Image.open(io.BytesIO(image_data)).convert("RGB") # Ensure RGB

                # Optional: Resize for faster processing / less VRAM
                max_size = (1024, 1024) # Keep reasonable size
                image.thumbnail(max_size, Image.Resampling.LANCZOS)

                # Clear history when a new image is explicitly set
                async with self.history_lock:
                    logger.info("New image received, clearing conversation history.")
                    self.message_history = []

                self.last_image = image
                self.last_image_timestamp = time.time()
                logger.info(f"Cached new image (resized to {image.size}).")
                return True
            except Exception as e:
                logger.error(f"Error processing image: {e}", exc_info=True)
                return False

    async def _build_messages_for_template(self, text):
        """Builds the list of messages in the format expected by apply_chat_template."""
        async with self.image_lock:
            current_image = self.last_image

        # --- System Prompt ---
        # Note: Adapt this prompt based on how you want Gemma 3 to behave with images.
        system_message = """You are VILA, a helpful vision-language assistant. Analyze the image provided and respond concisely (1-3 sentences) to the user's query in a natural, conversational way. If the query isn't about the image, respond conversationally without forcing image description."""
        messages = [{"role": "system", "content": system_message}]

        # --- History ---
        async with self.history_lock:
            # Append relevant history messages
            # Simple token counting to limit history size
            current_token_count = len(self.processor.tokenizer.encode(system_message)) # Start with system prompt tokens
            history_to_include = []
            for msg in reversed(self.message_history):
                msg_content = msg.get("content")
                if not msg_content: continue # Skip empty messages

                # Estimate token count (more accurate than len/4)
                # Check if content is list (multimodal) or string (text)
                if isinstance(msg_content, list): # Should be text only in history now
                     text_content = next((item['text'] for item in msg_content if item['type'] == 'text'), '')
                     msg_tokens = len(self.processor.tokenizer.encode(text_content))
                elif isinstance(msg_content, str):
                     msg_tokens = len(self.processor.tokenizer.encode(msg_content))
                else:
                     msg_tokens = 0 # Should not happen

                # Add message in the correct format (role/content dictionary)
                # Check token limit *before* adding
                if current_token_count + msg_tokens < self.max_history_len_tokens:
                    history_to_include.append(msg)
                    current_token_count += msg_tokens
                else:
                    logger.debug(f"History limit ({self.max_history_len_tokens}) reached. Truncating older messages.")
                    break # Stop adding history once limit is reached

            messages.extend(reversed(history_to_include)) # Add history in chronological order

        # --- Current User Input ---
        user_content = []
        if current_image:
            # The template expects the actual image object passed separately,
            # but we use a placeholder in the message list for structure.
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "content": text}) # Changed "text" key to "content"
        messages.append({"role": "user", "content": user_content}) # Content is now a list

        return messages, current_image

    async def _update_history(self, user_text, assistant_response):
        """Update message history, trimming old messages based on token count."""
        async with self.history_lock:
            # Store simple text content for history management
            # User message content is just text
            self.message_history.append({"role": "user", "content": user_text})
            # Assistant message content is also just text
            self.message_history.append({"role": "assistant", "content": assistant_response})

            # Trim history (using the same logic as in _build_messages_for_template for consistency)
            # Re-calculate token count to trim accurately
            system_tokens = len(self.processor.tokenizer.encode("system prompt placeholder")) # Rough estimate
            total_tokens = system_tokens
            trimmed_history = []
            for msg in reversed(self.message_history):
                msg_content = msg.get("content")
                if not msg_content: continue

                # History should only contain strings now
                if isinstance(msg_content, str):
                     msg_tokens = len(self.processor.tokenizer.encode(msg_content))
                else:
                     logger.warning(f"Unexpected content type in history: {type(msg_content)}")
                     msg_tokens = 0

                if total_tokens + msg_tokens < self.max_history_len_tokens:
                    trimmed_history.append(msg)
                    total_tokens += msg_tokens
                else:
                    break # Stop adding older messages
            self.message_history = list(reversed(trimmed_history))
            logger.debug(f"History updated. Approx tokens: {total_tokens}, Messages: {len(self.message_history)}")


    async def generate(self, text):
        """Generate a response using the latest image and text input (non-streaming)."""
        messages, image_input = await self._build_messages_for_template(text)

        if not messages:
            return "Error: Could not build messages for the model."

        try:
            start_time = time.time()

            # Prepare inputs using the processor and apply_chat_template
            # Pass the image separately if it exists
            processor_kwargs = {"messages": messages, "add_generation_prompt": True, "tokenize": True, "return_tensors": "pt"}
            if image_input:
                processor_kwargs["images"] = image_input # Pass the actual PIL image object here

            # This step combines text, history, image placeholder, and applies the correct chat template
            inputs = self.processor.apply_chat_template(**processor_kwargs).to(self.model.device)

            # Check input shape
            logger.debug(f"Input IDs shape: {inputs['input_ids'].shape}")

            # Generate response using run_in_executor for the blocking call
            generate_ids = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate(
                    **inputs,
                    max_new_tokens=128, # Keep response concise
                    do_sample=True,
                    temperature=0.7,
                    # pad_token_id=self.processor.tokenizer.eos_token_id # Often not needed with chat templates
                )
            )

            # Decode the generated tokens
            input_len = inputs["input_ids"].shape[1]
            output_ids = generate_ids[0][input_len:]
            generated_text = self.processor.decode(output_ids, skip_special_tokens=True).strip()

            elapsed = time.time() - start_time
            self.generation_count += 1

            # Update conversation history *after* successful generation
            await self._update_history(text, generated_text)

            logger.info(f"Gemma 3 generation #{self.generation_count} ({elapsed:.2f}s): '{generated_text[:100]}...'")
            return generated_text

        except asyncio.CancelledError:
             logger.info("Gemma generation task was cancelled.")
             raise # Re-raise cancellation
        except Exception as e:
            logger.error(f"Gemma 3 generation error: {e}", exc_info=True)
            if "out of memory" in str(e).lower() and self.device == "cuda:0":
                torch.cuda.empty_cache()
                logger.warning("Cleared CUDA cache after potential OOM error during Gemma 3 generation.")
            return f"Sorry, I encountered an error processing that request."


class KokoroTTSProcessor:
    _instance = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls):
        async with cls._lock:
            if cls._instance is None:
                logger.info("Initializing KokoroTTSProcessor instance...")
                cls._instance = cls()
                logger.info("KokoroTTSProcessor instance initialized.")
        return cls._instance

    def __init__(self):
        if KokoroTTSProcessor._instance is not None:
             raise Exception("This class is a singleton!")

        logger.info("Initializing Kokoro TTS processor...")
        start_time = time.time()
        try:
            # Initialize Kokoro TTS pipeline
            self.pipeline = KPipeline(lang_code='a') # Auto-detect language
            self.default_voice = KOKORO_VOICE # Use constant
            if not hasattr(self.pipeline, 'sr'):
                 logger.warning("Kokoro pipeline does not have 'sr' attribute. Assuming 24000 Hz.")
                 self.pipeline.sr = 24000 # Default Kokoro sample rate? Verify this.

            logger.info(f"Kokoro TTS processor initialized in {time.time() - start_time:.2f}s. Sample rate: {self.pipeline.sr}")

        except Exception as e:
            logger.error(f"Error initializing Kokoro TTS: {e}", exc_info=True)
            self.pipeline = None
        self.synthesis_count = 0

    async def synthesize_speech(self, text):
        """Convert text to speech using Kokoro TTS (simplified)."""
        if not text or not self.pipeline:
            logger.warning(f"TTS skipped: No text ('{text}') or pipeline not available.")
            return None

        try:
            logger.info(f"Synthesizing speech #{self.synthesis_count + 1} for text: '{text[:50]}...'")
            start_time = time.time()
            audio_segments = []

            # Use run_in_executor for the potentially blocking TTS generation
            generator = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipeline(
                    text,
                    voice=self.default_voice,
                    speed=1.0, # Default speed
                    split_pattern=r'[.!?]+|[,\uff0c\u3002\uff1f\uff01]+' # Split reasonably
                )
            )

            # Consume the generator items
            for _, _, audio_chunk in generator:
                 if audio_chunk is not None and len(audio_chunk) > 0:
                      audio_segments.append(audio_chunk)

            if not audio_segments:
                 logger.warning(f"TTS produced no audio for text: '{text[:50]}...'")
                 return None

            combined_audio = np.concatenate(audio_segments)
            elapsed = time.time() - start_time
            self.synthesis_count += 1
            # Use pipeline's sample rate for duration calculation
            audio_duration_sec = len(combined_audio) / self.pipeline.sr if self.pipeline and hasattr(self.pipeline, 'sr') else 0
            logger.info(f"Speech synthesis #{self.synthesis_count} complete ({elapsed:.2f}s): {audio_duration_sec:.2f}s audio")

            # Ensure output is float32 between -1 and 1
            if combined_audio.dtype != np.float32:
                 combined_audio = combined_audio.astype(np.float32)
                 # Normalize if necessary (e.g., if output is int16)
                 max_val = np.max(np.abs(combined_audio))
                 if max_val > 1.0:
                      if max_val > 32767: # Likely int16
                           combined_audio /= 32768.0
                      else: # Possibly just slightly > 1.0 float
                           combined_audio /= max_val

            return combined_audio

        except asyncio.CancelledError:
             logger.info("Speech synthesis task was cancelled.")
             raise # Re-raise cancellation
        except Exception as e:
            logger.error(f"Speech synthesis error: {e}", exc_info=True)
            return None


# --- WebSocket Handler ---

async def handle_client(websocket):
    """Handles WebSocket client connection, processing audio/images and generating responses."""
    client_id = websocket.remote_address
    logger.info(f"Client {client_id} connected.")

    detector = None # Initialize detector inside try block
    transcriber = None
    gemma_processor = None
    tts_processor = None

    try:
        # Initialize components specific to this client connection session
        detector = AudioSegmentDetector(websocket=websocket) # Pass websocket for interrupts
        transcriber = await WhisperTranscriber.get_instance()
        gemma_processor = await GemmaMultimodalProcessor.get_instance()
        tts_processor = await KokoroTTSProcessor.get_instance()

        # --- Main Processing Loop Task ---
        async def process_speech_queue():
            while True:
                generation_task = None # Define outside try block
                tts_task = None        # Define outside try block
                try:
                    speech_segment = await detector.get_next_segment()
                    if not speech_segment:
                        await asyncio.sleep(0.02) # Short sleep if no segment
                        continue

                    # --- Transcription ---
                    transcription = await transcriber.transcribe(speech_segment)
                    if not transcription or not any(c.isalnum() for c in transcription):
                        logger.info(f"Skipping empty or non-alphanumeric transcription: '{transcription}'")
                        continue

                    # --- Filter short/filler ---
                    words = [w for w in transcription.split() if any(c.isalnum() for c in w)]
                    if len(words) <= 1: # Filter single words
                        logger.info(f"Skipping single-word transcription: '{transcription}'")
                        continue
                    # More robust filler check using fullmatch
                    filler_patterns = [r'^(um+|uh+|ah+|oh+|hm+|mhm+|hmm+|huh)$', r'^(okay|yes|no|yeah|nah|bye|hi|hello)$', r'thank you\.?']
                    normalized_transcription = transcription.lower().strip().rstrip('.')
                    if any(re.fullmatch(pattern, normalized_transcription) for pattern in filler_patterns):
                        logger.info(f"Skipping likely filler phrase: '{transcription}'")
                        continue

                    logger.info(f"Processing valid transcription: '{transcription}'")

                    # --- Generation & TTS ---
                    # Set TTS playing flag only *before* starting potentially long operations
                    await detector.set_tts_playing(True)

                    # Create and store generation task
                    generation_task = asyncio.create_task(gemma_processor.generate(transcription))
                    await detector.set_current_tasks(generation_task=generation_task)
                    generated_text = await generation_task
                    # Clear task immediately after completion/error
                    await detector.set_current_tasks(generation_task=None)

                    if generated_text:
                        # Create and store TTS task
                        tts_task = asyncio.create_task(tts_processor.synthesize_speech(generated_text))
                        await detector.set_current_tasks(tts_task=tts_task)
                        synthesized_audio = await tts_task
                        # Clear task immediately after completion/error
                        await detector.set_current_tasks(tts_task=None)

                        if synthesized_audio is not None and len(synthesized_audio) > 0:
                            # Convert float32 audio back to int16 bytes
                            audio_int16 = (synthesized_audio * 32767).astype(np.int16)
                            audio_bytes = audio_int16.tobytes()
                            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

                            # Send audio response
                            await websocket.send(json.dumps({"audio": base64_audio}))
                            tts_sample_rate = tts_processor.pipeline.sr if tts_processor and hasattr(tts_processor.pipeline, 'sr') else SAMPLE_RATE
                            logger.info(f"Sent {len(audio_bytes)/ (tts_sample_rate*2):.2f}s audio response to client.")
                        else:
                             logger.warning("TTS synthesis produced no audio, skipping send.")
                    else:
                         logger.warning("Gemma generation produced no text, skipping TTS.")

                except asyncio.CancelledError:
                    logger.info("Processing task (Gen/TTS) was cancelled, likely due to interruption.")
                    # Cancellation should have been handled by handle_interruption or main loop exit
                    # Ensure tasks are cleared if cancellation happened here unexpectedly
                    if generation_task and not generation_task.done(): generation_task.cancel()
                    if tts_task and not tts_task.done(): tts_task.cancel()
                    await detector.set_current_tasks(generation_task=None, tts_task=None)
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Connection closed during generation/TTS processing.")
                    break # Exit loop if connection is closed
                except Exception as e:
                    logger.error(f"Error during generation/TTS processing task: {e}", exc_info=True)
                    # Attempt to send an error message?
                    try:
                         await websocket.send(json.dumps({"error": "Processing failed"}))
                    except Exception:
                         pass # Ignore send errors if connection is broken
                finally:
                    # Crucial: Always reset the playing flag after processing (success, error, or cancellation)
                    await detector.set_tts_playing(False)
                    # Ensure tasks are cleared in the detector state
                    # await detector.set_current_tasks(generation_task=None, tts_task=None) # Redundant if done in try/except, but safe

            logger.info(f"Speech processing queue task finished for {client_id}.")


        # --- Input Receiving Task ---
        async def receive_input():
            async for message in websocket:
                try:
                    data = json.loads(message)

                    # Handle audio chunks for VAD
                    if "realtime_input" in data:
                        for chunk in data["realtime_input"]["media_chunks"]:
                            mime_type = chunk.get("mime_type")
                            chunk_data = chunk.get("data")
                            if not chunk_data: continue

                            if mime_type == "audio/pcm":
                                audio_data = base64.b64decode(chunk_data)
                                if detector:
                                     await detector.add_audio(audio_data)
                            elif mime_type == "image/jpeg":
                                 # Check if TTS is playing before processing image to avoid disrupting history
                                 async with detector.tts_lock:
                                      can_process_image = not detector.tts_playing
                                 if can_process_image and gemma_processor:
                                     logger.info("Received inline image data.")
                                     image_data = base64.b64decode(chunk_data)
                                     await gemma_processor.set_image(image_data)
                                 elif not can_process_image:
                                      logger.debug("Ignoring image update while TTS is playing.")


                    # Handle standalone image
                    elif "image" in data:
                         # Check if TTS is playing before processing image
                         async with detector.tts_lock:
                             can_process_image = not detector.tts_playing
                         if can_process_image and gemma_processor:
                             logger.info("Received standalone image data.")
                             image_data = base64.b64decode(data["image"])
                             await gemma_processor.set_image(image_data)
                         elif not can_process_image:
                              logger.debug("Ignoring image update while TTS is playing.")

                    # Handle client confirmation that TTS playback finished (optional but good)
                    elif "tts_finished" in data and data["tts_finished"]:
                         logger.info("Client indicated TTS finished playing.")
                         if detector:
                             # Reset the interrupt pending flag now that client confirms silence
                              async with detector.tts_lock:
                                  detector.interrupt_pending = False


                except json.JSONDecodeError:
                    logger.error(f"Received invalid JSON message from {client_id}")
                except KeyError as e:
                    logger.error(f"Received message with missing key: {e} from {client_id}")
                except websockets.exceptions.ConnectionClosed:
                    logger.warning(f"Connection closed while receiving from {client_id}.")
                    break # Exit loop
                except Exception as e:
                    logger.error(f"Error processing received message from {client_id}: {e}", exc_info=True)

            logger.info(f"Receive input loop finished for {client_id}.")


        # Run tasks concurrently
        processing_task = asyncio.create_task(process_speech_queue(), name=f"ProcessQueue-{client_id}")
        receiving_task = asyncio.create_task(receive_input(), name=f"ReceiveInput-{client_id}")

        # Wait for either task to complete (e.g., due to connection close or internal error)
        done, pending = await asyncio.wait(
            {processing_task, receiving_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        logger.info(f"First task completed for client {client_id}. Done: {[d.get_name() for d in done]}. Pending: {[p.get_name() for p in pending]}")

        # Cancel any remaining tasks gracefully
        for task in pending:
            if not task.done():
                logger.info(f"Cancelling pending task: {task.get_name()}")
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0) # Wait briefly for cancellation
                except asyncio.CancelledError:
                    logger.info(f"Task {task.get_name()} successfully cancelled.")
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for task {task.get_name()} to cancel.")
                except Exception as e:
                    logger.error(f"Error during cancellation of task {task.get_name()}: {e}")

        logger.info(f"Finished waiting for all tasks for client {client_id}.")


    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Client {client_id} disconnected: {e.code} {e.reason}")
    except Exception as e:
        logger.error(f"Unhandled error in handle_client for {client_id}: {e}", exc_info=True)
    finally:
        logger.info(f"Cleaning up resources for client {client_id}.")
        # Ensure any lingering tasks associated with detector are cancelled
        if detector:
            logger.info(f"Performing final check/cancellation for detector tasks of client {client_id}")
            await detector.cancel_current_tasks()
            await detector.set_tts_playing(False) # Ensure flag is off

# --- Server Startup ---

async def main():
    """Initializes models and starts the WebSocket server."""
    # Initialize singletons eagerly at startup
    logger.info("Pre-loading models...")
    try:
        await WhisperTranscriber.get_instance()
        await GemmaMultimodalProcessor.get_instance()
        await KokoroTTSProcessor.get_instance()
        logger.info("All models loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize models: {e}. Server cannot start.", exc_info=True)
        return # Stop if models fail to load

    host = "0.0.0.0"
    port = 9073
    logger.info(f"Starting WebSocket server on {host}:{port}")

    # Set higher connection limits if needed
    # Increase open file limit if necessary: ulimit -n <number> (Linux/macOS)
    try:
        server = await websockets.serve(
            handle_client,
            host,
            port,
            ping_interval=20,    # Keep connection alive
            ping_timeout=60,     # Timeout for pong response
            close_timeout=10,    # Timeout for closing handshake
            max_size=2**24,      # Allow larger messages (e.g., 16MB for high-res images)
            # read_limit=2**24,    # Limit for single frame read
            # write_limit=2**24    # Limit for single frame write
        )
        logger.info(f"WebSocket server running on ws://{host}:{port}")
        await server.wait_closed() # Run until the server is closed

    except OSError as e:
         logger.critical(f"Failed to start server, possibly port {port} is already in use: {e}")
    except Exception as e:
        logger.critical(f"Server encountered critical error: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt).")
    finally:
        logger.info("Server shutdown complete.")
