import asyncio
import json
import websockets
import base64
import torch
# Corrected Gemma import if using Gemma 2
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, GemmaForConditionalGeneration, BitsAndBytesConfig
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
import traceback # For detailed error logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    # Added module and line number for better debugging
    format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Constants ---
SAMPLE_RATE = 16000
ENERGY_THRESHOLD = 0.015  # Adjust based on mic sensitivity
SILENCE_DURATION = 0.7    # Shorter silence to feel more responsive
MIN_SPEECH_DURATION = 0.5 # Shorter min duration
MAX_SPEECH_DURATION = 10  # Shorter max duration for faster turns
WEBSOCKET_PORT = 9073

class AudioSegmentDetector:
    """Detects speech segments based on audio energy levels and manages processing state"""

    def __init__(self,
                 sample_rate=SAMPLE_RATE,
                 energy_threshold=ENERGY_THRESHOLD,
                 silence_duration=SILENCE_DURATION,
                 min_speech_duration=MIN_SPEECH_DURATION,
                 max_speech_duration=MAX_SPEECH_DURATION):

        self.sample_rate = sample_rate
        self.bytes_per_sample = 2 # 16-bit PCM
        self.energy_threshold = energy_threshold
        self.silence_samples = int(silence_duration * sample_rate)
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.max_speech_samples = int(max_speech_duration * sample_rate)

        # Internal state for VAD
        self.audio_buffer = bytearray()
        self.is_speech_active = False
        self.silence_counter = 0
        self.speech_start_idx = 0
        self.buffer_lock = asyncio.Lock() # Protects buffer access
        self.segment_queue = asyncio.Queue()

        # Assistant response control state
        self.assistant_is_responding = False
        self.state_lock = asyncio.Lock() # Protects assistant_is_responding and current_processing_task
        self.current_processing_task = None
        self.last_interrupt_time = 0 # To prevent rapid-fire interrupts


    async def get_responding_state(self):
        """Safely get the current responding state."""
        async with self.state_lock:
            return self.assistant_is_responding

    async def set_assistant_responding(self, is_responding: bool, task: asyncio.Task = None):
        """Set assistant response state and optionally store the task."""
        async with self.state_lock:
            self.assistant_is_responding = is_responding
            # Store task only when starting, clear when stopping
            self.current_processing_task = task if is_responding else None
            logger.info(f"Assistant responding state set to: {is_responding}")
            if not is_responding:
                # Clear any VAD state when assistant stops, prevents carry-over noise detection
                self.is_speech_active = False
                self.silence_counter = 0

    async def cancel_current_processing(self):
        """Cancel any ongoing generation and TTS task"""
        async with self.state_lock:
            task_to_cancel = self.current_processing_task
            was_responding = self.assistant_is_responding
            # Immediately mark as not responding and clear task reference
            self.assistant_is_responding = False
            self.current_processing_task = None

        if task_to_cancel and not task_to_cancel.done():
            logger.info("Attempting to cancel ongoing processing task.")
            task_to_cancel.cancel()
            try:
                # Wait briefly for cancellation to register
                await asyncio.wait_for(asyncio.shield(task_to_cancel), timeout=0.2)
            except asyncio.CancelledError:
                logger.info("Ongoing processing task cancelled successfully.")
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for task cancellation acknowledgment (task might still be cancelling).")
            except Exception as e:
                 logger.error(f"Unexpected error during task cancellation await: {e}")
            finally:
                 # Ensure state is False even if cancellation had issues (already set under lock)
                 logger.info(f"Responding state confirmed False after cancellation attempt (was {was_responding}).")
                 # Reset VAD state after cancellation
                 self.is_speech_active = False
                 self.silence_counter = 0
        elif was_responding:
             # If state was responding but no task found, ensure state is False
             logger.info("Responding state was true but no active task found. Resetting state.")
             # State already set to False under lock


    async def add_audio(self, audio_bytes):
        """Add audio data to the buffer and check for speech segments"""
        async with self.buffer_lock:
            self.audio_buffer.extend(audio_bytes)
            buffer_len_samples = len(self.audio_buffer) // self.bytes_per_sample

            # Trim buffer if it gets excessively long (e.g., > 30 seconds)
            max_buffer_samples = 30 * self.sample_rate
            if buffer_len_samples > max_buffer_samples:
                 trim_amount_bytes = (buffer_len_samples - max_buffer_samples) * self.bytes_per_sample
                 self.audio_buffer = self.audio_buffer[trim_amount_bytes:]
                 # Adjust speech_start_idx relative to the new buffer start
                 self.speech_start_idx = max(0, self.speech_start_idx - trim_amount_bytes)
                 logger.warning(f"Audio buffer trimmed to {max_buffer_samples / self.sample_rate}s")
                 buffer_len_samples = max_buffer_samples # Update length after trim


            # Use only the newly added audio for energy calculation
            num_new_samples = len(audio_bytes) // self.bytes_per_sample
            if num_new_samples == 0:
                return None

            try:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                # Handle potential silence resulting in NaN/Inf energy
                if not np.all(np.isfinite(audio_array)):
                     energy = 0.0 # Treat non-finite values as silence
                elif len(audio_array) > 0:
                     energy = np.sqrt(np.mean(audio_array**2))
                else:
                     energy = 0.0
            except Exception as e:
                 logger.error(f"Error calculating energy: {e}")
                 energy = 0.0 # Treat error as silence


            # --- Speech Detection Logic ---
            if not self.is_speech_active and energy > self.energy_threshold:
                # Speech start detected
                self.is_speech_active = True
                # Mark start relative to the beginning of the *current* buffer content
                self.speech_start_idx = max(0, len(self.audio_buffer) - len(audio_bytes))
                self.silence_counter = 0
                logger.debug(f"Speech start detected (energy: {energy:.6f}) at buffer index {self.speech_start_idx}")

                # --- Interrupt Logic ---
                # Check if the assistant is currently responding
                is_responding = await self.get_responding_state()
                if is_responding:
                    # Debounce interrupts: Only cancel if enough time passed since last one
                    now = time.monotonic()
                    if now - self.last_interrupt_time > 1.0: # Only interrupt if > 1s since last
                         logger.info("User speech detected while assistant responding. Cancelling assistant.")
                         self.last_interrupt_time = now
                         # Run cancellation concurrently, don't block VAD
                         asyncio.create_task(self.cancel_current_processing())
                    else:
                         logger.debug("Ignoring potential interrupt signal (too soon after last).")


            elif self.is_speech_active:
                current_speech_len_bytes = len(self.audio_buffer) - self.speech_start_idx
                current_speech_len_samples = current_speech_len_bytes // self.bytes_per_sample

                if energy > self.energy_threshold:
                    # Continued speech
                    self.silence_counter = 0
                else:
                    # Potential end of speech (silence)
                    self.silence_counter += num_new_samples

                    # Check if enough silence to end speech segment
                    if self.silence_counter >= self.silence_samples:
                        speech_end_idx = len(self.audio_buffer) - (self.silence_counter * self.bytes_per_sample)
                        # Ensure start index is valid and before end index
                        if self.speech_start_idx >= 0 and self.speech_start_idx < speech_end_idx:
                            speech_segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])
                            segment_len_samples = len(speech_segment_bytes) // self.bytes_per_sample

                            # Reset VAD state for next detection
                            self.is_speech_active = False
                            self.silence_counter = 0
                            # Keep only the trailing silence part in the buffer
                            self.audio_buffer = self.audio_buffer[speech_end_idx:]
                            self.speech_start_idx = 0 # Reset relative start index


                            # Only queue if speech segment is within valid duration
                            if segment_len_samples >= self.min_speech_samples:
                                logger.info(f"Speech segment detected (silence end): {segment_len_samples / self.sample_rate:.2f}s")
                                await self.segment_queue.put(speech_segment_bytes)
                                return speech_segment_bytes # Indicate segment found
                            else:
                                logger.debug(f"Speech segment too short (silence end): {segment_len_samples / self.sample_rate:.2f}s. Discarding.")
                        else:
                             logger.warning(f"Invalid VAD indices on silence end: start={self.speech_start_idx}, end={speech_end_idx}. Resetting buffer.")
                             self.is_speech_active = False
                             self.silence_counter = 0
                             self.audio_buffer = bytearray() # Clear buffer on inconsistency
                             self.speech_start_idx = 0


                # Check if speech segment exceeds maximum duration (Force cut)
                # This check needs self.is_speech_active to be true
                if self.is_speech_active and current_speech_len_samples > self.max_speech_samples:
                    speech_end_idx = self.speech_start_idx + (self.max_speech_samples * self.bytes_per_sample)
                    # Clamp end index to current buffer length
                    speech_end_idx = min(speech_end_idx, len(self.audio_buffer))

                    if self.speech_start_idx < speech_end_idx:
                        speech_segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])
                        segment_len_samples = len(speech_segment_bytes) // self.bytes_per_sample

                        logger.info(f"Max duration speech segment cut: {segment_len_samples / self.sample_rate:.2f}s")

                        # Update buffer: Keep audio *after* the cut segment
                        self.audio_buffer = self.audio_buffer[speech_end_idx:]
                        # Reset relative start index for the *new* buffer content
                        self.speech_start_idx = 0
                        # Reset silence counter as we forced a cut, maybe speech continues
                        self.silence_counter = 0
                        # Keep self.is_speech_active = True if audio remains? Or reset?
                        # Let's reset it and let next chunk re-detect if energy is high
                        # self.is_speech_active = False # Re-evaluate on next chunk

                        # Queue the cut segment
                        await self.segment_queue.put(speech_segment_bytes)
                        return speech_segment_bytes # Indicate segment found
                    else:
                         logger.warning(f"Invalid VAD indices on max duration cut: start={self.speech_start_idx}, end={speech_end_idx}. Resetting buffer.")
                         self.is_speech_active = False
                         self.silence_counter = 0
                         self.audio_buffer = bytearray()
                         self.speech_start_idx = 0


        return None # No segment finalized in this call

    async def get_next_segment(self):
        """Get the next available speech segment from the queue"""
        try:
            # Use a small timeout to prevent blocking indefinitely
            return await asyncio.wait_for(self.segment_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None

# --- Model Processors (Singleton Pattern) ---

class WhisperTranscriber:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if hasattr(self, 'pipe') and self.pipe: return # Avoid re-init
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Whisper using device: {self.device}")
        self.torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
        model_id = "openai/whisper-large-v3"
        logger.info(f"Loading Whisper model: {model_id}...")
        self.pipe = None
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.pipe = pipeline(
                "automatic-speech-recognition", model=self.model, tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor, torch_dtype=self.torch_dtype,
                device=self.device, chunk_length_s=30, stride_length_s=[4, 2] # Use chunking
            )
            logger.info("Whisper model ready.")
            self.transcription_count = 0
        except Exception as e:
             logger.error(f"FATAL: Failed to load Whisper model: {e}", exc_info=True)
             raise RuntimeError(f"Could not initialize Whisper model: {e}") from e

    async def transcribe(self, audio_bytes, sample_rate=SAMPLE_RATE):
        if not self.pipe or not audio_bytes: return ""
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_array) < sample_rate * 0.2: return "" # Skip very short

            # Ensure generate_kwargs are passed correctly
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.pipe(
                    audio_array.copy(), # Pass copy to avoid potential issues
                    batch_size=8, # Adjust based on VRAM
                    generate_kwargs={ # Pass generate_kwargs here
                        "task": "transcribe",
                        "language": "english",
                        "temperature": 0.0
                    }
                )
            )
            text = result.get("text", "").strip()
            self.transcription_count += 1
            logger.info(f"Transcription #{self.transcription_count}: '{text}'")
            return text
        except Exception as e:
            logger.error(f"Transcription error: {e}\n{traceback.format_exc()}")
            return ""

class GemmaMultimodalProcessor:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if hasattr(self, 'model') and self.model: return # Avoid re-init
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Gemma using device: {self.device}")
        model_id = "google/gemma-2-9b-it" # Or your preferred Gemma model
        logger.info(f"Loading Gemma model: {model_id}...")
        self.model = None
        self.processor = None
        try:
            # Quantization for Gemma 2 9B
            quantization_config = BitsAndBytesConfig(
                 load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = GemmaForConditionalGeneration.from_pretrained(
                model_id, device_map="auto", quantization_config=quantization_config,
                torch_dtype=torch.bfloat16 # Match compute dtype
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            logger.info("Gemma model ready.")

            self.last_image = None
            self.image_lock = asyncio.Lock()
            self.message_history = []
            self.max_history_len = 4 # Turns (User + Assistant = 1 turn) -> 2 turns history
            self.history_lock = asyncio.Lock()
            self.generation_count = 0
            self.system_prompt = """You are a helpful assistant providing spoken responses about images and engaging in natural conversation. Keep responses concise, fluent, and conversational (1-3 short sentences). Use natural language suitable for speaking aloud.

Guidelines:
1. If the user asks about the image, describe relevant parts concisely.
2. If the user's input isn't about the image, respond naturally without forcing image descriptions.
3. If unsure about the image context, ask for clarification politely (e.g., "What about the image would you like to know?").
4. Maintain conversation context. Refer to previous turns naturally if needed.
5. Avoid overly long or complex sentences."""

        except Exception as e:
             logger.error(f"FATAL: Failed to load Gemma model: {e}", exc_info=True)
             raise RuntimeError(f"Could not initialize Gemma model: {e}") from e

    async def set_image(self, image_data):
        async with self.image_lock:
            try:
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                self.last_image = image
                logger.info(f"New image received. Size: {image.size}")
                async with self.history_lock:
                    self.message_history = [] # Clear history on new image
                    logger.info("Message history cleared due to new image.")
                return True
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                self.last_image = None
                return False

    async def _build_chat(self, text):
        """Build the chat structure for Gemma, including history and image."""
        async with self.history_lock:
            # Start with system prompt (if model supports it well in chat format)
            # Note: Gemma IT models expect history starting with 'user'
            chat = list(self.message_history) # Get current history

            # Add current user turn
            content = []
            async with self.image_lock: # Access image under its lock
                if self.last_image:
                    content.append(self.last_image) # Add PIL image object
                else:
                    logger.warning("Generating response without image context in prompt.")
            content.append(text) # Add user text
            chat.append({"role": "user", "content": content})
        return chat

    async def update_history(self, user_text, assistant_response):
        """Update message history, ensuring it doesn't exceed max length."""
        async with self.history_lock:
            # Add user message (text only for history)
            self.message_history.append({"role": "user", "content": user_text})
            # Add assistant response
            self.message_history.append({"role": "assistant", "content": assistant_response})
            # Trim history: Keep the last N messages (max_history_len * 2)
            self.message_history = self.message_history[-(self.max_history_len * 2):]
            logger.debug(f"History updated. Length: {len(self.message_history)}")

    async def generate(self, text):
        if not self.model or not self.processor:
             return "Error: Gemma model not ready."

        chat = await self._build_chat(text)
        try:
             # Prepare inputs using the processor
            prompt = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=self.last_image, return_tensors="pt").to(self.model.device)

            # Generate response using run_in_executor
            generate_ids = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
            )

            # Decode the generated tokens (excluding input)
            output_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            generated_text = self.processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

            # Update history *only after* successful generation
            # Important: Pass the original user text, not the full prompt
            await self.update_history(text, generated_text)

            self.generation_count += 1
            logger.info(f"Gemma generation #{self.generation_count} successful ({len(generated_text)} chars)")

            # Optional: Clear cache if memory issues persist
            # torch.cuda.empty_cache()

            return generated_text

        except Exception as e:
            logger.error(f"Gemma generation error: {e}\n{traceback.format_exc()}")
            # torch.cuda.empty_cache() # Optional cache clear on error
            return "Sorry, I encountered an error generating a response."

class KokoroTTSProcessor:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if hasattr(self, 'pipeline') and self.pipeline: return # Avoid re-init
        logger.info("Initializing Kokoro TTS processor...")
        self.pipeline = None
        self.is_ready = False
        self.target_sample_rate = None # Store expected sample rate
        try:
            # Check Kokoro's expected language codes and voices
            self.pipeline = KPipeline(lang_code='en') # Assuming English focus
            self.default_voice = 'en-US-Standard-F' # Example standard voice, check available ones
            # Attempt a dummy synthesis to check readiness and get sample rate
            dummy_audio = self.pipeline.synthesize("test", voice=self.default_voice)
            if isinstance(dummy_audio, np.ndarray) and dummy_audio.size > 0:
                 self.target_sample_rate = self.pipeline.target_sample_rate # Store rate
                 logger.info(f"Kokoro TTS processor initialized successfully. Voice: {self.default_voice}, Rate: {self.target_sample_rate}Hz")
                 self.is_ready = True
                 self.synthesis_count = 0
            else:
                 logger.error("Kokoro TTS dummy synthesis failed.")
                 self.pipeline = None # Mark as unusable

        except ImportError:
             logger.error("Kokoro library not found. Please install it.")
        except Exception as e:
            logger.error(f"Error initializing Kokoro TTS: {e}", exc_info=True)
            self.pipeline = None

    async def synthesize_speech(self, text):
        if not self.is_ready or not text:
            logger.warning(f"TTS skipped. Ready: {self.is_ready}, Text provided: {bool(text)}")
            return None

        try:
            start_time = time.time()
            logger.info(f"Synthesizing speech for text (first 50): '{text[:50]}...'")

            # Use run_in_executor for the blocking Kokoro call
            audio_data = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.pipeline.synthesize(text, voice=self.default_voice, speed=1.0)
            )

            if isinstance(audio_data, np.ndarray) and audio_data.size > 0:
                self.synthesis_count += 1
                duration = len(audio_data) / self.target_sample_rate if self.target_sample_rate else 'N/A'
                elapsed = time.time() - start_time
                logger.info(f"Speech synthesis #{self.synthesis_count} complete. Samples: {len(audio_data)}, Est. Duration: {duration}s, Time: {elapsed:.2f}s")

                # Ensure float32 for consistency before converting to int16
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                    # Normalize if it wasn't already in [-1, 1] range
                    max_abs = np.max(np.abs(audio_data))
                    if max_abs > 1.0:
                         audio_data /= max_abs

                # torch.cuda.empty_cache() # Optional
                return audio_data
            else:
                 logger.warning("Speech synthesis resulted in empty or invalid audio.")
                 return None

        except Exception as e:
            logger.error(f"Speech synthesis error: {e}\n{traceback.format_exc()}")
            # torch.cuda.empty_cache() # Optional
            return None


# --- WebSocket Handler & Pipeline ---

async def run_full_response_pipeline(speech_segment, detector, transcriber, gemma_processor, tts_processor, websocket):
     """Handles the ASR -> LLM -> TTS pipeline for a single speech segment."""
     # This function now runs as its own task, managed by detector.current_processing_task

     try:
          # 1. Mark start of processing (state already set before task creation)
          logger.info("Starting response pipeline...")

          # 2. Transcribe Speech
          transcription = await transcriber.transcribe(speech_segment)
          if not transcription:
               logger.info("Skipping empty transcription.")
               return # End pipeline early

          # --- Basic Filtering ---
          cleaned_transcription = re.sub(r'[^\w\s]', '', transcription).lower()
          words = [w for w in cleaned_transcription.split() if w]
          common_fillers = {'yes', 'no', 'ok', 'okay', 'um', 'uh', 'yeah', 'hmm', 'bye', 'hi', 'hello'}
          if not words or (len(words) <= 1 and words[0] in common_fillers):
               logger.info(f"Skipping filtered transcription: '{transcription}'")
               return # End pipeline early

          # 3. Send Interrupt Signal to Client (optional but good practice)
          # Tells client to stop playing any previous audio immediately
          logger.info("Sending interrupt signal to client before new response.")
          try:
               await websocket.send(json.dumps({"interrupt": True}))
          except websockets.exceptions.ConnectionClosed:
               logger.warning("Cannot send interrupt, connection closed.")
               return # Cannot continue if connection is closed

          # 4. Generate Response using Gemma
          logger.info("Generating response with Gemma...")
          generated_text = await gemma_processor.generate(transcription) # Pass original transcription
          if not generated_text or "error" in generated_text.lower():
               logger.error(f"Gemma generation failed or returned error: '{generated_text}'")
               # TODO: Optionally send a generic fallback audio message
               return

          # 5. Synthesize Speech using Kokoro TTS
          logger.info("Synthesizing speech with Kokoro TTS...")
          audio_response = await tts_processor.synthesize_speech(generated_text)

          # 6. Send Audio Response (if synthesis successful)
          if audio_response is not None and audio_response.size > 0:
               try:
                    # Convert float32 numpy array [-1, 1] to int16 bytes
                    audio_int16 = np.clip(audio_response * 32767, -32768, 32767).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()
                    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

                    logger.info(f"Sending synthesized audio ({len(audio_bytes)} bytes) to client.")
                    await websocket.send(json.dumps({"audio": base64_audio}))
                    logger.info("Audio sent successfully.")

               except websockets.exceptions.ConnectionClosed:
                    logger.warning("Connection closed before audio could be sent.")
               except Exception as send_err:
                    logger.error(f"Error sending audio to client: {send_err}")
          else:
               logger.warning("TTS synthesis failed or produced empty audio, no audio response sent.")

     except asyncio.CancelledError:
          # This specific task was cancelled (likely due to interruption)
          logger.info("Response pipeline task was cancelled.")
          # State is reset in the finally block below
          raise # Re-raise cancellation to be handled by the caller if needed

     except Exception as e:
          logger.error(f"Unhandled error in response pipeline: {e}\n{traceback.format_exc()}")

     finally:
          # 7. VERY IMPORTANT: Reset State
          # This runs whether the task completes, is cancelled, or errors out.
          logger.info("Response pipeline task finishing. Resetting responding state.")
          # Call the detector's method to reset state and clear the task reference
          await detector.set_assistant_responding(False)


async def process_speech_segments(detector, transcriber, gemma_processor, tts_processor, websocket):
    """Continuously checks for and processes detected speech segments."""
    while True:
        try:
            # Check connection state first
            if websocket.closed:
                 logger.info("WebSocket closed, stopping segment processing.")
                 break

            speech_segment = await detector.get_next_segment()

            if speech_segment:
                # Check if the assistant is *already* responding to a *previous* segment.
                # If yes, discard this new segment (the VAD interrupt handles the ongoing one).
                is_already_responding = await detector.get_responding_state()
                if is_already_responding:
                    logger.info("New segment detected, but assistant is busy with previous one. Discarding new segment.")
                    continue # Skip this segment

                # --- Start Processing the New Segment ---
                # Mark assistant as responding and store the new task
                logger.info("Creating new task for response pipeline.")
                pipeline_task = asyncio.create_task(
                    run_full_response_pipeline(
                        speech_segment, detector, transcriber, gemma_processor, tts_processor, websocket
                    ),
                    name=f"ResponsePipeline-{time.time()}" # Give task a name for logging
                )
                # Pass the task object to the state setter
                await detector.set_assistant_responding(True, pipeline_task)

            # Small sleep even if no segment, prevents tight loop, allows other tasks to run
            await asyncio.sleep(0.02)

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed during segment processing loop.")
            break # Exit loop
        except asyncio.CancelledError:
             logger.info("Segment processing task cancelled.")
             break # Exit loop
        except Exception as e:
            logger.error(f"Error in speech segment processing loop: {e}\n{traceback.format_exc()}")
            # Reset state just in case something went very wrong
            await detector.set_assistant_responding(False)
            await asyncio.sleep(1) # Pause after error

    logger.info("Segment processing loop finished.")


async def receive_messages(detector, gemma_processor, websocket):
    """Handles incoming WebSocket messages (audio chunks, images)."""
    while True:
        try:
            message = await websocket.recv()
            data = json.loads(message)

            is_responding_now = await detector.get_responding_state()

            # Handle image data (only process if assistant is NOT responding)
            if "image" in data:
                if not is_responding_now:
                    try:
                        image_data = base64.b64decode(data["image"])
                        logger.info("Received standalone image data.")
                        await gemma_processor.set_image(image_data)
                    except (base64.binascii.Error, ValueError, TypeError) as decode_err:
                        logger.error(f"Error decoding base64 image: {decode_err}")
                    except Exception as img_err:
                        logger.error(f"Error processing received image: {img_err}")
                else:
                    logger.info("Ignoring image received while assistant is responding.")


            # Handle audio data (always add to buffer)
            # Ensure the key matches what the client sends (e.g., "audio_data")
            elif "audio_data" in data:
                try:
                    audio_bytes = base64.b64decode(data["audio_data"])
                    # logger.debug(f"Received audio chunk: {len(audio_bytes)} bytes")
                    await detector.add_audio(audio_bytes)
                except (base64.binascii.Error, ValueError, TypeError) as decode_err:
                     logger.error(f"Error decoding base64 audio: {decode_err}")
                except Exception as audio_err:
                     logger.error(f"Error processing received audio: {audio_err}")

            # Handle explicit interrupt signal from client (if implemented)
            elif "interrupt" in data and data["interrupt"] is True:
                 if is_responding_now:
                      logger.info("Received explicit interrupt request from client.")
                      # Debounce like the VAD interrupt
                      now = time.monotonic()
                      if now - detector.last_interrupt_time > 1.0:
                           detector.last_interrupt_time = now
                           asyncio.create_task(detector.cancel_current_processing())

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed during message receive loop.")
            break # Exit loop cleanly
        except json.JSONDecodeError:
            logger.error(f"Received invalid JSON: {message[:200]}...")
        except asyncio.CancelledError:
             logger.info("Receive messages task cancelled.")
             break # Exit loop
        except Exception as e:
            logger.error(f"Error processing received message: {e}\n{traceback.format_exc()}")
            # Decide if the loop should continue or break on other errors

    logger.info("Receive message loop finished.")

async def send_keepalive(websocket):
    """Sends WebSocket pings periodically to keep connection alive."""
    while True:
        try:
            await websocket.ping()
            await asyncio.sleep(15) # Send ping every 15 seconds
        except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
            logger.info("Keepalive task stopped.")
            break
        except Exception as e:
             logger.error(f"Error in keepalive task: {e}")
             await asyncio.sleep(5) # Wait before retrying after an error


async def handle_client(websocket):
    """Main handler for a single WebSocket client connection."""
    client_ip = websocket.remote_address
    logger.info(f"Client connected from {client_ip}")

    # Initialize components for this client session
    # Singletons are fetched, detector is instantiated per client
    try:
        detector = AudioSegmentDetector()
        transcriber = WhisperTranscriber.get_instance()
        gemma_processor = GemmaMultimodalProcessor.get_instance()
        tts_processor = KokoroTTSProcessor.get_instance()

        # Crucial check: Ensure TTS is actually ready before proceeding
        if not tts_processor.is_ready:
             logger.error("TTS Processor is not ready. Cannot provide audio responses for this client.")
             await websocket.close(code=1011, reason="TTS service unavailable") # Close gracefully
             return
    except Exception as init_err:
         logger.error(f"Failed to initialize components for client {client_ip}: {init_err}")
         try:
             await websocket.close(code=1011, reason="Server component initialization failed")
         except: pass # Ignore errors during close
         return


    # --- Background Tasks for this client ---
    receive_task = None
    segment_proc_task = None
    keepalive_task = None
    all_tasks = []

    try:
        # Create tasks for handling different aspects of the connection
        receive_task = asyncio.create_task(receive_messages(detector, gemma_processor, websocket), name="Receiver")
        segment_proc_task = asyncio.create_task(process_speech_segments(detector, transcriber, gemma_processor, tts_processor, websocket), name="SegmentProcessor")
        keepalive_task = asyncio.create_task(send_keepalive(websocket), name="Keepalive")
        all_tasks = [receive_task, segment_proc_task, keepalive_task]

        # Wait for any task to complete (usually receive_task on disconnect)
        done, pending = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)

        logger.info(f"A client task finished for {client_ip}. Done: {[t.get_name() for t in done]}. Pending: {[t.get_name() for t in pending]}")

        # Log results/exceptions from the completed task(s)
        for task in done:
            try:
                result = task.result()
                logger.info(f"Task {task.get_name()} completed successfully.")
            except asyncio.CancelledError:
                 logger.info(f"Task {task.get_name()} was cancelled.")
            except Exception as e:
                 logger.error(f"Task {task.get_name()} failed with exception: {e}", exc_info=True)


    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Client {client_ip} connection closed: {e.code} {e.reason}")
    except Exception as e:
        logger.error(f"Unhandled exception in client handler for {client_ip}: {e}", exc_info=True)
    finally:
        logger.info(f"Cleaning up client session for {client_ip}")

        # 1. Cancel any ongoing processing pipeline task explicitly
        # Use a direct call, as detector instance might be gone if error was early
        try:
             await detector.cancel_current_processing()
             logger.info("Ensured any active processing task is cancelled.")
        except Exception as cancel_err:
             logger.error(f"Error during final cancellation check: {cancel_err}")


        # 2. Cancel all background tasks associated with this client
        logger.info("Cancelling remaining client tasks...")
        for task in all_tasks: # Use the list we created
             if task and not task.done():
                  task.cancel()

        # 3. Wait briefly for tasks to acknowledge cancellation
        if all_tasks:
             await asyncio.wait(all_tasks, timeout=1.0) # Wait max 1s

        logger.info(f"Client {client_ip} cleanup complete.")


async def main():
    """Initializes models and starts the WebSocket server."""
    logger.info("Initializing models...")
    try:
        # Pre-initialize singleton instances
        WhisperTranscriber.get_instance()
        GemmaMultimodalProcessor.get_instance()
        KokoroTTSProcessor.get_instance() # This also checks TTS readiness
        logger.info("Models initialized (or attempted).")
    except Exception as init_err:
         logger.error(f"FATAL: Core model initialization failed: {init_err}", exc_info=True)
         sys.exit(1)

    logger.info(f"Starting WebSocket server on 0.0.0.0:{WEBSOCKET_PORT}")
    try:
        # Specify longer timeouts if needed, defaults are often okay
        async with websockets.serve(
            handle_client,
            "0.0.0.0",
            WEBSOCKET_PORT,
            ping_interval=20,
            ping_timeout=40,
            close_timeout=10
        ):
            logger.info(f"WebSocket server running on ws://0.0.0.0:{WEBSOCKET_PORT}")
            await asyncio.Future()  # Run forever
    except OSError as e:
         logger.error(f"Server error: Could not bind to address/port ({e}). Check if port {WEBSOCKET_PORT} is already in use.")
    except Exception as e:
        logger.error(f"Server startup error: {e}", exc_info=True)

if __name__ == "__main__":
    # Optional: Increase asyncio debug logging
    # asyncio.get_event_loop().set_debug(True)
    # logging.getLogger('asyncio').setLevel(logging.DEBUG)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped manually.")
    except Exception as main_err:
        logger.critical(f"Unhandled exception in main execution: {main_err}", exc_info=True)
