import asyncio
import json
import websockets
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoProcessor, GemmaForConditionalGeneration # Corrected import
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
    format='%(asctime)s - %(levelname)s - %(module)s - %(lineno)d - %(message)s', # More detail
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Constants ---
SAMPLE_RATE = 16000
ENERGY_THRESHOLD = 0.015  # Adjust based on mic sensitivity
SILENCE_DURATION = 0.7    # Shorter silence to feel more responsive
MIN_SPEECH_DURATION = 0.5 # Shorter min duration
MAX_SPEECH_DURATION = 10  # Shorter max duration for faster turns

class AudioSegmentDetector:
    """Detects speech segments based on audio energy levels"""

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

        # Internal state
        self.audio_buffer = bytearray()
        self.is_speech_active = False
        self.silence_counter = 0
        self.speech_start_idx = 0
        self.lock = asyncio.Lock() # Protects buffer access
        self.segment_queue = asyncio.Queue()

        # Counters
        self.segments_detected = 0

        # Assistant response control state
        self.assistant_is_responding = False
        self.state_lock = asyncio.Lock() # Protects assistant_is_responding and current_processing_task
        self.current_processing_task = None

    async def set_assistant_responding(self, is_responding):
        """Set assistant response state and manage task reference"""
        async with self.state_lock:
            self.assistant_is_responding = is_responding
            # Clear task reference when assistant stops responding
            if not is_responding:
                 self.current_processing_task = None
            logger.info(f"Assistant responding state set to: {is_responding}")

    async def set_current_task(self, task):
        """Set the current processing task."""
        async with self.state_lock:
            self.current_processing_task = task

    async def cancel_current_processing(self):
        """Cancel any ongoing generation and TTS task"""
        async with self.state_lock:
            task_to_cancel = self.current_processing_task
            self.current_processing_task = None # Clear immediately
            is_responding = self.assistant_is_responding # Get current state under lock

        if task_to_cancel and not task_to_cancel.done():
            logger.info("Attempting to cancel ongoing processing task.")
            task_to_cancel.cancel()
            try:
                # Give cancellation a moment to propagate
                await asyncio.wait_for(task_to_cancel, timeout=0.5)
            except asyncio.CancelledError:
                logger.info("Ongoing processing task cancelled successfully.")
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for task cancellation.")
            except Exception as e:
                # Log other exceptions during await if they occur (less common)
                 logger.error(f"Unexpected error during task cancellation await: {e}")
            finally:
                 # Ensure state is reset even if cancellation had issues
                if is_responding: # Only reset if it was True before cancellation attempt
                     await self.set_assistant_responding(False)
        elif is_responding:
             # If there was no task but state was responding, reset state
             logger.info("No active task found, resetting responding state.")
             await self.set_assistant_responding(False)


    async def add_audio(self, audio_bytes):
        """Add audio data to the buffer and check for speech segments"""
        async with self.lock:
            self.audio_buffer.extend(audio_bytes)
            buffer_len_samples = len(self.audio_buffer) // self.bytes_per_sample

            # Trim buffer if it gets excessively long (e.g., > 30 seconds) to prevent memory issues
            max_buffer_samples = 30 * self.sample_rate
            if buffer_len_samples > max_buffer_samples:
                 trim_amount = (buffer_len_samples - max_buffer_samples) * self.bytes_per_sample
                 self.audio_buffer = self.audio_buffer[trim_amount:]
                 # Adjust speech_start_idx if necessary
                 self.speech_start_idx = max(0, self.speech_start_idx - trim_amount)
                 logger.warning(f"Audio buffer trimmed to {max_buffer_samples / self.sample_rate}s")
                 buffer_len_samples = max_buffer_samples # Update length after trim


            # Use only the newly added audio for energy calculation for efficiency
            num_new_samples = len(audio_bytes) // self.bytes_per_sample
            if num_new_samples == 0:
                return None

            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            energy = np.sqrt(np.mean(audio_array**2))

            # --- Speech Detection Logic ---
            if not self.is_speech_active and energy > self.energy_threshold:
                # Speech start detected
                self.is_speech_active = True
                # Mark start index relative to the beginning of the *current* buffer
                self.speech_start_idx = max(0, len(self.audio_buffer) - len(audio_bytes))
                self.silence_counter = 0
                logger.debug(f"Speech start detected (energy: {energy:.6f})")

                # --- Crucial Interrupt Logic ---
                # If assistant is responding when user starts talking, cancel it.
                async with self.state_lock: # Check state under lock
                     should_interrupt = self.assistant_is_responding
                if should_interrupt:
                    logger.info("User speech detected while assistant responding. Cancelling assistant.")
                    await self.cancel_current_processing() # Cancel ongoing task

            elif self.is_speech_active:
                current_speech_len_samples = (len(self.audio_buffer) - self.speech_start_idx) // self.bytes_per_sample

                if energy > self.energy_threshold:
                    # Continued speech
                    self.silence_counter = 0
                else:
                    # Potential end of speech
                    self.silence_counter += num_new_samples

                    # Check if enough silence to end speech segment
                    if self.silence_counter >= self.silence_samples:
                        speech_end_idx = len(self.audio_buffer) - (self.silence_counter * self.bytes_per_sample)
                        # Ensure start index is not past end index
                        if self.speech_start_idx < speech_end_idx:
                            speech_segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])
                            segment_len_samples = len(speech_segment_bytes) // self.bytes_per_sample

                            # Reset for next speech detection
                            self.is_speech_active = False
                            self.silence_counter = 0
                            # Keep only the silence part in the buffer
                            self.audio_buffer = self.audio_buffer[speech_end_idx:]
                            self.speech_start_idx = 0 # Reset relative start index


                            # Only process if speech segment is within valid duration
                            if segment_len_samples >= self.min_speech_samples:
                                self.segments_detected += 1
                                logger.info(f"Speech segment detected (silence): {segment_len_samples / self.sample_rate:.2f}s")
                                await self.segment_queue.put(speech_segment_bytes)
                                return speech_segment_bytes # Indicate segment found
                            else:
                                logger.debug(f"Speech segment too short (silence): {segment_len_samples / self.sample_rate:.2f}s. Discarding.")
                        else:
                             logger.warning("Speech start index was past end index after silence. Resetting.")
                             self.is_speech_active = False
                             self.silence_counter = 0
                             self.audio_buffer = bytearray() # Clear buffer on inconsistency
                             self.speech_start_idx = 0


                # Check if speech segment exceeds maximum duration (Force cut)
                if self.is_speech_active and current_speech_len_samples > self.max_speech_samples:
                    speech_end_idx = self.speech_start_idx + (self.max_speech_samples * self.bytes_per_sample)
                    speech_segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])
                    segment_len_samples = len(speech_segment_bytes) // self.bytes_per_sample

                    logger.info(f"Max duration speech segment cut: {segment_len_samples / self.sample_rate:.2f}s")

                    # Update buffer and start index for potential continuation
                    self.audio_buffer = self.audio_buffer[speech_end_idx:]
                    self.speech_start_idx = 0 # Start index is now start of the new buffer

                    # Process the cut segment
                    self.segments_detected += 1
                    await self.segment_queue.put(speech_segment_bytes)
                    # Continue detecting immediately in case speech continues right after cut
                    self.silence_counter = 0 # Reset silence counter
                    return speech_segment_bytes # Indicate segment found

        return None # No segment finalized in this call

    async def get_next_segment(self):
        """Get the next available speech segment"""
        try:
            # Short timeout allows loop to check state frequently
            return await asyncio.wait_for(self.segment_queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            return None

class WhisperTranscriber:
    """Handles speech transcription using Whisper large-v3 model with pipeline"""
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if hasattr(self, 'pipe'): # Avoid re-initialization
            return
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Whisper using device: {self.device}")
        self.torch_dtype = torch.float16 if self.device != "cpu" else torch.float32

        model_id = "openai/whisper-large-v3" # Using large-v3 directly
        logger.info(f"Loading Whisper model: {model_id}...")

        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True if self.device != "cpu" else False, # Only use on GPU
                use_safetensors=True
            )
            self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
                chunk_length_s=30, # Process in chunks
                stride_length_s=[4, 2] # Overlap chunks
            )
            logger.info("Whisper model ready for transcription.")
            self.transcription_count = 0
        except Exception as e:
             logger.error(f"FATAL: Failed to load Whisper model: {e}", exc_info=True)
             # Exit if core model fails to load
             sys.exit(f"Could not initialize Whisper model: {e}")


    async def transcribe(self, audio_bytes, sample_rate=SAMPLE_RATE):
        """Transcribe audio bytes to text using the pipeline"""
        if not audio_bytes:
            return ""
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_array) < sample_rate * 0.2: # Ignore very short segments < 200ms
                 logger.debug("Audio segment too short for transcription, skipping.")
                 return ""

            # Use the pipeline to transcribe - run in executor
            # Use generate_kwargs for better control (especially temperature)
            result = await asyncio.get_event_loop().run_in_executor(
                None, # Default executor
                lambda: self.pipe(
                    audio_array, # Pass array directly
                    batch_size=4, # Adjust based on VRAM
                     generate_kwargs={
                         "task": "transcribe",
                         "language": "english", # Force English
                         "temperature": 0.0 # Force deterministic output
                     }
                )
            )

            text = result.get("text", "").strip()
            self.transcription_count += 1
            logger.info(f"Transcription #{self.transcription_count}: '{text}'")
            return text

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return ""

class GemmaMultimodalProcessor:
    """Handles multimodal generation using Gemma model"""
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if hasattr(self, 'model'): # Avoid re-initialization
             return
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Gemma using device: {self.device}")

        model_id = "google/gemma-2-9b-it" # Using Gemma 2 9B IT
        logger.info(f"Loading Gemma model: {model_id}...")

        try:
            # Use 4-bit quantization for Gemma 2 9B
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                 load_in_4bit=True,
                 bnb_4bit_compute_dtype=torch.bfloat16
            )

            self.model = GemmaForConditionalGeneration.from_pretrained( # Use corrected class
                model_id,
                device_map="auto", # Let transformers handle device mapping
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16 # Consistent dtype
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            logger.info("Gemma model ready for multimodal generation.")

            self.last_image = None
            self.last_image_timestamp = 0
            self.image_lock = asyncio.Lock() # Protect image access/update

            # Message history management
            self.message_history = []
            self.max_history_len = 4 # Keep last 2 user, 2 assistant
            self.history_lock = asyncio.Lock() # Protect history access/update

            self.generation_count = 0
            self.system_prompt = """You are a helpful assistant providing spoken responses about images and engaging in natural conversation. Keep your responses concise, fluent, and conversational (1-3 short sentences). Use natural language suitable for speaking aloud.

Guidelines:
1. If the user asks about the image, describe relevant parts concisely.
2. If the user's input isn't about the image, respond naturally without forcing image descriptions.
3. If unsure about the image context, ask for clarification politely (e.g., "What about the image would you like to know?").
4. Maintain conversation context. Refer to previous turns naturally if needed.
5. Avoid overly long or complex sentences."""

        except Exception as e:
             logger.error(f"FATAL: Failed to load Gemma model: {e}", exc_info=True)
             # Exit if core model fails to load
             sys.exit(f"Could not initialize Gemma model: {e}")

    async def set_image(self, image_data):
        """Cache the most recent image received"""
        async with self.image_lock:
            try:
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                # Optional: Resize if images are consistently too large
                # max_size = (1024, 1024)
                # image.thumbnail(max_size, Image.Resampling.LANCZOS)

                self.last_image = image
                self.last_image_timestamp = time.time()
                logger.info(f"New image received and processed. Size: {image.size}")

                # Clear history when a new image is provided
                async with self.history_lock:
                    self.message_history = []
                    logger.info("Message history cleared due to new image.")
                return True
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                self.last_image = None # Ensure consistency
                return False

    async def _build_prompt(self, text):
        """Build the prompt string with history for the model"""
        async with self.history_lock:
            # Start with the system prompt
            chat = [{"role": "user", "content": self.system_prompt}] # Start fresh for prompt building

            # Add historical turns
            chat.extend(self.message_history)

            # Add current user turn (with image if available)
            content = []
            async with self.image_lock: # Access image under its lock
                if self.last_image:
                    content.append(self.last_image) # Add image object directly
                else:
                    logger.warning("Generating response without image context.")
            content.append(text)
            chat.append({"role": "user", "content": content})

        # Use processor to apply chat template
        try:
             # Important: Don't add generation prompt here for Gemma, it's added by the template
            prompt = self.processor.apply_chat_template(chat, tokenize=False)
            return prompt
        except Exception as e:
             logger.error(f"Error applying chat template: {e}")
             # Fallback to simple text if template fails
             return text


    async def update_history(self, user_text, assistant_response):
        """Update message history with new exchange"""
        async with self.history_lock:
            # Add user message (text only for history)
            self.message_history.append({"role": "user", "content": user_text})
            # Add assistant response
            self.message_history.append({"role": "assistant", "content": assistant_response})
            # Trim history
            if len(self.message_history) > self.max_history_len:
                # Keep the last max_history_len turns
                self.message_history = self.message_history[-self.max_history_len:]
            # logger.debug(f"History updated. Length: {len(self.message_history)}")


    async def generate(self, text):
        """Generate a response using the latest image and text input (non-streaming)"""
        prompt = await self._build_prompt(text)
        if not prompt:
             return "Sorry, I had trouble understanding that."

        try:
            # Tokenize the prompt
            inputs = self.processor(text=prompt, images=self.last_image, return_tensors="pt").to(self.model.device, dtype=self.model.dtype) # Ensure matching dtype

            # Generate response
            # Use run_in_executor for the blocking generate call
            generate_ids = await asyncio.get_event_loop().run_in_executor(
                None, # Default executor
                lambda: self.model.generate(
                    **inputs,
                    max_new_tokens=150, # Allow slightly longer for complex descriptions
                    do_sample=True,
                    temperature=0.7,
                    top_k=40,
                    top_p=0.9,
                )
            )

            # Decode the generated tokens
            # The generated IDs include the input, so slice it off
            output_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            generated_text = self.processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            generated_text = generated_text.strip() # Clean up output

            # Update conversation history *after* successful generation
            await self.update_history(text, generated_text)

            self.generation_count += 1
            logger.info(f"Gemma generation #{self.generation_count} successful ({len(generated_text)} chars)")

            # Explicitly clear cache (optional, use if memory issues persist)
            # torch.cuda.empty_cache()

            return generated_text

        except Exception as e:
            logger.error(f"Gemma generation error: {e}", exc_info=True)
            # Optionally clear cache on error too
            # torch.cuda.empty_cache()
            return "Sorry, I encountered an error while generating a response."


class KokoroTTSProcessor:
    """Handles text-to-speech conversion using Kokoro model"""
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if hasattr(self, 'pipeline'): # Avoid re-initialization
             return
        logger.info("Initializing Kokoro TTS processor...")
        try:
            # Initialize Kokoro TTS pipeline (assuming 'a' means multi-language or English focus)
            self.pipeline = KPipeline(lang_code='a')
            self.default_voice = 'en-US-Neural2-J' # Example English voice, check Kokoro docs for available voices
            logger.info(f"Kokoro TTS processor initialized successfully with voice: {self.default_voice}")
            self.synthesis_count = 0
            self.is_ready = True
        except ImportError:
             logger.error("Kokoro library not found. Please install it.")
             self.pipeline = None
             self.is_ready = False
        except Exception as e:
            logger.error(f"Error initializing Kokoro TTS: {e}", exc_info=True)
            self.pipeline = None
            self.is_ready = False # Mark as not ready

    async def synthesize_speech(self, text):
        """Convert text to speech using Kokoro TTS"""
        if not self.is_ready or not text:
            logger.warning(f"TTS skipped. Ready: {self.is_ready}, Text provided: {bool(text)}")
            return None

        try:
            start_time = time.time()
            logger.info(f"Synthesizing speech for text (first 50 chars): '{text[:50]}...'")

            # Run TTS in a thread pool executor
            # Use a robust split pattern for natural pauses
            split_pattern = r'[.!?]+' # Split on sentence endings

            # Kokoro's pipeline might return a generator or directly the result
            # Assuming it's blocking, run in executor
            audio_data = await asyncio.get_event_loop().run_in_executor(
                None, # Default executor
                lambda: self.pipeline.synthesize( # Use synthesize method directly if available
                    text,
                    voice=self.default_voice,
                    speed=1.0, # Adjust speed if needed
                    # If synthesize doesn't handle splitting, process text first:
                    # text_parts = re.split(split_pattern, text)
                    # combined_audio = np.concatenate([self.pipeline(...) for part in text_parts if part.strip()])
                    # For now, assuming synthesize handles the full text reasonably
                )
            )

            # Check the type of audio_data returned by Kokoro
            if isinstance(audio_data, np.ndarray):
                 combined_audio = audio_data
            elif isinstance(audio_data, list) and all(isinstance(seg, np.ndarray) for seg in audio_data):
                 combined_audio = np.concatenate(audio_data) if audio_data else None
            # Add other expected return types if necessary
            else:
                 logger.warning(f"Unexpected audio data type from Kokoro: {type(audio_data)}")
                 combined_audio = None


            if combined_audio is not None and combined_audio.size > 0:
                self.synthesis_count += 1
                duration = len(combined_audio) / self.pipeline.target_sample_rate if hasattr(self.pipeline, 'target_sample_rate') else 'N/A'
                elapsed = time.time() - start_time
                logger.info(f"Speech synthesis #{self.synthesis_count} complete. Duration: {duration}s, Time: {elapsed:.2f}s")
                # Ensure output is float32 between -1 and 1 if needed by downstream
                if combined_audio.dtype != np.float32:
                     combined_audio = combined_audio.astype(np.float32)
                     # Normalize if necessary (assuming Kokoro might output int16)
                     if np.issubdtype(combined_audio.dtype, np.integer):
                          max_val = np.iinfo(combined_audio.dtype).max
                          combined_audio = combined_audio / max_val

                # Explicitly clear cache (optional)
                # torch.cuda.empty_cache()
                return combined_audio
            else:
                 logger.warning("Speech synthesis resulted in empty audio.")
                 return None

        except Exception as e:
            logger.error(f"Speech synthesis error: {e}", exc_info=True)
            # Optionally clear cache on error too
            # torch.cuda.empty_cache()
            return None


# --- WebSocket Handler ---

async def handle_client(websocket):
    """Handles WebSocket client connection"""
    client_ip = websocket.remote_address
    logger.info(f"Client connected from {client_ip}")

    # Initialize necessary components for this client session
    detector = AudioSegmentDetector()
    transcriber = WhisperTranscriber.get_instance() # Singleton
    gemma_processor = GemmaMultimodalProcessor.get_instance() # Singleton
    tts_processor = KokoroTTSProcessor.get_instance() # Singleton

    if not tts_processor.is_ready:
         logger.error("TTS Processor is not ready. Cannot provide audio responses.")
         # Optionally send an error message to the client
         try:
             await websocket.send(json.dumps({"error": "TTS service unavailable."}))
         except websockets.exceptions.ConnectionClosed:
             pass # Client already disconnected
         return # End handler if TTS is critical and unavailable


    # --- Background Tasks for this client ---
    receive_task = None
    segment_task = None
    keepalive_task = None

    try:
        # Task to handle incoming messages (audio, images, control)
        async def receive_messages():
            nonlocal detector, gemma_processor # Allow modification
            async for message in websocket:
                try:
                    data = json.loads(message)

                    is_responding = detector.assistant_is_responding # Check current state

                    # Handle image data (only process if assistant is NOT responding)
                    if "image" in data and not is_responding:
                        try:
                            image_data = base64.b64decode(data["image"])
                            logger.info("Received standalone image data.")
                            await gemma_processor.set_image(image_data)
                        except (base64.binascii.Error, ValueError) as decode_err:
                            logger.error(f"Error decoding base64 image: {decode_err}")
                        except Exception as img_err:
                            logger.error(f"Error processing received image: {img_err}")

                    # Handle audio data (always add to buffer)
                    elif "audio_data" in data: # Assuming 'audio_data' key for raw bytes
                        try:
                            audio_bytes = base64.b64decode(data["audio_data"])
                            # logger.debug(f"Received audio chunk: {len(audio_bytes)} bytes")
                            await detector.add_audio(audio_bytes)
                        except (base64.binascii.Error, ValueError) as decode_err:
                             logger.error(f"Error decoding base64 audio: {decode_err}")
                        except Exception as audio_err:
                             logger.error(f"Error processing received audio: {audio_err}")

                    # Handle explicit interrupt signal from client (if implemented)
                    elif "interrupt" in data and data["interrupt"] is True:
                         if is_responding:
                              logger.info("Received explicit interrupt request from client.")
                              await detector.cancel_current_processing()

                    # Handle other potential message types if needed

                except json.JSONDecodeError:
                    logger.error(f"Received invalid JSON message: {message[:100]}...") # Log first 100 chars
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Connection closed during message receive.")
                    break # Exit loop cleanly
                except Exception as e:
                    logger.error(f"Error processing received message: {e}", exc_info=True)

            logger.info("Receive message loop finished.")


        # Task to process detected speech segments
        async def process_speech_segments():
            nonlocal detector, transcriber, gemma_processor, tts_processor # Allow modification
            while True:
                try:
                    speech_segment = await detector.get_next_segment()

                    if speech_segment:
                        # --- Check if assistant is already responding ---
                        # This check prevents starting a new process if one is already underway
                        # The VAD's internal interrupt handles stopping the *ongoing* process
                        async with detector.state_lock:
                             if detector.assistant_is_responding:
                                  logger.info("New speech segment detected, but assistant is already responding. Discarding segment.")
                                  continue # Skip this segment, let the VAD handle interruption

                        # --- Start Processing Pipeline ---
                        processing_task = asyncio.create_task(
                            run_full_response_pipeline(
                                speech_segment, detector, transcriber, gemma_processor, tts_processor, websocket
                            )
                        )
                        await detector.set_current_task(processing_task)

                    # Short sleep to prevent tight loop when queue is empty
                    await asyncio.sleep(0.01)

                except websockets.exceptions.ConnectionClosed:
                    logger.info("Connection closed during segment processing.")
                    break # Exit loop
                except asyncio.CancelledError:
                     logger.info("Segment processing task cancelled.")
                     break # Exit loop
                except Exception as e:
                    logger.error(f"Error processing speech segment: {e}", exc_info=True)
                    # Reset state in case of unexpected error within the loop
                    await detector.set_assistant_responding(False)
                    # Short pause after error
                    await asyncio.sleep(1)

            logger.info("Process speech segments loop finished.")


        # Task to send keepalive pings
        async def send_keepalive():
            while True:
                try:
                    await websocket.ping()
                    await asyncio.sleep(15) # Send ping every 15 seconds
                except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
                    logger.info("Keepalive task stopped.")
                    break
                except Exception as e:
                     logger.error(f"Error in keepalive task: {e}")
                     # Wait before retrying after an error
                     await asyncio.sleep(5)


        # Run tasks concurrently
        receive_task = asyncio.create_task(receive_messages())
        segment_task = asyncio.create_task(process_speech_segments())
        keepalive_task = asyncio.create_task(send_keepalive())

        # Wait for tasks to complete (normally receive_task ends first on disconnect)
        done, pending = await asyncio.wait(
            [receive_task, segment_task, keepalive_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
             logger.info(f"Cancelling pending task: {task.get_name()}")
             task.cancel()
             try:
                  await task # Allow cancellation to complete
             except asyncio.CancelledError:
                  pass # Expected
             except Exception as e:
                  logger.error(f"Error during pending task cleanup: {e}")


    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Connection closed normally: {e.code} {e.reason}")
    except Exception as e:
        logger.error(f"Unhandled exception in client handler for {client_ip}: {e}", exc_info=True)
    finally:
        logger.info(f"Cleaning up client session for {client_ip}")
        # Ensure any lingering processing task is cancelled
        await detector.cancel_current_processing()
        # Explicitly cancel tasks if they haven't finished (redundant with above but safe)
        for task in [receive_task, segment_task, keepalive_task]:
             if task and not task.done():
                  task.cancel()
        logger.info(f"Client {client_ip} disconnected.")


async def run_full_response_pipeline(speech_segment, detector, transcriber, gemma_processor, tts_processor, websocket):
     """Handles the ASR -> LLM -> TTS pipeline for a single speech segment."""
     try:
          # 1. Set state: Assistant is now responding
          await detector.set_assistant_responding(True)

          # 2. Transcribe Speech
          transcription = await transcriber.transcribe(speech_segment)
          if not transcription or len(transcription) < 3 : # Basic filter for very short/empty
               logger.info(f"Skipping empty or very short transcription: '{transcription}'")
               await detector.set_assistant_responding(False) # Reset state
               return # End pipeline early

          # --- More Robust Filtering ---
          # Remove punctuation for word count check
          cleaned_transcription = re.sub(r'[^\w\s]', '', transcription).lower()
          words = [w for w in cleaned_transcription.split() if w] # Get non-empty words
          # Filter common short fillers / single words
          common_fillers = {'yes', 'no', 'ok', 'okay', 'um', 'uh', 'yeah', 'hmm', 'bye'}
          if not words or len(words) <= 1 or (len(words) == 1 and words[0] in common_fillers) :
               logger.info(f"Skipping transcription due to filtering (fillers/short): '{transcription}'")
               await detector.set_assistant_responding(False) # Reset state
               return # End pipeline early


          # Send interrupt signal to client *before* sending audio
          # This tells client to stop playing any previous audio immediately
          logger.info("Sending interrupt signal to client.")
          try:
               await websocket.send(json.dumps({"interrupt": True}))
          except websockets.exceptions.ConnectionClosed:
               logger.warning("Cannot send interrupt, connection closed.")
               await detector.set_assistant_responding(False) # Reset state
               return # Cannot continue if connection is closed


          # 3. Generate Response using Gemma
          logger.info("Generating response with Gemma...")
          generated_text = await gemma_processor.generate(transcription)
          if not generated_text or "error" in generated_text.lower():
               logger.error(f"Gemma generation failed or returned error: '{generated_text}'")
               # Optionally send a generic fallback message
               # fallback_audio = await tts_processor.synthesize_speech("Sorry, I couldn't process that.")
               # if fallback_audio... send it
               await detector.set_assistant_responding(False) # Reset state
               return

          # 4. Synthesize Speech using Kokoro TTS
          logger.info("Synthesizing speech with Kokoro TTS...")
          audio_response = await tts_processor.synthesize_speech(generated_text)
          if audio_response is not None:
               # 5. Send Audio Response
               try:
                    # Convert float32 numpy array to int16 bytes
                    audio_int16 = (audio_response * 32767).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()
                    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

                    logger.info(f"Sending synthesized audio ({len(audio_bytes)} bytes) to client.")
                    await websocket.send(json.dumps({"audio": base64_audio}))

               except websockets.exceptions.ConnectionClosed:
                    logger.warning("Connection closed before audio could be sent.")
               except Exception as send_err:
                    logger.error(f"Error sending audio to client: {send_err}")
          else:
               logger.warning("TTS synthesis failed, no audio response sent.")


     except asyncio.CancelledError:
          logger.info("Response pipeline task was cancelled.")
          # State should be handled by the cancel_current_processing method
          raise # Re-raise cancellation

     except Exception as e:
          logger.error(f"Error in response pipeline: {e}", exc_info=True)
          # Ensure state is reset on unexpected error
          await detector.set_assistant_responding(False)

     finally:
          # 6. Reset State: Assistant finished responding (or was cancelled/errored out)
          # Check if the task was cancelled - if so, cancel_current_processing already reset the state
          if not asyncio.current_task().cancelled():
             await detector.set_assistant_responding(False)


async def main():
    """Main function to start the WebSocket server"""
    logger.info("Initializing models...")
    try:
        # Pre-initialize singleton instances to load models on startup
        WhisperTranscriber.get_instance()
        GemmaMultimodalProcessor.get_instance()
        KokoroTTSProcessor.get_instance()
        logger.info("Models initialized successfully.")
    except Exception as init_err:
         logger.error(f"FATAL: Model initialization failed: {init_err}", exc_info=True)
         sys.exit(1) # Exit if models can't load


    logger.info("Starting WebSocket server on 0.0.0.0:9073")
    try:
        # Increased timeouts for potentially slower connections or processing
        async with websockets.serve(
            handle_client,
            "0.0.0.0",
            9073,
            ping_interval=20,    # Check connection every 20s
            ping_timeout=40,     # Allow 40s for pong response
            close_timeout=10     # Allow 10s for close handshake
        ):
            logger.info("WebSocket server running on ws://0.0.0.0:9073")
            await asyncio.Future()  # Run forever
    except OSError as e:
         logger.error(f"Server error: Could not bind to address/port. {e}")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)

if __name__ == "__main__":
    # Set numpy print options for debugging if needed
    # np.set_printoptions(threshold=10)
    asyncio.run(main())
