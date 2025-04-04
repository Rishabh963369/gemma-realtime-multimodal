import asyncio
import json
import websockets
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, Gemma3ForConditionalGeneration
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
from threading import Thread
from transformers import TextIteratorStreamer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Audio Segment Detector ---
class AudioSegmentDetector:
    """Detects speech segments based on audio energy levels and manages TTS/Generation interruption."""

    def __init__(self,
                 sample_rate=16000,
                 energy_threshold=0.015,
                 silence_duration=0.8,
                 min_speech_duration=0.8,
                 max_speech_duration=15):

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
        self.lock = asyncio.Lock()  # Protects audio_buffer and VAD state
        self.segment_queue = asyncio.Queue()

        # Counters
        self.segments_detected = 0

        # TTS playback and generation control
        self._tts_playing = False # Internal state variable
        self.tts_lock = asyncio.Lock() # Protects _tts_playing
        self.current_generation_task = None
        self.current_tts_task = None
        self.current_remaining_text_task = None # Track the text collection task specifically
        self.task_lock = asyncio.Lock() # Protects task handles

    @property
    async def tts_playing(self):
        """Check TTS playback state safely."""
        async with self.tts_lock:
            return self._tts_playing

    async def set_tts_playing(self, is_playing: bool):
        """Set TTS playback state safely."""
        async with self.tts_lock:
            if self._tts_playing != is_playing:
                logger.info(f"Setting TTS playing state to: {is_playing}")
                self._tts_playing = is_playing

    async def cancel_current_tasks(self):
        """Cancel any ongoing generation and TTS tasks."""
        async with self.task_lock:
            cancelled_something = False
            if self.current_generation_task and not self.current_generation_task.done():
                logger.info("Cancelling current generation task.")
                self.current_generation_task.cancel()
                try:
                    await self.current_generation_task
                except asyncio.CancelledError:
                    logger.info("Generation task cancelled successfully.")
                except Exception as e:
                    logger.error(f"Error awaiting cancelled generation task: {e}")
                self.current_generation_task = None
                cancelled_something = True

            # Also cancel remaining text collection if it exists
            if self.current_remaining_text_task and not self.current_remaining_text_task.done():
                logger.info("Cancelling current remaining text collection task.")
                self.current_remaining_text_task.cancel()
                try:
                    await self.current_remaining_text_task
                except asyncio.CancelledError:
                    logger.info("Remaining text collection task cancelled successfully.")
                except Exception as e:
                    logger.error(f"Error awaiting cancelled remaining text task: {e}")
                self.current_remaining_text_task = None
                # No need to set cancelled_something = True here, handled by generation task

            if self.current_tts_task and not self.current_tts_task.done():
                logger.info("Cancelling current TTS task.")
                self.current_tts_task.cancel()
                try:
                    await self.current_tts_task
                except asyncio.CancelledError:
                    logger.info("TTS task cancelled successfully.")
                except Exception as e:
                    logger.error(f"Error awaiting cancelled TTS task: {e}")
                self.current_tts_task = None
                cancelled_something = True

            # Clear TTS playing state *only if* we cancelled something that implies playback
            # This needs to happen *after* awaiting the cancelled tasks.
            if cancelled_something:
                 await self.set_tts_playing(False) # Use the safe setter method


    async def set_current_tasks(self, generation_task=None, tts_task=None, remaining_text_task=None):
        """Set current generation and TTS tasks safely."""
        async with self.task_lock:
            # Important: Don't overwrite existing tasks if None is passed, unless explicitly clearing
            if generation_task is not None or (generation_task is None and remaining_text_task is None and tts_task is None): # Clear gen task if all are None or explicitly set
                 self.current_generation_task = generation_task
            if remaining_text_task is not None or (generation_task is None and remaining_text_task is None and tts_task is None): # Clear text task if all are None or explicitly set
                self.current_remaining_text_task = remaining_text_task
            if tts_task is not None or (generation_task is None and remaining_text_task is None and tts_task is None): # Clear tts task if all are None or explicitly set
                self.current_tts_task = tts_task
            logger.debug(f"Set current tasks: Gen={self.current_generation_task is not None}, TTS={self.current_tts_task is not None}, Text={self.current_remaining_text_task is not None}")


    async def add_audio(self, audio_bytes):
        """Add audio data to the buffer and check for speech segments"""
        async with self.lock: # Protects audio_buffer and VAD state
            self.audio_buffer.extend(audio_bytes)

            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_array) > 0:
                energy = np.sqrt(np.mean(audio_array**2))

                segment_to_process = None
                reset_speech_state_after_processing = False
                trim_index = -1

                if not self.is_speech_active and energy > self.energy_threshold:
                    self.is_speech_active = True
                    self.speech_start_idx = max(0, len(self.audio_buffer) - len(audio_bytes))
                    self.silence_counter = 0
                    logger.info(f"Speech start detected (energy: {energy:.6f})")

                elif self.is_speech_active:
                    if energy > self.energy_threshold:
                        self.silence_counter = 0
                    else:
                        # Potential end of speech
                        self.silence_counter += len(audio_array)

                        # Check if enough silence to end speech segment
                        if self.silence_counter >= self.silence_samples:
                            speech_end_idx = len(self.audio_buffer) - self.silence_counter
                            potential_segment = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])

                            reset_speech_state_after_processing = True
                            trim_index = speech_end_idx # Mark where to trim if segment is valid

                            if len(potential_segment) >= self.min_speech_samples * 2:  # Check length
                                segment_to_process = potential_segment
                                logger.info(f"Speech segment detected (silence end): {len(segment_to_process)/2/self.sample_rate:.2f}s")
                            else:
                                # Segment too short, ignore but reset state and trim buffer now
                                logger.info(f"Short speech segment ignored ({len(potential_segment)/2/self.sample_rate:.2f}s)")
                                if trim_index != -1:
                                    self.audio_buffer = self.audio_buffer[trim_index:]
                                self.is_speech_active = False
                                self.silence_counter = 0
                                # Don't return anything, just reset and continue

                        # Check if speech segment exceeds maximum duration (regardless of silence)
                        # Use >= to ensure exactly max duration triggers it too
                        elif (len(self.audio_buffer) - self.speech_start_idx) >= self.max_speech_samples * 2:
                            segment_end_idx = self.speech_start_idx + self.max_speech_samples * 2
                            segment_to_process = bytes(self.audio_buffer[self.speech_start_idx : segment_end_idx])

                            # Update start index for the *next* potential segment immediately after this one
                            self.speech_start_idx = segment_end_idx
                            # Don't reset silence counter or is_speech_active here, as speech might be continuing
                            # Don't trim the buffer here, as the next segment starts immediately

                            logger.info(f"Max duration speech segment detected: {len(segment_to_process)/2/self.sample_rate:.2f}s")
                            reset_speech_state_after_processing = False # Explicitly false for max duration cut


                # --- Cancellation Logic & Queueing (if a segment was identified) ---
                if segment_to_process:
                    self.segments_detected += 1

                    # 1. Check if cancellation is needed
                    needs_cancel = False
                    async with self.tts_lock: # Briefly lock to read the state
                        if self._tts_playing:
                            needs_cancel = True

                    # 2. Perform cancellation *after* releasing tts_lock
                    if needs_cancel:
                        logger.info("Detected new speech segment while TTS might be active. Cancelling ongoing tasks.")
                        await self.cancel_current_tasks() # This internally sets _tts_playing = False if needed

                    # 3. Add the valid segment to the queue for processing
                    await self.segment_queue.put(segment_to_process)

                    # 4. Trim buffer and reset state if this segment ended due to silence
                    if reset_speech_state_after_processing:
                        # Ensure trim_index is valid before trimming
                        if trim_index != -1:
                             self.audio_buffer = self.audio_buffer[trim_index:]
                             logger.debug(f"Buffer trimmed at index {trim_index}, new size {len(self.audio_buffer)}")
                        else:
                             logger.warning("Reset state requested but trim_index was invalid.")
                        # Reset VAD state
                        self.is_speech_active = False
                        self.silence_counter = 0
                        logger.debug("VAD state reset (is_speech_active=False)")

                    return segment_to_process # Indicate a segment was processed and queued

            return None # No segment processed in this call

    async def get_next_segment(self):
        """Get the next available speech segment from the queue."""
        try:
            # Use a small timeout to prevent blocking indefinitely if the queue is empty
            return await asyncio.wait_for(self.segment_queue.get(), timeout=0.05) # Slightly longer timeout
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error getting segment from queue: {e}")
            return None

# --- Whisper Transcriber ---
class WhisperTranscriber:
    """Handles speech transcription using Whisper large-v3 model with pipeline"""
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Creating WhisperTranscriber instance")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if WhisperTranscriber._instance is not None:
             raise Exception("This class is a singleton!")
        else:
             WhisperTranscriber._instance = self

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Whisper using device: {self.device}")
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32

        model_id = "openai/whisper-large-v3" # Using large-v3 instead of turbo, adjust if needed
        logger.info(f"Loading Whisper model: {model_id}...")

        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None # Use Flash Attention if available
            ).to(self.device)

            self.processor = AutoProcessor.from_pretrained(model_id)

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
                chunk_length_s=30, # Process in chunks
                stride_length_s=5  # Overlap chunks
            )
            logger.info("Whisper model ready for transcription")
        except Exception as e:
            logger.error(f"FATAL: Failed to load Whisper model: {e}", exc_info=True)
            raise # Re-raise to stop server startup

        self.transcription_count = 0

    async def transcribe(self, audio_bytes, sample_rate=16000):
        """Transcribe audio bytes to text using the pipeline"""
        if not audio_bytes:
            return ""
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_array) < sample_rate * 0.1: # Ignore very short segments (e.g., < 100ms)
                logger.debug("Skipping transcription for very short audio segment.")
                return ""

            # Run blocking inference in executor
            result = await asyncio.get_event_loop().run_in_executor(
                None,  # Use default executor (thread pool)
                lambda: self.pipe(
                    {"array": audio_array, "sampling_rate": sample_rate},
                    generate_kwargs={
                        "task": "transcribe",
                        "language": "english", # Force English
                        "temperature": 0.0 # Force greedy decoding
                    },
                    # return_timestamps=False, # Keep disabled unless debugging
                )
            )

            text = result.get("text", "").strip()
            if text:
                self.transcription_count += 1
                logger.info(f"Transcription #{self.transcription_count}: '{text}'")
            else:
                 logger.info("Transcription result was empty.")

            return text

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return ""

# --- Gemma Multimodal Processor ---
class GemmaMultimodalProcessor:
    """Handles multimodal generation using Gemma model"""
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Creating GemmaMultimodalProcessor instance")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if GemmaMultimodalProcessor._instance is not None:
             raise Exception("This class is a singleton!")
        else:
             GemmaMultimodalProcessor._instance = self

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Gemma using device: {self.device}")

        model_id = "google/gemma-3-4b-it"
        logger.info(f"Loading Gemma model: {model_id}...")

        try:
            # Using bfloat16 as recommended for Gemma 3 if available, else float16
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto", # Let accelerate handle device placement
                torch_dtype=dtype, # Use bfloat16 or float16
                # load_in_8bit=True, # Keep commented unless VRAM is tight
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None # Use Flash Attention if available
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            logger.info("Gemma model ready for multimodal generation")

        except Exception as e:
            logger.error(f"FATAL: Failed to load Gemma model: {e}", exc_info=True)
            raise # Re-raise to stop server startup

        self.last_image = None
        self.last_image_timestamp = 0
        self.lock = asyncio.Lock() # Protects image and history

        self.message_history = []
        self.max_history_turns = 3 # Keep last N turns (User + Assistant = 1 turn)
        self.system_prompt = """You are a helpful assistant providing spoken responses about images and engaging in natural conversation. Keep your responses concise, fluent, and conversational. Use natural oral language that's easy to listen to.

When responding:
1. If the user's question or comment is clearly about the image, provide a relevant, focused response about what you see.
2. If the user's input is not clearly related to the image or lacks context:
   - Don't force image descriptions into your response.
   - Respond naturally as in a normal conversation.
   - If needed, politely ask for clarification (e.g., "Could you please be more specific about what you'd like to know about the image?")
3. Keep responses concise:
   - Aim for 2-4 short sentences maximum.
   - Focus on the most relevant information.
   - Use conversational language, avoid jargon unless asked.

Maintain conversation context and refer to previous exchanges naturally when relevant. If the user's request is unclear, ask them to repeat or clarify in a friendly way."""

        self.generation_count = 0

    async def set_image(self, image_data):
        """Cache the most recent image received and clear history."""
        async with self.lock:
            try:
                image = Image.open(io.BytesIO(image_data)).convert("RGB") # Ensure RGB

                # Optional: Resize if images are very large
                max_size = 1024
                if max(image.size) > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    logger.info(f"Resized image to fit within {max_size}x{max_size}")

                # Clear message history when a new image context is set
                if self.last_image is not None:
                     logger.info("New image received, clearing conversation history.")
                     self.message_history = []
                else:
                     logger.info("First image received.")

                self.last_image = image
                self.last_image_timestamp = time.time()
                return True
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                self.last_image = None # Ensure invalid image doesn't persist
                return False

    # Make history update async safe as it modifies shared state under lock
    async def _update_history(self, user_text, assistant_response):
        """Update message history with new exchange, trimming old turns."""
        async with self.lock: # Protect history updates
            # Add user message (text only for history)
            self.message_history.append({"role": "user", "content": user_text}) # Store text only

            # Add assistant response
            self.message_history.append({"role": "assistant", "content": assistant_response})

            # Trim history: Keep last `max_history_turns` exchanges (user + assistant pairs)
            max_messages = self.max_history_turns * 2
            if len(self.message_history) > max_messages:
                self.message_history = self.message_history[-max_messages:]
                logger.debug(f"Trimmed message history to {len(self.message_history)} messages.")

    async def _build_messages(self, text):
        """Build messages array with history for the model"""
        # Needs lock as it reads history and last_image
        async with self.lock:
            messages = [{"role": "system", "content": self.system_prompt}]

            # Add conversation history
            messages.extend(self.message_history)

            # Add current user message with image (if available)
            content = []
            current_image = self.last_image # Get image under lock
            if current_image:
                content.append({"type": "image"}) # Placeholder, actual image added later
            content.append({"type": "text", "text": text})

            messages.append({"role": "user", "content": content})

            return messages, current_image # Return image used for this context

    async def generate_streaming(self, text):
        """Generate a response using the latest image and text input with streaming."""
        # Build messages and get the image associated with this request
        messages, current_image = await self._build_messages(text)

        if not current_image:
            logger.warning("No image available for multimodal generation, responding text-only.")
            # Fallback: Simple text response
            return None, f"Sorry, I don't have an image to look at right now. Can you describe what you want to talk about?"

        try:
            # Prepare inputs using the image obtained earlier
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                images=current_image # Use the specific image for this request
            ).to(self.model.device)

            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = dict(
                inputs, # Pass processed inputs directly
                streamer=streamer,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.pad_token_id # Explicitly set pad token if needed
            )

            # Run model.generate in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True)
            thread.start()

            self.generation_count += 1
            logger.info(f"Gemma generation #{self.generation_count} started for: '{text}'")

            return streamer, text # Return streamer and original user text

        except Exception as e:
            logger.error(f"Gemma streaming generation error: {e}", exc_info=True)
            error_response = "Sorry, I encountered an error trying to process that."
            return None, error_response


# --- Kokoro TTS Processor ---
class KokoroTTSProcessor:
    """Handles text-to-speech conversion using Kokoro model"""
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Creating KokoroTTSProcessor instance")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if KokoroTTSProcessor._instance is not None:
             raise Exception("This class is a singleton!")
        else:
             KokoroTTSProcessor._instance = self

        logger.info("Initializing Kokoro TTS processor...")
        # Set default lang code here before try block
        self.lang_code = 'a' # <<< CORRECT CODE IS 'a' HERE

        try:
            logger.info(f"Attempting to initialize Kokoro with lang_code: '{self.lang_code}'") # Should log 'a'
            self.pipeline = KPipeline(lang_code=self.lang_code) # <<< USES 'a' HERE

            # ... rest of the init ...

        except AssertionError as e:
             # Log the specific lang_code that failed
             logger.error(f"FATAL: Kokoro language code assertion failed for lang_code='{self.lang_code}': {e}", exc_info=True) # Error log should show 'a' if this code runs
             logger.error(f"Check if lang_code '{self.lang_code}' is valid in your Kokoro installation's LANG_CODES.")
             self.pipeline = None
             raise # Stop server startup



    async def _synthesize_internal(self, text, split_pattern=None):
        """Internal synthesis logic shared by initial and remaining synthesis"""
        if not text or not self.pipeline:
            logger.warning("Skipping synthesis - empty text or no TTS pipeline.")
            return None

        try:
            audio_segments = []
            # Use list() to consume the generator within the executor function
            full_command = lambda: list(self.pipeline(
                    text,
                    voice=self.default_voice,
                    speed=1.0, # Adjust speed if desired
                    split_pattern=split_pattern
                ))

            generator_output = await asyncio.get_event_loop().run_in_executor(
                None, full_command
            )

            for _gs, _ps, audio_data in generator_output:
                if audio_data is not None and len(audio_data) > 0:
                    # Ensure audio data is numpy array float32
                    if not isinstance(audio_data, np.ndarray):
                        logger.warning(f"Kokoro returned non-numpy audio data type: {type(audio_data)}. Attempting conversion.")
                        try:
                            # Attempt conversion assuming it's list-like or similar
                            audio_data = np.array(audio_data, dtype=np.float32)
                        except Exception as conv_e:
                            logger.error(f"Failed to convert Kokoro audio data to numpy array: {conv_e}")
                            continue # Skip this segment
                    # Ensure float32 if it's already numpy
                    if audio_data.dtype != np.float32:
                         audio_data = audio_data.astype(np.float32)

                    audio_segments.append(audio_data)
                else:
                     logger.warning("Kokoro generated an empty or None audio segment.")

            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                logger.info(f"Synthesis complete: generated {len(combined_audio)} samples ({len(combined_audio)/self.sample_rate:.2f}s)")
                return combined_audio
            else:
                logger.warning(f"Synthesis resulted in no valid audio data for text: '{text[:50]}...'")
                return None

        except Exception as e:
            logger.error(f"Speech synthesis error for text '{text[:50]}...': {e}", exc_info=True)
            return None


    async def synthesize_initial_speech(self, text):
        """Convert initial text chunk to speech rapidly (minimal splitting)."""
        if not text: return None
        logger.info(f"Synthesizing initial speech for: '{text}'")
        split_pattern = r'[.!?]+'
        audio_data = await self._synthesize_internal(text, split_pattern=split_pattern)
        if audio_data is not None:
             self.synthesis_count += 1
             logger.info(f"Initial speech synthesis count: {self.synthesis_count}")
        return audio_data


    async def synthesize_remaining_speech(self, text):
        """Convert remaining text to speech with more natural splitting."""
        if not text: return None
        logger.info(f"Synthesizing remaining speech for: '{text[:50]}...'")
        split_pattern = r'[.!?,\-;:]+'
        audio_data = await self._synthesize_internal(text, split_pattern=split_pattern)
        # Only increment total count if needed, maybe initial is enough
        # if audio_data is not None: self.synthesis_count += 1
        return audio_data

# --- WebSocket Handler ---
async def handle_client(websocket):
    """Handles WebSocket client connection, orchestrating the components."""
    client_id = websocket.remote_address
    logger.info(f"Client connected: {client_id}")

    # Initialize per-connection state
    detector = AudioSegmentDetector() # Each client gets its own detector/state machine
    # Get singleton instances for models
    try:
        transcriber = WhisperTranscriber.get_instance()
        gemma_processor = GemmaMultimodalProcessor.get_instance()
        tts_processor = KokoroTTSProcessor.get_instance()
    except Exception as model_init_error:
         logger.error(f"Failed to get model instances for client {client_id}: {model_init_error}")
         await websocket.close(code=1011, reason="Server Model Error")
         return


    if not tts_processor.pipeline: # Critical check
        logger.error("TTS Processor not available. Closing connection.")
        await websocket.close(code=1011, reason="Server TTS Error")
        return

    speech_task = None
    receive_task = None
    keepalive_task = None

    try:
        # --- Task Definitions ---
        async def send_keepalive():
            """Sends pings to keep connection alive."""
            while True:
                try:
                    await websocket.ping()
                    await asyncio.sleep(15) # Send ping every 15 seconds
                except asyncio.CancelledError:
                     logger.info("Keepalive task cancelled.")
                     break
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Keepalive: Connection closed.")
                    break
                except Exception as e:
                    logger.error(f"Keepalive error: {e}")
                    break

        async def collect_remaining_text_and_synthesize(streamer, initial_text, original_user_text):
            """Collects remaining text from streamer, updates history, and synthesizes remaining audio."""
            remaining_text_list = []
            remaining_audio = None
            complete_response = initial_text # Start with the initial part
            current_remaining_tts_task = None # Track task specific to this function call

            if streamer:
                try:
                    async for chunk in streamer: # Async iteration over TextIteratorStreamer
                        if chunk:
                            remaining_text_list.append(chunk)

                except asyncio.CancelledError:
                    logger.info("Remaining text collection was cancelled.")
                except Exception as e:
                    logger.error(f"Error collecting remaining text: {e}", exc_info=True)
                # --- FINALLY block executes even on cancellation ---
                finally:
                    remaining_text = "".join(remaining_text_list)
                    # Ensure complete_response is updated even if errors occurred during streaming
                    if remaining_text:
                        complete_response += remaining_text

                    logger.info(f"Collected remaining text ({len(remaining_text)} chars). Full response len: {len(complete_response)} chars.")

                    # Update history with the *complete* response *now*
                    if complete_response and original_user_text:
                         try:
                             # Use the async history update method
                             await gemma_processor._update_history(original_user_text, complete_response)
                         except Exception as hist_e:
                              logger.error(f"Failed to update history: {hist_e}")

                    # Synthesize remaining text if any was collected
                    if remaining_text:
                        # Create and store the handle for the remaining TTS task
                        current_remaining_tts_task = asyncio.create_task(
                             tts_processor.synthesize_remaining_speech(remaining_text)
                        )
                        await detector.set_current_tasks(tts_task=current_remaining_tts_task)

                        try:
                            remaining_audio = await current_remaining_tts_task
                            if remaining_audio is not None:
                                logger.info("Remaining audio synthesis successful.")
                            else:
                                logger.warning("Remaining audio synthesis returned None.")

                        except asyncio.CancelledError:
                            logger.info("Remaining TTS synthesis was cancelled.")
                            remaining_audio = None # Ensure no audio is sent
                        except Exception as e:
                             logger.error(f"Error during remaining TTS synthesis: {e}")
                             remaining_audio = None
                        # No finally needed here for task clearing, handled later

            # Send remaining audio if synthesized successfully
            if remaining_audio is not None:
                try:
                    # Ensure audio is float32 before converting
                    if remaining_audio.dtype != np.float32:
                        remaining_audio = remaining_audio.astype(np.float32)
                    # Scale and convert to int16
                    scaled_audio = (remaining_audio * 32767).clip(-32768, 32767).astype(np.int16)
                    audio_bytes = scaled_audio.tobytes()
                    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

                    await websocket.send(json.dumps({
                        "audio": base64_audio,
                        "sample_rate": tts_processor.sample_rate # Send sample rate
                    }))
                    logger.info(f"Sent remaining audio chunk ({len(audio_bytes)} bytes)")
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Could not send remaining audio: Connection closed.")
                except Exception as e:
                    logger.error(f"Error sending remaining audio: {e}")

            # --- Final Cleanup for this specific flow ---
            logger.info("Finished processing initial and remaining TTS flow.")
            # Reset TTS playing state ONLY IF this task wasn't cancelled midway
            # Check if the task we created still exists and completed normally
            async with detector.task_lock:
                # Check if the TTS task we were responsible for is done
                if current_remaining_tts_task and current_remaining_tts_task.done() and not current_remaining_tts_task.cancelled():
                     # Also check if the overall remaining text task is done (which is this function's task)
                     remaining_text_task_handle = detector.current_remaining_text_task
                     if remaining_text_task_handle and remaining_text_task_handle.done() and not remaining_text_task_handle.cancelled():
                          await detector.set_tts_playing(False) # Safe to assume TTS finished
                     else:
                           logger.info("Remaining text task was cancelled or not found, not resetting tts_playing state here.")

                # Clear the specific TTS task handle associated with this function call
                if detector.current_tts_task == current_remaining_tts_task:
                     detector.current_tts_task = None # Clear it

            # Clear remaining text task handle from the detector *if* it points to this task instance
            # Get the current task
            this_task = asyncio.current_task()
            async with detector.task_lock:
                 if detector.current_remaining_text_task == this_task:
                      detector.current_remaining_text_task = None
                      logger.debug("Cleared remaining text task handle from detector.")


        async def detect_speech_and_process():
            """Handles VAD, ASR, LLM, and initial TTS."""
            while True:
                try:
                    speech_segment = await detector.get_next_segment() # Non-blocking check

                    if speech_segment:
                        transcription = await transcriber.transcribe(speech_segment)

                        # --- Filter Transcription ---
                        if not transcription:
                             logger.info("Skipping empty transcription.")
                             continue

                        cleaned_transcription = transcription.strip().lower()
                        if not any(c.isalnum() for c in cleaned_transcription):
                            logger.info(f"Skipping transcription with no alphanumeric chars: '{transcription}'")
                            continue

                        filler_patterns = [
                            r'^(um|uh|ah|oh|hm|mhm|hmm)$',
                            r'^(okay|ok|yes|no|yeah|nah|got it)$',
                            r'^(bye|goodbye)$',
                             r'^thanks?$', r'^thank\s?you$' # Add thanks
                        ]
                        words = [w for w in cleaned_transcription.split() if any(c.isalnum() for c in w)]
                        is_filler = len(words) <= 2 and any(re.fullmatch(pattern, cleaned_transcription) for pattern in filler_patterns)

                        if is_filler:
                            logger.info(f"Skipping likely filler/short transcription: '{transcription}'")
                            continue

                        # --- Start Processing Valid Transcription ---
                        logger.info(f"Processing transcription: '{transcription}'")

                        # 1. Send Interrupt Signal (client should stop playback)
                        try:
                            logger.info("Sending interrupt signal due to new valid speech.")
                            await websocket.send(json.dumps({"interrupt": True}))
                        except websockets.exceptions.ConnectionClosed:
                             logger.warning("Could not send interrupt: Connection closed.")
                             break # Exit loop if connection lost
                        except Exception as e:
                             logger.error(f"Error sending interrupt signal: {e}")

                        # 2. Mark TTS as potentially starting & Store Tasks
                        await detector.set_tts_playing(True)

                        # 3. Start LLM Generation (Streaming)
                        generation_stream_task = asyncio.create_task(
                             gemma_processor.generate_streaming(transcription)
                        )
                        await detector.set_current_tasks(generation_task=generation_stream_task)

                        try:
                            # Await the *start* of generation to get the streamer
                            streamer, user_text_for_history = await generation_stream_task
                            # Generation task itself is done, clear its handle
                            await detector.set_current_tasks(generation_task=None)

                            if streamer is None:
                                logger.warning(f"Generation failed or returned no streamer for '{transcription}'. Response: '{user_text_for_history}'")
                                if user_text_for_history != transcription: # Check if it's an error message
                                    error_audio_task = asyncio.create_task(tts_processor.synthesize_initial_speech(user_text_for_history))
                                    await detector.set_current_tasks(tts_task=error_audio_task)
                                    try:
                                        error_audio = await error_audio_task
                                        if error_audio is not None:
                                            if error_audio.dtype != np.float32: error_audio = error_audio.astype(np.float32)
                                            scaled_audio = (error_audio * 32767).clip(-32768, 32767).astype(np.int16)
                                            audio_bytes = scaled_audio.tobytes()
                                            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                                            await websocket.send(json.dumps({
                                                "audio": base64_audio,
                                                "sample_rate": tts_processor.sample_rate
                                            }))
                                            logger.info("Sent TTS for generation error message.")
                                        await asyncio.sleep(1) # Pause after error TTS
                                    except asyncio.CancelledError:
                                        logger.info("Error TTS task was cancelled.")
                                    except Exception as e:
                                        logger.error(f"Failed to send error TTS: {e}")
                                    finally:
                                         await detector.set_current_tasks(tts_task=None) # Clear TTS task handle
                                await detector.set_tts_playing(False) # Reset TTS playing state as generation failed
                                continue # Skip to next VAD segment

                            # 4. Process Initial Text from Streamer
                            initial_text = ""
                            min_chars_for_initial_tts = 35
                            sentence_end_pattern = re.compile(r'[.!?,\-;:]')
                            found_sentence_end = False

                            async for first_chunk in streamer:
                                initial_text += first_chunk
                                if sentence_end_pattern.search(first_chunk):
                                    found_sentence_end = True
                                if (found_sentence_end and len(initial_text) >= min_chars_for_initial_tts // 2) or \
                                   len(initial_text) >= min_chars_for_initial_tts:
                                    break
                                if len(initial_text) > min_chars_for_initial_tts * 3:
                                    logger.warning("Breaking initial text collection early due to length.")
                                    break

                            logger.info(f"Collected initial text ({len(initial_text)} chars): '{initial_text}'")

                            # 5. Synthesize and Send Initial Audio
                            if initial_text:
                                initial_tts_task = asyncio.create_task(
                                    tts_processor.synthesize_initial_speech(initial_text)
                                )
                                await detector.set_current_tasks(tts_task=initial_tts_task)

                                try:
                                    initial_audio = await initial_tts_task
                                    if initial_audio is not None:
                                        if initial_audio.dtype != np.float32: initial_audio = initial_audio.astype(np.float32)
                                        scaled_audio = (initial_audio * 32767).clip(-32768, 32767).astype(np.int16)
                                        audio_bytes = scaled_audio.tobytes()
                                        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                                        await websocket.send(json.dumps({
                                            "audio": base64_audio,
                                            "sample_rate": tts_processor.sample_rate
                                        }))
                                        logger.info(f"Sent initial audio chunk ({len(audio_bytes)} bytes)")

                                        # 6. Start Collecting Remaining Text & Audio IN BACKGROUND
                                        remaining_task = asyncio.create_task(
                                            collect_remaining_text_and_synthesize(streamer, initial_text, user_text_for_history)
                                        )
                                        # Store handle for this background task
                                        await detector.set_current_tasks(remaining_text_task=remaining_task, tts_task=None) # Clear initial TTS handle
                                        # Let this task run. The main loop continues.

                                    else:
                                        logger.warning("Initial TTS synthesis returned None. Cannot proceed with this turn.")
                                        await detector.set_tts_playing(False)
                                        await detector.set_current_tasks(tts_task=None) # Clear initial TTS task handle

                                except asyncio.CancelledError:
                                    logger.info("Initial TTS task was cancelled (likely by new speech).")
                                    # State (_tts_playing=False) handled by cancel_current_tasks
                                    # Ensure task handles are cleared if cancellation happened before setting remaining task
                                    await detector.set_current_tasks(tts_task=None, remaining_text_task=None)
                                except websockets.exceptions.ConnectionClosed:
                                    logger.warning("Connection closed while sending initial audio.")
                                    break # Exit loop
                                except Exception as e:
                                    logger.error(f"Error during initial TTS processing: {e}", exc_info=True)
                                    await detector.set_tts_playing(False)
                                    await detector.set_current_tasks(tts_task=None)
                            else:
                                logger.warning("No initial text was collected from the streamer.")
                                await detector.set_tts_playing(False)
                                await detector.set_current_tasks(tts_task=None) # Clear any lingering initial TTS task handle

                        except asyncio.CancelledError:
                             logger.info("LLM Generation task was cancelled before streamer obtained.")
                             # State should have been handled by cancel_current_tasks
                             await detector.set_current_tasks(generation_task=None)
                        except Exception as e:
                             logger.error(f"Error awaiting generation task/streamer: {e}", exc_info=True)
                             await detector.set_tts_playing(False)
                             await detector.set_current_tasks(generation_task=None)

                    else:
                        # No segment detected, sleep briefly
                        await asyncio.sleep(0.02)

                except asyncio.CancelledError:
                     logger.info("Detect speech task cancelled.")
                     break # Exit loop cleanly on cancellation
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Connection closed during speech processing loop.")
                    break
                except Exception as e:
                    logger.error(f"Error in detect_speech_and_process loop: {e}", exc_info=True)
                    try:
                        await detector.cancel_current_tasks() # Attempt to cancel anything running
                        # await detector.set_tts_playing(False) # cancel_current_tasks should do this
                        # await detector.set_current_tasks() # Clear all tasks explicitly? cancel might be enough
                    except Exception as recovery_e:
                        logger.error(f"Error during error recovery: {recovery_e}")
                    await asyncio.sleep(1) # Pause briefly after an error

        async def receive_messages():
            """Receives audio and image messages from the client."""
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    msg_timestamp = time.time()

                    # Check TTS status once per message
                    is_tts_playing_now = await detector.tts_playing

                    # Handle image data (only if TTS is not active)
                    if not is_tts_playing_now:
                        image_to_set = None
                        if "image" in data:
                            logger.debug("Received standalone image frame.")
                            image_to_set = base64.b64decode(data["image"])
                        elif "realtime_input" in data and "media_chunks" in data["realtime_input"]:
                             for chunk in data["realtime_input"]["media_chunks"]:
                                 if chunk.get("mime_type") == "image/jpeg":
                                     logger.debug("Received image chunk in realtime_input.")
                                     image_to_set = base64.b64decode(chunk["data"])
                                     break # Process only first image in chunk list
                        if image_to_set:
                             await gemma_processor.set_image(image_to_set)


                    # Handle audio data regardless of TTS state
                    if "realtime_input" in data and "media_chunks" in data["realtime_input"]:
                        for chunk in data["realtime_input"]["media_chunks"]:
                            if chunk.get("mime_type") == "audio/pcm":
                                audio_data = base64.b64decode(chunk["data"])
                                await detector.add_audio(audio_data) # Add audio to VAD buffer

                except asyncio.CancelledError:
                     logger.info("Receive messages task cancelled.")
                     break
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON message: {message[:100]}...")
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Connection closed while receiving messages.")
                    break
                except Exception as e:
                    logger.error(f"Error processing received message: {e}", exc_info=True)
                    # Should we break the loop on processing errors? Depends on severity.
                    # For now, log and continue trying to receive.


        # --- Run Tasks Concurrently ---
        logger.info("Starting worker tasks for client...")
        speech_task = asyncio.create_task(detect_speech_and_process(), name="SpeechProcessor")
        receive_task = asyncio.create_task(receive_messages(), name="MessageReceiver")
        keepalive_task = asyncio.create_task(send_keepalive(), name="KeepaliveSender")

        tasks = [speech_task, receive_task, keepalive_task]
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # --- Cleanup on Task Completion/Error ---
        logger.info(f"A task finished for client {client_id}. Cleaning up...")
        for task in pending:
             if not task.done():
                 logger.info(f"Cancelling pending task: {task.get_name()}")
                 task.cancel()
                 try:
                     await task # Await cancellation (with timeout?)
                 except asyncio.CancelledError:
                      logger.info(f"Task {task.get_name()} cancellation confirmed.")
                 except Exception as e:
                      logger.error(f"Error awaiting cancelled task {task.get_name()}: {e}")

        # Log results of completed tasks (optional)
        for task in done:
             try:
                  result = task.result()
                  logger.info(f"Task {task.get_name()} completed with result: {result}")
             except Exception as e:
                  logger.error(f"Task {task.get_name()} failed with exception: {e}", exc_info=True)


        logger.info(f"Finished cleaning up tasks for client {client_id}.")

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"Client {client_id} disconnected normally.")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"Client {client_id} connection closed error: {e}")
    except Exception as e:
        logger.error(f"Unhandled error in handle_client for {client_id}: {e}", exc_info=True)
    finally:
        logger.info(f"Ensuring final cleanup for client {client_id}")
        # Ensure any lingering tasks *associated with the detector* are cancelled
        await detector.cancel_current_tasks()
        # Cancel top-level tasks if somehow still pending (should be handled above)
        for task in [speech_task, receive_task, keepalive_task]:
            if task and not task.done():
                task.cancel()
        logger.info(f"Client {client_id} handler finished.")


# --- Main Server ---
async def main():
    """Initializes models and starts the WebSocket server."""
    # Initialize models eagerly at startup
    logger.info("--- Initializing Models ---")
    try:
        WhisperTranscriber.get_instance()
        GemmaMultimodalProcessor.get_instance()
        KokoroTTSProcessor.get_instance() # This will now use lang_code='a'
        logger.info("--- All Models Initialized ---")
    except Exception as e:
        # Error logging now happens inside model __init__ methods
        # logger.error(f"Failed to initialize models: {e}. Server cannot start.", exc_info=True)
        sys.exit(1) # Exit if essential models fail to load


    # Start the WebSocket server
    host = "0.0.0.0"
    port = 9074
    logger.info(f"Starting WebSocket server on {host}:{port}")

    MAX_WEBSOCKET_MESSAGE_SIZE = 10 * 1024 * 1024 # 10 MB example

    try:
        # Set higher recursion depth if needed for deep async calls (use with caution)
        # sys.setrecursionlimit(2000)

        server = await websockets.serve(
            handle_client,
            host,
            port,
            ping_interval=20,    # Send pings every 20s
            ping_timeout=30,     # Wait 30s for pong response
            close_timeout=10,    # Wait 10s for close handshake
            max_size=MAX_WEBSOCKET_MESSAGE_SIZE
        )
        logger.info(f"WebSocket server running on ws://{host}:{port}")
        await server.wait_closed() # Keep server running until stopped
    except OSError as e:
         logger.error(f"Failed to start server on {host}:{port}: {e}")
         logger.error("Is the port already in use?")
    except Exception as e:
        logger.error(f"Server encountered an unrecoverable error: {e}", exc_info=True)
    finally:
        logger.info("WebSocket server stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt).")
    except Exception as e:
         logger.critical(f"Unhandled exception in top-level execution: {e}", exc_info=True)
