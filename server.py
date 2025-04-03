#!/usr/bin/env python3
import asyncio
import json
import websockets
import base64
import torch
import numpy as np
import logging
import sys
import io
from PIL import Image
import time
import threading
from typing import Optional, AsyncGenerator, List

# --- Model Imports ---
# Whisper (using faster-whisper)
from faster_whisper import WhisperModel
# Gemma (using transformers)
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TextIteratorStreamer, BitsAndBytesConfig
# TTS (using Kokoro)
from kokoro import KPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 9073
AUDIO_SAMPLE_RATE = 16000
AUDIO_ENERGY_THRESHOLD = 0.015 # VAD sensitivity
AUDIO_SILENCE_DURATION = 0.6 # Seconds of silence to detect end of speech
AUDIO_MIN_SPEECH_DURATION = 0.4 # Seconds minimum speech length
AUDIO_MAX_SPEECH_DURATION = 10.0 # Seconds maximum speech segment length before forcing split
WHISPER_MODEL_SIZE = "large-v3" # Options: "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"
WHISPER_COMPUTE_TYPE = "float16" # Options: "float16", "int8_float16", "int8" (fastest, potentially less accurate), "float32"
GEMMA_MODEL_ID = "google/gemma-3-4b-it"
ENABLE_GEMMA_QUANTIZATION = True # Set to False to disable 4-bit quantization
KOKORO_VOICE = 'af_sarah'
KOKORO_SPEED = 1.0

# --- Audio Segment Detector ---
class AudioSegmentDetector:
    def __init__(self, sample_rate=AUDIO_SAMPLE_RATE, energy_threshold=AUDIO_ENERGY_THRESHOLD,
                 silence_duration=AUDIO_SILENCE_DURATION, min_speech_duration=AUDIO_MIN_SPEECH_DURATION,
                 max_speech_duration=AUDIO_MAX_SPEECH_DURATION):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_samples = int(silence_duration * sample_rate)
        self.min_speech_samples = int(min_speech_duration * sample_rate) * 2 # *2 for bytes (int16)
        self.max_speech_samples = int(max_speech_duration * sample_rate) * 2 # *2 for bytes (int16)
        self.audio_buffer = bytearray()
        self.is_speech_active = False
        self.silence_counter = 0
        self.speech_start_idx = 0
        self.buffer_lock = asyncio.Lock()
        self.segment_queue = asyncio.Queue(maxsize=1) # Only hold one segment to process immediately
        self.segments_detected = 0
        self._tts_playing = False
        self.tts_lock = asyncio.Lock()
        self.current_processing_task: Optional[asyncio.Task] = None
        self.task_lock = asyncio.Lock()
        logger.info(f"AudioSegmentDetector initialized (Threshold: {energy_threshold}, Silence: {silence_duration}s, MinSpeech: {min_speech_duration}s, MaxSpeech: {max_speech_duration}s)")

    async def is_tts_playing(self) -> bool:
        async with self.tts_lock:
            return self._tts_playing

    async def set_tts_playing(self, is_playing: bool):
        async with self.tts_lock:
            if self._tts_playing != is_playing:
                logger.info(f"TTS Playing state changed to: {is_playing}")
                self._tts_playing = is_playing

    async def cancel_current_tasks(self):
        """Cancels the main processing task if it's running."""
        async with self.task_lock:
            if self.current_processing_task and not self.current_processing_task.done():
                logger.warning("Interrupting current processing task due to new speech segment.")
                self.current_processing_task.cancel()
                try:
                    await self.current_processing_task # Allow cancellation to propagate
                except asyncio.CancelledError:
                    logger.info("Current processing task successfully cancelled.")
                except Exception as e:
                    logger.error(f"Error during task cancellation: {e}", exc_info=True)
            self.current_processing_task = None
            # Clear the queue as we are starting fresh
            while not self.segment_queue.empty():
                try:
                    self.segment_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            await self.set_tts_playing(False) # Ensure TTS is marked as stopped

    async def set_current_processing_task(self, task: Optional[asyncio.Task]):
        async with self.task_lock:
            self.current_processing_task = task

    async def add_audio(self, audio_bytes: bytes):
        """Adds audio data and detects speech segments."""
        if not audio_bytes:
            return None

        segment_to_process = None
        async with self.buffer_lock:
            self.audio_buffer.extend(audio_bytes)
            # Analyze only the newly added chunk for energy
            try:
                # Ensure buffer length is even for int16 conversion
                if len(audio_bytes) % 2 != 0:
                    logger.warning(f"Received audio chunk with odd length: {len(audio_bytes)}. Skipping last byte.")
                    audio_bytes = audio_bytes[:-1]
                if not audio_bytes: return None

                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                energy = np.sqrt(np.mean(audio_array**2)) if len(audio_array) > 0 else 0.0

                if not self.is_speech_active and energy > self.energy_threshold:
                    # Speech start detected
                    self.is_speech_active = True
                    # Start segment slightly before the trigger point if possible
                    look_back_samples = int(0.1 * self.sample_rate) * 2 # 100ms lookback in bytes
                    self.speech_start_idx = max(0, len(self.audio_buffer) - len(audio_bytes) - look_back_samples)
                    self.silence_counter = 0
                    logger.info(f"Speech start detected (energy: {energy:.4f})")

                elif self.is_speech_active:
                    if energy > self.energy_threshold:
                        # Continue speech
                        self.silence_counter = 0
                    else:
                        # Potential silence
                        self.silence_counter += len(audio_bytes)
                        if self.silence_counter >= self.silence_samples:
                            # Speech end detected by silence
                            speech_end_idx = len(self.audio_buffer) - self.silence_counter
                            speech_segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])
                            speech_duration = len(speech_segment_bytes) / 2 / self.sample_rate

                            self.is_speech_active = False
                            self.silence_counter = 0
                            # Keep only the silence part in buffer for next detection
                            self.audio_buffer = self.audio_buffer[speech_end_idx:]
                            self.speech_start_idx = 0 # Reset start index

                            if len(speech_segment_bytes) >= self.min_speech_samples:
                                self.segments_detected += 1
                                logger.info(f"Speech segment [Silence End] detected: {speech_duration:.2f}s ({len(speech_segment_bytes)} bytes) - Seg #{self.segments_detected}")
                                segment_to_process = speech_segment_bytes
                            else:
                                logger.info(f"Speech segment too short ({speech_duration:.2f}s). Discarding.")

                    # Check for max duration limit
                    current_speech_len = len(self.audio_buffer) - self.speech_start_idx
                    if self.is_speech_active and current_speech_len > self.max_speech_samples:
                        logger.info(f"Max speech duration ({self.max_speech_samples/2/self.sample_rate:.1f}s) reached. Forcing segment split.")
                        # Take the max duration segment
                        speech_segment_bytes = bytes(self.audio_buffer[self.speech_start_idx : self.speech_start_idx + self.max_speech_samples])
                        speech_duration = len(speech_segment_bytes) / 2 / self.sample_rate

                        # Update buffer: keep the part after the segment
                        self.audio_buffer = self.audio_buffer[self.speech_start_idx + self.max_speech_samples:]
                        # Reset speech start index for the remaining buffer
                        self.speech_start_idx = 0
                        # We are still potentially in speech, so don't reset is_speech_active or silence_counter yet

                        self.segments_detected += 1
                        logger.info(f"Speech segment [Max Duration] detected: {speech_duration:.2f}s ({len(speech_segment_bytes)} bytes) - Seg #{self.segments_detected}")
                        segment_to_process = speech_segment_bytes

            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}", exc_info=True)
                # Reset state on error to avoid getting stuck
                self.audio_buffer = bytearray()
                self.is_speech_active = False
                self.silence_counter = 0
                self.speech_start_idx = 0

        if segment_to_process:
            # If a segment is ready, cancel any ongoing processing for the *previous* segment
            await self.cancel_current_tasks()
            try:
                # Try putting the new segment in the queue. If full (shouldn't be with size 1 and cancel logic), log it.
                self.segment_queue.put_nowait(segment_to_process)
                logger.debug("Segment added to queue.")
            except asyncio.QueueFull:
                logger.warning("Segment queue was full unexpectedly. Discarding segment.")
            return segment_to_process # Return it for immediate feedback if needed

        return None

    async def get_next_segment(self) -> Optional[bytes]:
        """Gets the next detected speech segment from the queue."""
        try:
            # Short timeout to avoid blocking the event loop excessively
            return await asyncio.wait_for(self.segment_queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            return None
        except asyncio.QueueEmpty:
             return None # Should not happen with wait_for, but good practice

# --- Whisper Transcriber (using Faster Whisper) ---
class WhisperTranscriber:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Creating WhisperTranscriber instance...")
            cls._instance = cls()
        return cls._instance

    def __init__(self, model_size=WHISPER_MODEL_SIZE, compute_type=WHISPER_COMPUTE_TYPE):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = compute_type if self.device == "cuda" else "float32"
        self.model_size = model_size

        if self.device == "cuda" and not torch.cuda.is_available():
             logger.warning("CUDA specified but not available. Falling back to CPU.")
             self.device = "cpu"
             self.compute_type = "float32"

        logger.info(f"Loading Faster Whisper model: {self.model_size} (Compute: {self.compute_type}, Device: {self.device})")
        try:
            # device_index can be a list for multi-GPU [0, 1]
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type, device_index=0)
            self.transcription_count = 0
            logger.info("Faster Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Faster Whisper model: {e}", exc_info=True)
            raise RuntimeError("Whisper model initialization failed") from e

    async def transcribe(self, audio_bytes: bytes, sample_rate=AUDIO_SAMPLE_RATE) -> str:
        """Transcribes audio bytes using Faster Whisper."""
        try:
            # Ensure buffer length is even for int16 conversion
            if len(audio_bytes) % 2 != 0:
                logger.warning(f"Transcription received audio with odd length: {len(audio_bytes)}. Skipping last byte.")
                audio_bytes = audio_bytes[:-1]
            if not audio_bytes:
                logger.info("Audio empty, skipping transcription.")
                return ""

            # faster-whisper expects float32 numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_array) < sample_rate * 0.1: # Filter out very short segments (e.g., < 100ms)
                logger.info(f"Audio too short ({len(audio_array)/sample_rate:.3f}s) for transcription.")
                return ""

            start_time = time.perf_counter()

            # Run blocking CTranslate2 inference in an executor thread
            segments, info = await asyncio.get_event_loop().run_in_executor(
                None,  # Use default executor
                lambda: self.model.transcribe(
                    audio_array,
                    language="en",
                    task="transcribe",
                    beam_size=5, # Default, adjust if needed
                    # word_timestamps=False, # Disable for slight speedup if not needed
                    condition_on_previous_text=False # Usually faster for non-long form
                )
            )

            # Concatenate segments generator
            text = "".join(segment.text for segment in segments).strip()

            end_time = time.perf_counter()
            duration = end_time - start_time
            audio_duration = len(audio_array) / sample_rate

            self.transcription_count += 1
            logger.info(f"Transcription #{self.transcription_count} ({(duration*1000):.1f}ms, Audio: {audio_duration:.2f}s): '{text}'")
            # logger.debug(f"Detected language: {info.language} (Prob: {info.language_probability:.2f})")

            return text

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return "" # Return empty string on error

# --- Gemma Multimodal Processor ---
class GemmaMultimodalProcessor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Creating GemmaMultimodalProcessor instance...")
            cls._instance = cls()
        return cls._instance

    def __init__(self, model_id=GEMMA_MODEL_ID, use_quantization=ENABLE_GEMMA_QUANTIZATION):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.use_quantization = use_quantization if self.device == "cuda" else False # Quantization only on CUDA

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA specified but not available. Falling back to CPU for Gemma.")
            self.device = "cpu"
            self.use_quantization = False

        # Determine torch dtype (bfloat16 preferred on Ampere+)
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            self.torch_dtype = torch.bfloat16
            logger.info("Using bfloat16 for Gemma.")
        else:
            self.torch_dtype = torch.float16 # Fallback for older GPUs or CPU
            logger.info("Using float16 for Gemma.")


        # --- Quantization Configuration (Optional) ---
        quantization_config = None
        if self.use_quantization:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4", # Recommended type
                    bnb_4bit_compute_dtype=self.torch_dtype, # Dtype for computation after dequantization
                    bnb_4bit_use_double_quant=True, # Minor memory saving
                )
                logger.info("Enabled 4-bit quantization for Gemma (NF4, compute dtype: %s).", self.torch_dtype)
            except ImportError:
                logger.error("bitsandbytes library not found. Cannot enable 4-bit quantization.")
                quantization_config = None
            except Exception as e:
                logger.error(f"Error setting up quantization: {e}", exc_info=True)
                quantization_config = None
        else:
             logger.info("4-bit quantization for Gemma is disabled.")


        # --- Flash Attention Configuration ---
        attn_implementation = None
        if self.device == "cuda":
            # Check if flash_attn is installed and GPU is compatible (Compute Capability >= 8.0)
            try:
                if torch.cuda.get_device_capability()[0] >= 8:
                    # Test import flash_attn - will raise ImportError if not installed
                    import flash_attn
                    attn_implementation = "flash_attention_2"
                    logger.info("Flash Attention 2 detected and enabled for Gemma.")
                else:
                    logger.info("GPU compute capability lower than 8.0. Using default attention (SDPA if available, else eager).")
                    # Use 'sdpa' if available and desired, otherwise Transformers default
                    # attn_implementation = "sdpa" # Or None to let Transformers decide
            except ImportError:
                logger.warning("flash_attn library not installed. Cannot use Flash Attention 2.")
            except Exception as e:
                logger.warning(f"Could not automatically determine Flash Attention support: {e}")


        # --- Load Model ---
        logger.info(f"Loading Gemma model: {self.model_id} to device: {self.device}")
        s_time = time.time()
        try:
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map=self.device, # Map directly to the target device
                torch_dtype=self.torch_dtype if not quantization_config else None, # dtype for non-quantized parts, BnB handles compute dtype
                quantization_config=quantization_config, # Apply quantization if enabled

                low_cpu_mem_usage=True # Useful for large models
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            logger.info(f"Gemma model loaded in {time.time() - s_time:.2f} seconds.")
            logger.info(f"Gemma final config: Attn: {self.model.config.attn_implementation}, Quantized: {self.model.is_quantized}")

        except Exception as e:
            logger.error(f"Failed to load Gemma model: {e}", exc_info=True)
            raise RuntimeError("Gemma model initialization failed") from e

        self.last_image: Optional[Image.Image] = None
        self.image_lock = asyncio.Lock()
        self.message_history = [] # Store tuples of (user_text, assistant_response)
        self.generation_count = 0
        self.max_history_len = 3 # Keep last 3 turns (User + Assistant = 1 turn)

    async def set_image(self, image_data: bytes):
        """Sets the image for the next multimodal interaction."""
        async with self.image_lock:
            try:
                if not image_data or len(image_data) < 100: # Basic sanity check
                    logger.warning("Invalid or empty image data received.")
                    return False
                image = Image.open(io.BytesIO(image_data)).convert("RGB") # Ensure RGB
                # Optional: Resize if images are very large
                # max_size = (1024, 1024)
                # image.thumbnail(max_size, Image.Resampling.LANCZOS)
                self.last_image = image
                 # Clear history when a new image context is provided
                self.message_history = []
                logger.info(f"Image set successfully (Size: {image.size}). History cleared.")
                return True
            except Exception as e:
                logger.error(f"Error processing image: {e}", exc_info=True)
                self.last_image = None # Ensure last_image is cleared on error
                return False

    def _build_messages(self, text: str) -> List[dict]:
        """Builds the message list for the Gemma prompt, including history and image."""
        # System prompt
        messages = [{"role": "system", "content": [{"type": "text", "text": "You are K, a friendly and helpful multimodal AI assistant. Respond concisely and naturally, as if in a spoken conversation. If an image is provided, describe it or answer questions about it relevant to the user's query. Otherwise, engage in the ongoing conversation."}]}]

        # Add conversation history
        for user_msg, assistant_msg in self.message_history:
            messages.append({"role": "user", "content": [{"type": "text", "text": user_msg}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_msg}]})

        # Add current user turn (with image if available)
        user_content = []
        asyncio.run(self.image_lock.acquire()) # Need sync access to check last_image
        if self.last_image:
            user_content.append({"type": "image"}) # Placeholder for processor
        self.image_lock.release()
        user_content.append({"type": "text", "text": text})
        messages.append({"role": "user", "content": user_content})

        return messages

    def _update_history(self, user_text: str, assistant_response: str):
        """Adds the latest turn to the history and truncates if necessary."""
        self.message_history.append((user_text, assistant_response))
        if len(self.message_history) > self.max_history_len:
            self.message_history = self.message_history[-self.max_history_len:]
        # logger.debug(f"History updated. Turns: {len(self.message_history)}")


    async def generate_streaming(self, text: str) -> Optional[TextIteratorStreamer]:
        """Generates text response using Gemma in a streaming fashion."""
        start_time = time.perf_counter()
        try:
            messages = self._build_messages(text)

            # Handle image processing (needs to be sync within the lock for safety)
            async with self.image_lock:
                prompt_kwargs = {"image": self.last_image} if self.last_image else {}

            # Prepare inputs (tokenization) - must happen after image check
            # Let processor handle image integration based on template
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True, # Add the prompt for the assistant's turn
                tokenize=True,
                return_tensors="pt",
                 **prompt_kwargs # Pass image here if present
            ).to(self.model.device)

             # Ensure streamer is created fresh for each call
            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True, # Don't stream the input prompt
                skip_special_tokens=True
            )

            # Generation arguments
            generation_kwargs = dict(
                inputs, # Contains input_ids, attention_mask, pixel_values (if image)
                streamer=streamer,
                max_new_tokens=150, # Limit response length
                do_sample=True,     # Use sampling for more natural responses
                temperature=0.7,    # Control randomness
                top_p=0.9,          # Nucleus sampling
                use_cache=True      # Important for generation speed
            )

            # Run generation in a separate thread to avoid blocking asyncio event loop
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            # Don't wait for thread completion here; return the streamer immediately

            self.generation_count += 1
            duration = time.perf_counter() - start_time
            logger.info(f"Gemma generation #{self.generation_count} started ({(duration*1000):.1f}ms prep time). Image included: {self.last_image is not None}")

             # Forget the image after using it for this generation turn
            async with self.image_lock:
                self.last_image = None
                logger.debug("Image context used and cleared for this turn.")

            return streamer

        except Exception as e:
            logger.error(f"Gemma streaming generation error: {e}", exc_info=True)
            # Clean up image context if an error occurred before clearing it
            async with self.image_lock:
                self.last_image = None
            return None # Indicate failure


# --- Kokoro TTS Processor ---
class KokoroTTSProcessor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
             logger.info("Creating KokoroTTSProcessor instance...")
             cls._instance = cls()
        return cls._instance

    def __init__(self, lang_code='a', voice=KOKORO_VOICE, speed=KOKORO_SPEED):
        logger.info(f"Loading Kokoro TTS (Voice: {voice}, Speed: {speed})...")
        s_time = time.time()
        try:
            self.pipeline = KPipeline(lang_code=lang_code)
            self.default_voice = voice
            self.default_speed = speed
            # Pre-warm? Generate a short silent sample?
            # self.pipeline(" ", voice=self.default_voice, speed=self.default_speed)
            self.synthesis_count = 0
            logger.info(f"Kokoro TTS loaded in {time.time() - s_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to load Kokoro TTS model: {e}", exc_info=True)
            raise RuntimeError("Kokoro TTS initialization failed") from e

    async def synthesize_speech(self, text: str) -> Optional[np.ndarray]:
        """Synthesizes speech from text using Kokoro TTS."""
        if not text or not text.strip():
            # logger.debug("Empty text received for TTS, skipping.")
            return None
        if not self.pipeline:
            logger.error("Kokoro TTS pipeline not initialized.")
            return None

        start_time = time.perf_counter()
        try:
            audio_segments = []
            # Run blocking TTS generation in an executor thread
            # Split pattern helps create slightly more natural pauses for streaming
            generator = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipeline(
                    text,
                    voice=self.default_voice,
                    speed=self.default_speed,
                    split_pattern=r'[.!?,\-;:]+' # Split on common punctuation
                )
            )

            for _, _, audio in generator:
                if audio is not None and len(audio) > 0:
                    audio_segments.append(audio)

            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                duration = time.perf_counter() - start_time
                self.synthesis_count += 1
                logger.info(f"TTS Synthesis #{self.synthesis_count} ({(duration*1000):.1f}ms, Samples: {len(combined_audio)}): '{text[:50]}...'")
                return combined_audio
            else:
                logger.warning(f"TTS generation yielded no audio for text: '{text[:50]}...'")
                return None

        except Exception as e:
            logger.error(f"Kokoro TTS error for text '{text[:50]}...': {e}", exc_info=True)
            return None

# --- WebSocket Client Handler ---
async def handle_client(websocket: websockets.WebSocketServerProtocol):
    """Handles a single client connection."""
    client_id = websocket.remote_address
    logger.info(f"Client connected: {client_id}")

    # Instantiate processors per connection using singletons
    try:
        detector = AudioSegmentDetector() # Separate state per connection
        transcriber = WhisperTranscriber.get_instance() # Shared model
        gemma_processor = GemmaMultimodalProcessor.get_instance() # Shared model
        tts_processor = KokoroTTSProcessor.get_instance() # Shared model
    except Exception as e:
        logger.error(f"Failed to initialize models for client {client_id}: {e}", exc_info=True)
        try:
            await websocket.close(code=1011, reason="ModelInitializationError")
        except websockets.exceptions.ConnectionClosed:
            pass # Ignore if already closed
        return # Stop handling this client

    # --- Core Processing Task ---
    async def process_speech_segment(speech_segment: bytes):
        """Processes a single detected speech segment."""
        try:
            # 1. Transcribe Audio to Text
            transcription = await transcriber.transcribe(speech_segment)
            if not transcription or not any(c.isalnum() for c in transcription):
                logger.info(f"Skipping empty or non-alphanumeric transcription: '{transcription}'")
                return # Don't proceed if transcription is empty/useless

            await detector.set_tts_playing(True) # Mark TTS as active

            # 2. Generate Text Response (Streaming)
            streamer: Optional[TextIteratorStreamer] = await gemma_processor.generate_streaming(transcription)

            if streamer is None:
                logger.error("Gemma failed to generate a streaming response.")
                error_speech = "Sorry, I encountered an issue generating a response."
                # Synthesize and send error audio
                audio_data = await tts_processor.synthesize_speech(error_speech)
                if audio_data is not None:
                    audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
                    await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
                return # Stop processing this segment

            # 3. Stream TTS based on Gemma's output
            full_response = ""
            sentence_buffer = ""
            # More comprehensive sentence terminators
            sentence_end_chars = {'.', '!', '?', '\n'}
            min_chunk_len = 15 # Min chars before attempting TTS for a chunk

            active_tts_tasks: List[asyncio.Task] = []

            async def play_audio_chunk(text_chunk: str):
                """Inner function to synthesize and send one audio chunk."""
                if not text_chunk or not text_chunk.strip():
                    return
                logger.debug(f"Synthesizing chunk: '{text_chunk[:60]}...'")
                audio_data = await tts_processor.synthesize_speech(text_chunk)
                if audio_data is not None and len(audio_data) > 0:
                    try:
                        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
                        await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
                        logger.debug(f"Sent audio chunk ({len(audio_bytes)} bytes)")
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("Connection closed while sending audio chunk.")
                        raise # Re-raise to be caught by outer loop
                    except Exception as e:
                        logger.error(f"Error sending audio chunk: {e}", exc_info=True)
                else:
                    logger.warning(f"TTS failed or produced empty audio for chunk: '{text_chunk[:60]}...'")

            # Process the stream from Gemma
            try:
                for chunk in streamer: # This loop yields control, allowing TTS tasks to run
                    if not isinstance(chunk, str): continue # Safety check
                    full_response += chunk
                    sentence_buffer += chunk

                    # Check if the buffer likely ends a sentence or clause, or is long enough
                    ends_sentence = sentence_buffer and sentence_buffer[-1] in sentence_end_chars
                    long_enough = len(sentence_buffer) > min_chunk_len

                    # Trigger TTS if sentence ends OR buffer is long and ends with comma/semicolon (natural pause)
                    if ends_sentence or (long_enough and sentence_buffer[-1] in {',', ';', ':'}):
                        text_to_speak = sentence_buffer.strip()
                        sentence_buffer = "" # Reset buffer
                        if text_to_speak:
                            # Create TTS task and let it run in the background
                            task = asyncio.create_task(play_audio_chunk(text_to_speak))
                            active_tts_tasks.append(task)

                    # Prevent tasks list from growing indefinitely if TTS is slow
                    # Remove completed tasks (non-blocking check)
                    active_tts_tasks = [t for t in active_tts_tasks if not t.done()]

                # After Gemma finishes, process any remaining text in the buffer
                text_to_speak = sentence_buffer.strip()
                if text_to_speak:
                    task = asyncio.create_task(play_audio_chunk(text_to_speak))
                    active_tts_tasks.append(task)

                # Wait for all TTS tasks for this response to complete
                if active_tts_tasks:
                     logger.debug(f"Waiting for {len(active_tts_tasks)} TTS task(s) to complete...")
                     await asyncio.gather(*active_tts_tasks, return_exceptions=True) # Wait and log errors if any
                     logger.debug("All TTS tasks for this segment finished.")


            except websockets.exceptions.ConnectionClosed:
                 logger.warning("Connection closed during Gemma streaming/TTS.")
                 # Cancel any remaining TTS tasks
                 for task in active_tts_tasks:
                     if not task.done():
                         task.cancel()
                 return # Exit processing
            except Exception as stream_ex:
                 logger.error(f"Error during Gemma streaming or TTS chunking: {stream_ex}", exc_info=True)
                 # Attempt to send a generic error message
                 error_speech = "Sorry, an error occurred while generating the response audio."
                 await play_audio_chunk(error_speech) # Use the inner function
                 # Cancel any remaining TTS tasks
                 for task in active_tts_tasks:
                      if not task.done():
                          task.cancel()

            # 4. Update Gemma's history (only if response was generated)
            if full_response:
                gemma_processor._update_history(transcription, full_response.strip())

        except asyncio.CancelledError:
            logger.info("Processing task cancelled.")
            # TTS state should be handled by cancel_current_tasks or finally block
            # Re-raise to ensure the task runner knows it was cancelled
            raise

        except Exception as e:
            logger.error(f"Error in process_speech_segment: {e}", exc_info=True)
            # Attempt to send an error message if connection is still open
            try:
                error_speech = "I encountered an unexpected error. Please try again."
                audio_data = await tts_processor.synthesize_speech(error_speech)
                if audio_data is not None:
                    audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
                    await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
            except Exception as send_err:
                logger.error(f"Could not send final error audio: {send_err}")
        finally:
            # Ensure TTS state is reset regardless of how the function exits
            await detector.set_tts_playing(False)

    # --- Background Task: Detect Speech Segments ---
    async def detect_speech_segments():
        """Runs in the background, waiting for segments from the detector."""
        while True:
            try:
                speech_segment = await detector.get_next_segment()
                if speech_segment:
                    # Cancel any previous task (handled by detector.add_audio now)
                    # Create and store the new processing task
                    logger.info("Creating new processing task...")
                    task = asyncio.create_task(process_speech_segment(speech_segment))
                    await detector.set_current_processing_task(task)
                    # Optional: Wait briefly to allow task to start? Not usually needed.
                else:
                    # No segment ready, sleep briefly to yield control
                    await asyncio.sleep(0.01) # Short sleep to prevent busy-waiting
            except asyncio.CancelledError:
                logger.info("Speech detection loop cancelled.")
                break # Exit loop if cancelled
            except Exception as e:
                logger.error(f"Error in speech detection loop: {e}", exc_info=True)
                await asyncio.sleep(1) # Wait a bit longer after an error

    # --- Background Task: Receive Client Messages ---
    async def receive_messages():
        """Runs in the background, receiving messages from the client."""
        async for message in websocket:
            try:
                data = json.loads(message)
                tts_active = await detector.is_tts_playing()

                # Handle audio chunks
                if "realtime_input" in data and "media_chunks" in data["realtime_input"]:
                    for chunk in data["realtime_input"]["media_chunks"]:
                        if chunk.get("mime_type") == "audio/pcm":
                            audio_data = base64.b64decode(chunk.get("data", ""))
                            if audio_data:
                                await detector.add_audio(audio_data)
                        elif chunk.get("mime_type") == "image/jpeg":
                            if tts_active:
                                logger.info("Ignoring image received while TTS is playing.")
                                continue
                            image_data = base64.b64decode(chunk.get("data", ""))
                            if image_data:
                                logger.info("Received image chunk.")
                                await gemma_processor.set_image(image_data)
                            else:
                                logger.warning("Received empty image chunk data.")

                # Handle standalone image messages
                elif "image" in data:
                    if tts_active:
                         logger.info("Ignoring standalone image received while TTS is playing.")
                         continue
                    image_data = base64.b64decode(data.get("image", ""))
                    if image_data:
                         logger.info("Received standalone image message.")
                         await gemma_processor.set_image(image_data)
                    else:
                        logger.warning("Received empty standalone image data.")

                # Handle other message types if needed
                # elif "text" in data:
                #     # Example: Allow text input to interrupt/override speech
                #     if tts_active:
                #         logger.info("Text received, interrupting current speech/TTS.")
                #         await detector.cancel_current_tasks()
                #     # Process text input... (Needs separate processing logic)

            except json.JSONDecodeError:
                logger.warning(f"Received invalid JSON from {client_id}")
            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed while receiving messages.")
                break # Exit loop
            except asyncio.CancelledError:
                logger.info("Message receiving loop cancelled.")
                break # Exit loop
            except Exception as e:
                logger.error(f"Error processing message from {client_id}: {e}", exc_info=True)
                # Continue receiving messages if possible

    # --- Start Background Tasks ---
    detect_task = asyncio.create_task(detect_speech_segments())
    receive_task = asyncio.create_task(receive_messages())

    logger.info(f"Started background tasks for client {client_id}")

    # Keep connection alive by handling potential closure and waiting for tasks
    try:
        # Wait until one of the tasks finishes (usually receive_messages on disconnect)
        done, pending = await asyncio.wait(
            [detect_task, receive_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Log which task finished
        for task in done:
            task_name = task.get_coro().__name__ if hasattr(task.get_coro(), '__name__') else 'unknown task'
            logger.info(f"Task '{task_name}' completed for client {client_id}.")
            try:
                 task.result() # Check for exceptions in the completed task
            except asyncio.CancelledError:
                 logger.info(f"Task '{task_name}' was cancelled.")
            except Exception as e:
                 logger.error(f"Task '{task_name}' raised an exception: {e}", exc_info=True)


    except Exception as e:
        logger.error(f"Error during main client handling loop for {client_id}: {e}", exc_info=True)

    finally:
        logger.info(f"Cleaning up tasks for client {client_id}...")
        # Cancel all pending tasks associated with this client connection
        for task in pending:
            task.cancel()
        if not detect_task.done(): detect_task.cancel()
        if not receive_task.done(): receive_task.cancel()

        # Also cancel the currently active processing task, if any
        await detector.cancel_current_tasks()

        # Wait briefly for cancellations to propagate
        await asyncio.sleep(0.1)

        # Attempt to close the websocket cleanly if it's not already closed
        if websocket.open:
             try:
                 await websocket.close(code=1000, reason="ClientDisconnecting")
                 logger.info(f"WebSocket connection closed cleanly for {client_id}.")
             except websockets.exceptions.ConnectionClosed:
                 logger.info(f"WebSocket connection already closed for {client_id}.") # Expected if client disconnected first
             except Exception as e:
                 logger.error(f"Error closing websocket for {client_id}: {e}")
        else:
            logger.info(f"WebSocket connection was already closed for {client_id}.")

        logger.info(f"Client {client_id} disconnected and cleaned up.")


# --- Main Server Function ---
async def main():
    # Initialize singleton models eagerly at startup
    logger.info("Pre-initializing models...")
    try:
        WhisperTranscriber.get_instance()
        GemmaMultimodalProcessor.get_instance()
        KokoroTTSProcessor.get_instance()
        logger.info("All models pre-initialized successfully.")
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to initialize models on startup: {e}", exc_info=True)
        logger.critical("Server cannot start without models. Exiting.")
        sys.exit(1) # Exit if essential models fail

    logger.info(f"Starting WebSocket server on {WEBSOCKET_HOST}:{WEBSOCKET_PORT}")

    # Configure WebSocket server settings for robustness
    async with websockets.serve(
        handle_client,
        WEBSOCKET_HOST,
        WEBSOCKET_PORT,
        ping_interval=20,      # Send pings every 20 seconds
        ping_timeout=40,       # Wait 40 seconds for a pong response
        close_timeout=10,      # Wait 10 seconds for graceful close handshake
        max_size=2**22,        # Increase max message size (e.g., 4MB) for images/long audio
        # read_limit=2**22,    # Corresponds to max_size in recent versions
        # write_limit=2**22,   # Corresponds to max_size in recent versions
        compression=None       # Disable compression unless tested for performance impact
    ):
        logger.info("Server started successfully. Waiting for connections...")
        # Keep the server running indefinitely
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt).")
    except Exception as e:
        logger.critical(f"Server encountered critical error: {e}", exc_info=True)
        sys.exit(1)
