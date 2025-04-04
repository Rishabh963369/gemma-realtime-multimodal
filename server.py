import asyncio
import json
import websockets
import base64
import torch
import numpy as np
import logging
import sys
import time
import io # Needed for image processing
from PIL import Image # Needed for image processing
from accelerate import Accelerator
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoModelForCausalLM, # Gemma 2 uses this for multimodal
    AutoProcessor,         # Use AutoProcessor for both Whisper and Gemma Multimodal
    pipeline,
    TextIteratorStreamer,
    BitsAndBytesConfig
)

# Optional TTS - Attempt to import Kokoro
try:
    from kokoro import KPipeline # Replace if your import path is different
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    class KPipeline:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs):
             print("Warning: KokoroTTS called, but library is not available.")
             yield None, None, np.array([], dtype=np.float32)

# --- Configuration Constants ---
WHISPER_MODEL_ID = "openai/whisper-large-v3"
# *** Using Gemma 2 9B Instruct (Multimodal) as requested ***
GEMMA_MODEL_ID = "google/gemma-2-9b-it" # Switched back to Gemma 2 9B for image support
WEBSOCKET_PORT = 9073
AUDIO_SAMPLE_RATE = 16000
DEFAULT_TTS_VOICE = 'en_amy' # Example Kokoro voice (adjust)
TORCH_DTYPE = torch.bfloat16 # Recommended
IMAGE_RESIZE_FACTOR = 0.75 # Optional: Resize images slightly to save tokens/processing
# ---

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
# ---

# --- Check for Flash Attention 2 ---
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
    logger.info("Flash Attention 2 found, will be used if applicable.")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    logger.info("Flash Attention 2 not found. Using default attention mechanisms (SDPA if available).")
ATTN_IMPLEMENTATION = "flash_attention_2" if FLASH_ATTN_AVAILABLE else "sdpa"
logger.info(f"Selected attention implementation: {ATTN_IMPLEMENTATION}")
# ---

# --- Audio Segment Detector Class ---
# (Keep your existing AudioSegmentDetector class code here - it's unchanged)
class AudioSegmentDetector:
    def __init__(self, sample_rate=AUDIO_SAMPLE_RATE, energy_threshold=0.015, silence_duration=0.5, min_speech_duration=0.5, max_speech_duration=10):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_samples = int(silence_duration * sample_rate)
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.max_speech_samples = int(max_speech_duration * sample_rate)
        self.audio_buffer = bytearray()
        self.is_speech_active = False
        self.silence_counter = 0
        self.speech_start_idx = 0
        self.lock = asyncio.Lock()
        self.segment_queue = asyncio.Queue(maxsize=1) # Queue for detected speech segments
        self.segments_detected = 0
        self.tts_playing = False
        self.tts_lock = asyncio.Lock()
        self.current_processing_task = None # Unified task for generation/TTS
        self.task_lock = asyncio.Lock() # Lock for managing cancellation

    async def set_tts_playing(self, is_playing):
        async with self.tts_lock:
            self.tts_playing = is_playing
            # logger.debug(f"TTS Playing set to: {is_playing}")

    async def cancel_current_tasks(self):
        """Cancels the current speech processing task."""
        async with self.task_lock:
            cancelled_something = False
            if self.current_processing_task and not self.current_processing_task.done():
                self.current_processing_task.cancel()
                try:
                    await self.current_processing_task
                except asyncio.CancelledError:
                    # logger.info("Current processing task cancelled successfully.")
                    pass
                except Exception as e:
                    logger.error(f"Error during task cancellation wait: {e}")
                cancelled_something = True

            self.current_processing_task = None

            while not self.segment_queue.empty():
                try: self.segment_queue.get_nowait()
                except asyncio.QueueEmpty: break

            await self.set_tts_playing(False) # Ensure TTS is marked as stopped
            # if cancelled_something: logger.info("Cancellation complete.")

    async def set_current_tasks(self, processing_task=None):
        """Sets the current active processing task."""
        async with self.task_lock:
            self.current_processing_task = processing_task
            # logger.debug(f"Set current task: {processing_task is not None}")

    async def add_audio(self, audio_bytes):
        """Adds audio data and detects speech segments based on VAD logic."""
        async with self.tts_lock:
            if self.tts_playing: return None # Don't process audio if TTS is active

        async with self.lock:
            self.audio_buffer.extend(audio_bytes)
            chunk_samples = len(audio_bytes) // 2
            if chunk_samples == 0: return None

            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if np.any(audio_array): energy = np.sqrt(np.mean(audio_array**2))
            else: energy = 0.0

            detected_segment = None
            if not self.is_speech_active:
                if energy > self.energy_threshold:
                    self.is_speech_active = True
                    self.speech_start_idx = len(self.audio_buffer) - len(audio_bytes)
                    self.silence_counter = 0
                    logger.info(f"Speech start detected (energy: {energy:.6f})")
            else: # Speech is active
                if energy > self.energy_threshold: self.silence_counter = 0
                else: self.silence_counter += chunk_samples

                if self.silence_counter >= self.silence_samples:
                    speech_end_idx = len(self.audio_buffer) - self.silence_counter * 2
                    segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])
                    segment_duration_samples = len(segment_bytes) // 2
                    self.is_speech_active = False
                    self.audio_buffer = self.audio_buffer[speech_end_idx:]
                    self.speech_start_idx = 0; self.silence_counter = 0
                    if segment_duration_samples >= self.min_speech_samples:
                        logger.info(f"Speech segment detected (Silence): {segment_duration_samples / self.sample_rate:.2f}s")
                        detected_segment = segment_bytes
                    else: logger.info(f"Discarding short segment (Silence): {segment_duration_samples / self.sample_rate:.2f}s")

                current_buffer_len = len(self.audio_buffer)
                if self.is_speech_active and self.speech_start_idx < current_buffer_len:
                    current_speech_duration_samples = (current_buffer_len - self.speech_start_idx) // 2
                    if current_speech_duration_samples > self.max_speech_samples:
                        speech_end_idx = self.speech_start_idx + self.max_speech_samples * 2
                        segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])
                        self.audio_buffer = self.audio_buffer[speech_end_idx:]
                        self.speech_start_idx = 0; self.silence_counter = 0
                        logger.info(f"Max duration speech segment extracted: {len(segment_bytes) / 2 / self.sample_rate:.2f}s")
                        detected_segment = segment_bytes
                elif self.is_speech_active and self.speech_start_idx >= current_buffer_len:
                    logger.warning("Inconsistent state in VAD. Resetting.")
                    self.is_speech_active = False; self.speech_start_idx = 0; self.silence_counter = 0

            if detected_segment:
                self.segments_detected += 1
                await self.cancel_current_tasks()
                while not self.segment_queue.empty():
                    try: self.segment_queue.get_nowait()
                    except asyncio.QueueEmpty: break
                await self.segment_queue.put(detected_segment)
                return detected_segment
            return None

    async def get_next_segment(self):
        """Retrieves the next detected speech segment from the queue."""
        try: return await asyncio.wait_for(self.segment_queue.get(), timeout=0.05)
        except asyncio.TimeoutError: return None
# ---

# --- Whisper Transcriber Class ---
# (Keep your existing WhisperTranscriber class code here)
# (Using device_map="auto" and direct model inference is appropriate)
class WhisperTranscriber:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Initializing WhisperTranscriber...")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if WhisperTranscriber._instance is not None: raise Exception("WhisperTranscriber is a singleton!")
        self.accelerator = Accelerator()
        logger.info(f"Attempting to load Whisper model {WHISPER_MODEL_ID} using device_map='auto' with {TORCH_DTYPE}")
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                WHISPER_MODEL_ID, torch_dtype=TORCH_DTYPE, low_cpu_mem_usage=True,
                use_safetensors=True, device_map="auto",
                # attn_implementation=ATTN_IMPLEMENTATION # Add if needed and stable
            )
            self.processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
            self.transcription_count = 0
            logger.info(f"Whisper model {WHISPER_MODEL_ID} loaded successfully using device_map.")
            try: logger.info(f"Whisper device map: {self.model.hf_device_map}")
            except AttributeError: logger.info("Could not retrieve detailed Whisper device map.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True); raise

    async def transcribe(self, audio_bytes, sample_rate=AUDIO_SAMPLE_RATE):
        if not audio_bytes or len(audio_bytes) < 1000:
             logger.warning(f"Audio too short for transcription ({len(audio_bytes)} bytes)"); return ""
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            inputs = self.processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
            with torch.no_grad():
                predicted_ids = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.model.generate(
                        inputs["input_features"].to(TORCH_DTYPE), language="english",
                        task="transcribe", max_new_tokens=128
                    )
                )
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            text = transcription.strip()
            self.transcription_count += 1
            logger.info(f"Transcription #{self.transcription_count}: '{text}'")
            return text
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True); return ""
# ---

# --- Gemma Multimodal Processor Class ---
class GemmaMultimodalProcessor: # Renamed back
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Initializing GemmaMultimodalProcessor...")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if GemmaMultimodalProcessor._instance is not None:
            raise Exception("GemmaMultimodalProcessor is a singleton!")

        self.accelerator = Accelerator()
        logger.info(f"Attempting to load Gemma model {GEMMA_MODEL_ID} using device_map='auto' with {TORCH_DTYPE}")

        try:
            # Use AutoProcessor for multimodal models (includes tokenizer + image processor)
            self.processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)
            # AutoModelForCausalLM works for Gemma 2 MM
            self.model = AutoModelForCausalLM.from_pretrained(
                GEMMA_MODEL_ID,
                device_map="auto",
                torch_dtype=TORCH_DTYPE,
                attn_implementation=ATTN_IMPLEMENTATION
            )

            # Re-add image attributes
            self.last_image = None
            self.last_image_timestamp = 0
            self.lock = asyncio.Lock() # Lock for accessing shared image/history state
            self.message_history = []
            self.generation_count = 0
            logger.info(f"Gemma model {GEMMA_MODEL_ID} loaded successfully using device_map.")
            try: logger.info(f"Gemma device map: {self.model.hf_device_map}")
            except AttributeError: logger.info("Could not retrieve detailed Gemma device map.")

        except Exception as e:
             logger.error(f"Failed to load Gemma model: {e}", exc_info=True)
             raise

    # Re-add set_image method exactly as provided by user/initial code
    async def set_image(self, image_data):
        """Processes and stores the latest image received."""
        async with self.lock:
            try:
                if not image_data or len(image_data) < 100:
                    logger.warning("Invalid or empty image data received")
                    return False
                image = Image.open(io.BytesIO(image_data)).convert("RGB") # Ensure RGB

                # Optional resize (keep if desired)
                if IMAGE_RESIZE_FACTOR < 1.0:
                    width, height = image.size
                    resized_image = image.resize((int(width * IMAGE_RESIZE_FACTOR), int(height * IMAGE_RESIZE_FACTOR)), Image.Resampling.LANCZOS)
                    self.last_image = resized_image
                    logger.info(f"Image set successfully (resized to {resized_image.size})")
                else:
                    self.last_image = image # Use original image
                    logger.info(f"Image set successfully (original size: {image.size})")

                # Crucially, clear history when a new image context is set
                self.message_history = []
                self.last_image_timestamp = time.time()
                logger.info("Gemma conversation history cleared due to new image.")
                return True
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}", exc_info=True)
                self.last_image = None # Ensure image is cleared on error
                return False

    # Builds the message structure for the multimodal model
    def _build_messages(self, text):
        """Constructs the message list including history and current turn."""
        # Gemma 2 uses a specific turn structure.
        # The processor handles the <image> token replacement.
        messages = []
        # Add history
        messages.extend(self.message_history)

        # Add current user turn (potentially with placeholder for image)
        user_content = []
        if self.last_image is not None:
            # The processor expects an object/dict signaling an image,
            # or rely on passing the image via the `images` kwarg to processor call.
            # Let's structure it clearly for the template:
             user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": text})
        messages.append({"role": "user", "content": user_content})
        # logger.debug(f"Built messages structure: {messages}")
        return messages


    # Updates history (simple version: keep only last Q&A)
    def _update_history(self, user_text, assistant_response):
         # Reconstruct user turn content (text only, image is implicit context)
        user_turn = {"role": "user", "content": [{"type": "text", "text": user_text}]}
        # Assistant turn
        assistant_turn = {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]}
        # Keep only the last interaction
        self.message_history = [user_turn, assistant_turn]
        logger.debug("Gemma history updated with last interaction.")


    async def generate_streaming(self, text):
        """Generates text response stream, considering the last image if available."""
        try:
            current_image = None
            # Safely get the current image and build messages under lock
            async with self.lock:
                 messages = self._build_messages(text)
                 current_image = self.last_image # Get the PIL image object

            # Prepare inputs using the multimodal processor
            # Pass BOTH text (via messages structure) and the actual image
            try:
                if current_image:
                    # Processor combines text messages and image data
                    inputs = self.processor(text=messages, images=current_image, return_tensors="pt", add_generation_prompt=True).to(self.model.device)
                    logger.info("Processing text query with image context.")
                else:
                    # Text-only processing
                    inputs = self.processor(text=messages, return_tensors="pt", add_generation_prompt=True).to(self.model.device)
                    logger.info("Processing text query without image context.")
            except Exception as proc_e:
                 logger.error(f"Error during multimodal processing: {proc_e}", exc_info=True)
                 return None, "Sorry, I had trouble understanding the input with the image."


            streamer = TextIteratorStreamer(
                self.processor.tokenizer, # Use tokenizer from the AutoProcessor
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=256, # Adjust as needed
                do_sample=False,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.eos_token_id # Use EOS for padding
             )

            # Run generation in executor thread
            loop = asyncio.get_running_loop()
            generation_future = loop.run_in_executor(None, lambda: self.model.generate(**generation_kwargs))

            # Stream initial part
            initial_text = ""
            timeout_seconds = 5.0; start_time = time.time(); first_chunk = True
            try:
                 async for chunk in streamer:
                      initial_text += chunk
                      if first_chunk: logger.info(f"First Gemma chunk received after {(time.time()-start_time):.2f}s") ; first_chunk=False
                      if time.time() - start_time > timeout_seconds and not initial_text:
                           logger.warning("Timeout waiting for initial text chunk."); break
                      if "." in chunk or "!" in chunk or "?" in chunk or "\n" in chunk or len(initial_text) > 30: break
                 else: logger.info("Streamer finished before initial break.")
            except asyncio.CancelledError: logger.info("Gemma initial streaming cancelled."); raise
            except Exception as e:
                 logger.error(f"Error during initial Gemma streaming: {e}", exc_info=True)
                 if not generation_future.done(): generation_future.cancel()
                 return None, "Sorry, encountered error during generation."

            self.generation_count += 1
            logger.info(f"Gemma Initial Text #{self.generation_count}: '{initial_text.strip()}'")

            return streamer, initial_text.strip()

        except Exception as e:
            logger.error(f"Gemma streaming setup error: {e}", exc_info=True)
            return None, "Sorry, couldn’t prepare the response generator."
# ---

# --- Kokoro TTS Processor Class ---
# (Keep your existing KokoroTTSProcessor class code here)
class KokoroTTSProcessor:
    _instance = None
    @classmethod
    def get_instance(cls):
        if not KOKORO_AVAILABLE: return None
        if cls._instance is None:
            logger.info("Initializing KokoroTTSProcessor...")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if not KOKORO_AVAILABLE: self.pipeline = None; return
        if KokoroTTSProcessor._instance is not None: raise Exception("KokoroTTSProcessor is a singleton!")
        try:
            self.pipeline = KPipeline(lang_code='a') # <-- CORRECTED LINE
            # You might need to adjust the voice. Let's try 'af_sarah' as used before or check Kokoro docs.
            # If 'en_amy' fails later during synthesis, change this voice name.
            self.default_voice = 'af_sarah' # Changed voice as well, might be more compatible
            self.synthesis_count = 0
            logger.info(f"Kokoro TTS loaded with lang_code='a' and default voice 
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS: {e}", exc_info=True); self.pipeline = None

    async def synthesize_speech(self, text):
        if not self.pipeline: logger.error("KokoroTTS pipeline unavailable."); return None
        if not text or not text.strip(): logger.warning("TTS empty text."); return None
        try:
            def sync_synthesize():
                audio_segments = []
                try:
                     generator = self.pipeline(text, voice=self.default_voice, speed=1.0, split_pattern=r'[.!?]+')
                     for _, _, audio in generator:
                         if audio is not None and audio.size > 0: audio_segments.append(audio)
                except Exception as synth_err: logger.error(f"Kokoro synthesis call error: {synth_err}"); return None
                if not audio_segments: return None
                return np.concatenate(audio_segments)
            combined_audio = await asyncio.get_event_loop().run_in_executor(None, sync_synthesize)
            if combined_audio is not None and combined_audio.size > 0:
                self.synthesis_count += 1
                logger.info(f"TTS Synthesis #{self.synthesis_count}: '{text[:50]}...' -> {combined_audio.shape[0] / AUDIO_SAMPLE_RATE:.2f}s audio")
                return combined_audio
            else: logger.warning(f"TTS synthesis empty audio: '{text[:50]}...'"); return None
        except Exception as e: logger.error(f"Kokoro TTS task error: {e}", exc_info=True); return None
# ---

# --- WebSocket Handler ---
async def handle_client(websocket):
    client_address = websocket.remote_address
    logger.info(f"Client connected: {client_address}")

    detector = AudioSegmentDetector()
    transcriber = WhisperTranscriber.get_instance()
    gemma_processor = GemmaMultimodalProcessor.get_instance() # Use multimodal instance
    tts_processor = KokoroTTSProcessor.get_instance()

    current_speech_processing_task = None

    async def process_speech_segment(segment_bytes):
        """Handles the full pipeline: transcribe -> generate (multimodal) -> TTS."""
        nonlocal current_speech_processing_task
        start_time = time.time()
        try:
            logger.info(f"Processing speech segment of {len(segment_bytes)/2/AUDIO_SAMPLE_RATE:.2f}s")
            transcription = await transcriber.transcribe(segment_bytes)
            if not transcription or not any(c.isalnum() for c in transcription):
                logger.info(f"Skipping empty transcription: '{transcription}'"); await detector.set_tts_playing(False); return

            logger.info(f"User said: '{transcription}' ({(time.time() - start_time):.2f}s)")
            transcription_time = time.time()
            await detector.set_tts_playing(True) # Mark TTS potentially starting

            # Generate response using multimodal processor
            gemma_streamer, initial_text = await gemma_processor.generate_streaming(transcription)
            generation_start_time = time.time(); logger.info(f"Gemma initial generation took: {(generation_start_time - transcription_time):.2f}s")

            if gemma_streamer is None: # Handle generation failure
                logger.error("Gemma generation failed, sending error TTS.")
                if tts_processor:
                    error_audio = await tts_processor.synthesize_speech(initial_text) # Contains error
                    if error_audio is not None:
                        audio_bytes = (error_audio * 32767).astype(np.int16).tobytes()
                        await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
                await detector.set_tts_playing(False); return

            # Synthesize and send initial TTS
            if initial_text and tts_processor:
                tts_start_time = time.time()
                initial_audio = await tts_processor.synthesize_speech(initial_text)
                tts_end_time = time.time()
                if initial_audio is not None:
                    audio_bytes = (initial_audio * 32767).astype(np.int16).tobytes()
                    logger.info(f"Sending initial TTS audio ({len(audio_bytes)/2/AUDIO_SAMPLE_RATE:.2f}s) (TTS took {(tts_end_time - tts_start_time):.2f}s)")
                    await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
                else: logger.warning("Initial TTS synthesis failed.")

            # Consume rest of Gemma stream and synthesize remaining TTS
            remaining_text = ""; full_response = initial_text
            try:
                async for chunk in gemma_streamer: remaining_text += chunk; full_response += chunk
                logger.info(f"Finished consuming Gemma stream. Total length: {len(full_response)}")
            except asyncio.CancelledError: logger.info("Gemma streaming iteration cancelled."); remaining_text = ""
            except Exception as e: logger.error(f"Error iterating Gemma streamer: {e}"); remaining_text = ""

            if remaining_text.strip() and tts_processor:
                tts_remaining_start = time.time()
                remaining_audio = await tts_processor.synthesize_speech(remaining_text.strip())
                tts_remaining_end = time.time()
                if remaining_audio is not None:
                    audio_bytes = (remaining_audio * 32767).astype(np.int16).tobytes()
                    logger.info(f"Sending remaining TTS audio ({len(audio_bytes)/2/AUDIO_SAMPLE_RATE:.2f}s) (TTS took {(tts_remaining_end - tts_remaining_start):.2f}s)")
                    await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
                else: logger.warning("Remaining TTS synthesis failed.")

            # Update history if generation was successful
            if gemma_streamer is not None:
                 async with gemma_processor.lock:
                     gemma_processor._update_history(transcription, full_response.strip())

        except asyncio.CancelledError: logger.info(f"Speech processing task cancelled."); await detector.set_tts_playing(False); raise
        except Exception as e:
            logger.error(f"Error in process_speech_segment: {e}", exc_info=True)
            if tts_processor: # Try sending error TTS
                try:
                    error_audio = await tts_processor.synthesize_speech("Sorry, an error occurred.")
                    if error_audio is not None:
                         audio_bytes = (error_audio * 32767).astype(np.int16).tobytes()
                         await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
                except Exception as send_e: logger.error(f"Failed to send error TTS: {send_e}")
        finally:
            if not asyncio.current_task().cancelled(): await detector.set_tts_playing(False)
            current_speech_processing_task = None; end_time = time.time()
            logger.info(f"Total segment processing time: {(end_time - start_time):.2f}s")

    async def speech_segment_consumer():
        nonlocal current_speech_processing_task
        while True:
            segment = await detector.get_next_segment()
            if segment:
                if current_speech_processing_task and not current_speech_processing_task.done():
                     logger.info("New segment, cancelling previous processing task.")
                     current_speech_processing_task.cancel()
                     try: await current_speech_processing_task
                     except asyncio.CancelledError: logger.info("Previous task cancellation confirmed.")
                     except Exception as e: logger.error(f"Error waiting for previous task cancellation: {e}")
                logger.info("Creating new speech processing task.")
                current_speech_processing_task = asyncio.create_task(process_speech_segment(segment))
                await detector.set_current_tasks(processing_task=current_speech_processing_task)
            await asyncio.sleep(0.01)

    async def receive_data():
        """Handles incoming WebSocket messages (audio AND images)."""
        async for message in websocket:
            try:
                data = json.loads(message)
                tts_active = False
                async with detector.tts_lock: tts_active = detector.tts_playing

                # --- Image Handling Re-added ---
                # Check for standalone image first
                if "image" in data and gemma_processor:
                    if tts_active: logger.warning("Ignoring incoming image because TTS is active.")
                    else:
                        image_data = base64.b64decode(data["image"])
                        if image_data:
                            logger.info("Received standalone image data.")
                            # Setting image will clear history in Gemma processor
                            await gemma_processor.set_image(image_data)
                            # Optional: Cancel ongoing tasks if new image arrives?
                            # await detector.cancel_current_tasks()

                # Check for media chunks (audio or image)
                if "realtime_input" in data and "media_chunks" in data["realtime_input"]:
                     for chunk in data["realtime_input"]["media_chunks"]:
                         mime_type = chunk.get("mime_type")
                         chunk_data = base64.b64decode(chunk.get("data", ""))

                         if mime_type == "audio/pcm":
                             await detector.add_audio(chunk_data) # Feed VAD
                         elif mime_type == "image/jpeg" and gemma_processor:
                             if tts_active: logger.warning("Ignoring incoming image chunk because TTS is active.")
                             elif chunk_data:
                                 logger.info("Received image data chunk.")
                                 await gemma_processor.set_image(chunk_data)
                                 # Optional: Cancel ongoing tasks?
                                 # await detector.cancel_current_tasks()
                         # else: logger.debug(f"Ignoring chunk with mime_type: {mime_type}")
                # --- End of Image Handling ---

            except json.JSONDecodeError: logger.warning(f"Received non-JSON message. Ignoring.")
            except websockets.exceptions.ConnectionClosed: logger.info("Connection closed during receive."); break
            except Exception as e: logger.error(f"Error processing received message: {e}", exc_info=True)

    async def send_keepalive():
        while True:
            try: await websocket.ping(); await asyncio.sleep(20)
            except asyncio.CancelledError: logger.info("Keepalive task cancelled."); break
            except websockets.exceptions.ConnectionClosed: logger.info("Keepalive task: Connection closed."); break
            except Exception as e: logger.error(f"Keepalive task error: {e}"); await asyncio.sleep(20)

    # Main management loop for the connection (same as before)
    receive_task, consumer_task, keepalive_task = None, None, None
    try:
        logger.info(f"Starting tasks for client {client_address}")
        consumer_task = asyncio.create_task(speech_segment_consumer(), name=f"Consumer_{client_address}")
        receive_task = asyncio.create_task(receive_data(), name=f"Receiver_{client_address}")
        keepalive_task = asyncio.create_task(send_keepalive(), name=f"Keepalive_{client_address}")
        tasks = [receive_task, keepalive_task, consumer_task]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try: task.result(); logger.info(f"Task {task.get_name()} finished.")
            except websockets.exceptions.ConnectionClosed: logger.info(f"Task {task.get_name()} finished: Connection Closed.")
            except asyncio.CancelledError: logger.info(f"Task {task.get_name()} cancelled.")
            except Exception as e: logger.error(f"Task {task.get_name()} error: {e}", exc_info=True)
    except websockets.exceptions.ConnectionClosed as e: logger.info(f"Client {client_address} disconnected: {e.code}")
    except Exception as e: logger.error(f"WebSocket handler error: {e}", exc_info=True)
    finally:
        logger.info(f"Cleaning up connection for {client_address}")
        all_tasks = [receive_task, consumer_task, keepalive_task, current_speech_processing_task]
        for task in all_tasks:
            if task and not task.done():
                task.cancel()
                try: await task
                except asyncio.CancelledError: pass
                except Exception as e: logger.error(f"Error during final task cancellation ({task.get_name()}): {e}")
        logger.info(f"Finished cleanup for {client_address}")
# ---

# --- Main Application Setup ---
async def main():
    logger.info("Pre-initializing models...")
    try:
        WhisperTranscriber.get_instance()
        GemmaMultimodalProcessor.get_instance() # Get multimodal instance
        KokoroTTSProcessor.get_instance()
        logger.info("Models initialized.")
    except Exception as e:
        logger.error(f"Fatal error during model initialization: {e}", exc_info=True); sys.exit(1)

    logger.info(f"Starting WebSocket server on 0.0.0.0:{WEBSOCKET_PORT}")
    async with websockets.serve(handle_client, "0.0.0.0", WEBSOCKET_PORT,
                                 ping_interval=20, ping_timeout=120, close_timeout=10):
        await asyncio.Future() # Run forever

if __name__ == "__main__":
    print("-" * 30)
    print("Starting Real-time ASR + LLM (Multimodal) + TTS Server")
    print(f" - Whisper Model: {WHISPER_MODEL_ID}")
    print(f" - Gemma Model:   {GEMMA_MODEL_ID} (Multimodal)") # Updated
    print(f" - TTS Enabled:   {KOKORO_AVAILABLE}")
    print(f" - Flash Attn:    {FLASH_ATTN_AVAILABLE} ({ATTN_IMPLEMENTATION})")
    print(f" - DType:         {TORCH_DTYPE}")
    print(f" - WebSocket Port:{WEBSOCKET_PORT}")
    print("-" * 30)
    print("Ensure required libraries are installed:")
    print("  pip install transformers torch accelerate websockets numpy Pillow") # Added Pillow
    print("Optional (highly recommended for speed):")
    print("  pip install flash-attn --no-build-isolation")
    print("Optional (for VRAM saving - requires code changes to enable):")
    print("  pip install bitsandbytes")
    print("Optional (for TTS):")
    print("  pip install kokoro") # Replace with actual Kokoro install command
    print("-" * 30)
    asyncio.run(main())
# --- End of Script ---
