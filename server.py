import asyncio
import json
import websockets
import base64
import torch
# Ensure specific imports are present
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    Gemma3ForConditionalGeneration, # Corrected potentially missing import
    TextIteratorStreamer,
    BitsAndBytesConfig
)
import numpy as np
import logging
import sys
import io
from PIL import Image
import time
import os
from datetime import datetime
from kokoro import KPipeline # Import Kokoro TTS library
import re
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
# Reduce verbosity of less critical libraries
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
# Keep transformers logging at INFO for now to see loading messages
# logging.getLogger("transformers").setLevel(logging.WARNING)
# logging.getLogger("PIL").setLevel(logging.INFO)


logger = logging.getLogger(__name__)

# --- Global Thread Pool Executor for CPU-bound tasks ---
# To avoid blocking the asyncio event loop with synchronous library calls
# Use slightly fewer workers than cores potentially to leave room for GPU etc.
executor_workers = max(1, (os.cpu_count() or 4) - 1)
executor = ThreadPoolExecutor(max_workers=executor_workers)
logger.info(f"ThreadPoolExecutor initialized with {executor_workers} workers.")


# --- Singleton Pattern for Models ---
# Ensures models are loaded only once
class SingletonMeta(type):
    _instances = {}
    _lock = asyncio.Lock() # Use asyncio lock for async context if needed later

    async def get_instance(cls, *args, **kwargs):
        # Use an async lock to prevent race conditions during first creation
        # even though initialization itself might be synchronous now.
        async with cls._lock:
            if cls not in cls._instances:
                logger.info(f"Creating new instance of {cls.__name__}")
                # Run potentially blocking __init__ in executor
                loop = asyncio.get_running_loop()
                instance = await loop.run_in_executor(
                    executor,
                    cls._create_instance_sync, # Call a sync wrapper
                    *args, **kwargs
                )
                cls._instances[cls] = instance
                logger.info(f"Instance of {cls.__name__} created.")
            return cls._instances[cls]

    def _create_instance_sync(cls, *args, **kwargs):
        # This synchronous method is run in the executor
        return super(SingletonMeta, cls).__call__(*args, **kwargs)

# --- Audio Segment Detector ---
class AudioSegmentDetector:
    """Detects speech segments based on audio energy levels."""
    # No asyncio loop needed in __init__
    def __init__(self,
                 sample_rate=16000,
                 energy_threshold=0.015,
                 silence_duration=0.7,
                 min_speech_duration=0.5,
                 max_speech_duration=15):

        self.sample_rate = sample_rate
        self.bytes_per_sample = 2 # 16-bit PCM
        self.energy_threshold = energy_threshold
        self.silence_samples = int(silence_duration * sample_rate)
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.max_speech_samples = int(max_speech_duration * sample_rate)

        self.audio_buffer = bytearray()
        self.is_speech_active = False
        self.silence_counter = 0
        self.speech_start_idx = 0
        self.segment_queue = asyncio.Queue()
        self.segments_detected_count = 0
        # REMOVED: self._loop = asyncio.get_event_loop() # Error-prone in __init__

        logger.info(f"AudioSegmentDetector params: "
                    f"Threshold={energy_threshold}, Silence={silence_duration}s, "
                    f"MinSpeech={min_speech_duration}s, MaxSpeech={max_speech_duration}s")

    async def _calculate_energy(self, audio_bytes):
        """Helper to calculate energy asynchronously."""
        if not audio_bytes:
            return 0.0
        # Get the loop *when needed*
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            self._sync_calculate_energy,
            audio_bytes
        )

    def _sync_calculate_energy(self, audio_bytes):
        """Synchronous energy calculation."""
        # ... (rest of the method remains the same)
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_array) == 0:
                return 0.0
            energy = np.sqrt(np.mean(audio_array**2))
            return energy
        except Exception as e:
            logger.error(f"Error calculating energy: {e}", exc_info=True)
            return 0.0

    async def add_audio(self, audio_bytes, is_tts_playing_func, cancel_tasks_func):
        """Add audio data, detect segments, and trigger interrupt if needed."""
        # ... (rest of the method remains largely the same)
        self.audio_buffer.extend(audio_bytes)
        current_buffer_len_samples = len(self.audio_buffer) // self.bytes_per_sample

        # Analyze only the newly added chunk for energy
        energy = await self._calculate_energy(audio_bytes)
        new_samples_count = len(audio_bytes) // self.bytes_per_sample

        if not self.is_speech_active:
            if energy > self.energy_threshold:
                # --- Speech Start ---
                self.is_speech_active = True
                self.speech_start_idx = max(0, len(self.audio_buffer) - len(audio_bytes))
                self.silence_counter = 0
                logger.info(f"Speech start detected (Energy: {energy:.4f})")

                # --- INTERRUPT LOGIC ---
                if await is_tts_playing_func():
                    logger.warning("New speech detected during TTS playback. Sending interrupt.")
                    await cancel_tasks_func() # Cancel ongoing TTS/Generation

        elif self.is_speech_active:
            if energy > self.energy_threshold:
                # --- Continued Speech ---
                self.silence_counter = 0
            else:
                # --- Potential Silence ---
                self.silence_counter += new_samples_count

                # Check for end of speech (silence duration met)
                if self.silence_counter >= self.silence_samples:
                    speech_end_idx = len(self.audio_buffer) - (self.silence_counter * self.bytes_per_sample)
                    # Ensure start index is not past end index
                    speech_start_idx_safe = min(self.speech_start_idx, speech_end_idx)
                    speech_segment_bytes = bytes(self.audio_buffer[speech_start_idx_safe:speech_end_idx])
                    segment_len_samples = len(speech_segment_bytes) // self.bytes_per_sample

                    # Reset buffer and state
                    self.audio_buffer = self.audio_buffer[speech_end_idx:]
                    self.is_speech_active = False
                    self.silence_counter = 0
                    self.speech_start_idx = 0 # Reset start index for new buffer

                    if segment_len_samples >= self.min_speech_samples:
                        self.segments_detected_count += 1
                        logger.info(f"Speech segment [Silence End] detected ({segment_len_samples / self.sample_rate:.2f}s). Queueing.")
                        await self.segment_queue.put(speech_segment_bytes)
                    else:
                         logger.info(f"Speech segment below min duration ({segment_len_samples / self.sample_rate:.2f}s). Discarding.")


            # Check for max speech duration limit
            current_speech_len_bytes = (len(self.audio_buffer) - self.speech_start_idx)
            if self.is_speech_active and current_speech_len_bytes >= self.max_speech_samples * self.bytes_per_sample:
                logger.warning(f"Max speech duration ({self.max_speech_duration}s) exceeded.")
                # Take exactly max_speech_samples worth of bytes from speech_start_idx
                speech_end_idx = self.speech_start_idx + (self.max_speech_samples * self.bytes_per_sample)
                speech_segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])

                # Keep the buffer after the max duration point
                self.audio_buffer = self.audio_buffer[speech_end_idx:]
                # We are still potentially in speech, just forced a segment break
                # Reset speech start index relative to the *new* buffer start
                self.speech_start_idx = 0
                self.silence_counter = 0 # Reset silence counter as we forced a break

                segment_len_samples = len(speech_segment_bytes) // self.bytes_per_sample
                if segment_len_samples >= self.min_speech_samples: # Check min duration again
                    self.segments_detected_count += 1
                    logger.info(f"Speech segment [Max Duration] detected ({segment_len_samples / self.sample_rate:.2f}s). Queueing.")
                    await self.segment_queue.put(speech_segment_bytes)

                    # --- INTERRUPT LOGIC ---
                    if await is_tts_playing_func():
                         logger.warning("Max speech duration hit during TTS playback. Sending interrupt.")
                         await cancel_tasks_func()
                else:
                    logger.info(f"Max duration segment too short after cut ({segment_len_samples / self.sample_rate:.2f}s). Discarding.")


        # Keep buffer size manageable
        max_buffer_samples = (self.max_speech_samples + self.silence_samples)
        max_buffer_bytes = max_buffer_samples * self.bytes_per_sample
        if len(self.audio_buffer) > max_buffer_bytes:
            keep_bytes = max_buffer_bytes
            discard_bytes = len(self.audio_buffer) - keep_bytes
            self.audio_buffer = self.audio_buffer[discard_bytes:]
            # Adjust speech_start_idx if it was within the discarded part
            self.speech_start_idx = max(0, self.speech_start_idx - discard_bytes)


    async def get_next_segment(self, timeout=0.1):
        """Get the next available speech segment."""
        try:
            return await asyncio.wait_for(self.segment_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

# --- Whisper Transcriber ---
class WhisperTranscriber(metaclass=SingletonMeta):
    """Handles speech transcription using Whisper."""
    def __init__(self):
        # This __init__ is now guaranteed to run in an executor thread
        # DO NOT call asyncio event loop functions here
        logger.info("Initializing WhisperTranscriber (sync part)...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda:0" else torch.float32
        model_id = "openai/whisper-large-v3"

        logger.info(f"Loading Whisper model: {model_id} on {self.device} with {self.torch_dtype}")
        try:
            # These are synchronous HFace calls, okay in executor
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_id)
            # Pipeline creation is also synchronous
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                # Pass generate_kwargs here instead of max_new_tokens in pipeline init
                # max_new_tokens=128, # Deprecated here
                chunk_length_s=20,
                batch_size=8,
                return_timestamps=False,
                torch_dtype=self.torch_dtype,
                device=self.device,
                generate_kwargs={"language": "english", "task": "transcribe", "temperature": 0.0, "max_new_tokens": 128} # Correct place
            )
            logger.info("WhisperTranscriber sync initialization complete.")
            self.transcription_count = 0
            # REMOVED: self._loop = asyncio.get_event_loop() # Error!
        except Exception as e:
            logger.error(f"Error during sync Whisper initialization: {e}", exc_info=True)
            raise

    async def transcribe(self, audio_bytes, sample_rate=16000):
        """Transcribe audio bytes to text asynchronously."""
        if len(audio_bytes) < sample_rate * 0.2 * 2 :
            logger.warning(f"Audio segment too short ({len(audio_bytes)} bytes), skipping.")
            return ""
        start_time = time.time()
        # Get loop when needed
        loop = asyncio.get_running_loop()
        try:
            # Prepare audio data (run in executor to avoid blocking)
            audio_input = await loop.run_in_executor(
                executor, self._prepare_audio, audio_bytes, sample_rate
            )
            if audio_input is None: return ""

            # Run pipeline in executor
            # The lambda captures self.pipe correctly
            result = await loop.run_in_executor(
                executor,
                lambda: self.pipe(audio_input) # Pass generate_kwargs in pipeline init now
            )

            text = result.get("text", "").strip()
            duration = time.time() - start_time
            self.transcription_count += 1
            logger.info(f"Transcription #{self.transcription_count} ('{text}') took {duration:.3f}s")
            return text
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Transcription error after {duration:.3f}s: {e}", exc_info=True)
            return ""

    def _prepare_audio(self, audio_bytes, sample_rate):
        """Synchronous audio preparation."""
        # ... (no changes needed)
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return {"array": audio_array, "sampling_rate": sample_rate}
        except Exception as e:
            logger.error(f"Error preparing audio: {e}", exc_info=True)
            return None

# --- Gemma Multimodal Processor ---
class GemmaMultimodalProcessor(metaclass=SingletonMeta):
    """Handles multimodal generation using Gemma."""
    def __init__(self):
        # Sync init in executor
        logger.info("Initializing GemmaMultimodalProcessor (sync part)...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_id = "google/gemma-3-4b-it"
        quantization_config = None
        torch_dtype = torch.bfloat16

        if torch.cuda.is_available():
             quantization_config = BitsAndBytesConfig(load_in_8bit=True)
             logger.info("Using 8-bit quantization for Gemma.")
        else:
             logger.info("Using bfloat16 for Gemma on CPU (quantization disabled).")

        logger.info(f"Loading Gemma model: {model_id} on {self.device}")
        try:
            # Synchronous calls
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                 trust_remote_code=True,
                 # Avoid device_map='auto' with streaming if possible
                 # Explicitly move later if needed
            )
             # Manually move model to device if not quantized (quantized uses device_map implicitly)
            if not quantization_config and self.device == "cuda:0":
                 self.model.to(self.device)

            self.processor = AutoProcessor.from_pretrained(model_id)
            self.tokenizer = self.processor.tokenizer
            logger.info("GemmaMultimodalProcessor sync initialization complete.")

            self.last_image = None
            self.last_image_timestamp = 0
            self.message_history = []
            self.max_history_turns = 3
            self.generation_count = 0
            # REMOVED: self._loop = asyncio.get_event_loop() # Error!
        except Exception as e:
            logger.error(f"Error during sync Gemma initialization: {e}", exc_info=True)
            raise

    async def set_image(self, image_data):
        """Cache the most recent image received."""
        loop = asyncio.get_running_loop() # Get loop when needed
        try:
            image = await loop.run_in_executor(
                executor, self._process_image, image_data
            )
            if image:
                self.last_image = image
                self.last_image_timestamp = time.time()
                self.message_history = []
                logger.info(f"New image set (size {image.size}). History cleared.")
                return True
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
        return False

    def _process_image(self, image_data):
        """Synchronous image processing (resizing)."""
        # ... (no changes needed)
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return image
        except Exception as e:
            logger.error(f"Pillow error processing image: {e}", exc_info=True)
            return None

    def _build_prompt(self, text):
        """Builds the prompt string including history for the model."""
        # ... (no changes needed)
        system_prompt = """You are a helpful, conversational AI assistant. You are looking at an image provided by the user. Respond concisely and naturally based on the user's spoken request. Keep responses suitable for text-to-speech (1-3 sentences usually). If the request is clearly about the image, describe what you see relevant to the request. If the request is conversational or unclear, respond naturally without forcing image description. Acknowledge the conversation history implicitly."""
        history = "\n".join([f"User: {turn['user']}\nAssistant: {turn['assistant']}" for turn in self.message_history])
        full_prompt = f"{system_prompt}\n\n{history}\n\nUser: {text}\nAssistant:"
        return full_prompt

    def _update_history(self, user_text, assistant_response):
        """Update message history."""
        # ... (no changes needed)
        self.message_history.append({"user": user_text, "assistant": assistant_response})
        if len(self.message_history) > self.max_history_turns:
            self.message_history = self.message_history[-self.max_history_turns:]

    async def generate_response_stream(self, text):
        """Generate response with streaming. Yields text chunks."""
        if self.last_image is None:
            logger.warning("Cannot generate response, no image available.")
            yield "I don't have an image context right now. Could you provide one?"
            return

        start_time = time.time()
        prompt = self._build_prompt(text)
        loop = asyncio.get_running_loop() # Get loop when needed

        try:
            # Input processing might involve CPU work, run in executor
            inputs = await loop.run_in_executor(
                 executor,
                 lambda: self.processor(text=prompt, images=self.last_image, return_tensors="pt")#.to(self.model.device, dtype=self.model.dtype) # Move inside generate maybe?
            )
            # Move inputs to device just before generation
            inputs = inputs.to(self.model.device)

        except Exception as e:
            logger.error(f"Error processing Gemma inputs: {e}", exc_info=True)
            yield "Sorry, I had trouble processing that request with the image."
            return

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            inputs, # Unpack inputs dictionary
            streamer=streamer,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

        # Run generation in a separate thread - model.generate is blocking
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()

        generated_text = ""
        token_count = 0
        first_token_time = None

        try:
            # Iterate through the streamer in the main async thread
            for new_text in streamer:
                if not first_token_time:
                    first_token_time = time.time()
                yield new_text
                generated_text += new_text
                # Simple word count approx token count
                token_count += len(new_text.split())
            # Wait for the generation thread to finish *after* streamer is exhausted
            # Use join with a timeout? Let's assume it finishes if streamer is done.
            # thread.join() # Might block if generate hangs

            duration = time.time() - start_time
            ttft = (first_token_time - start_time) if first_token_time else duration
            self.generation_count += 1
            logger.info(f"Gemma Stream #{self.generation_count}: TTFT={ttft:.3f}s, Total={duration:.3f}s, Tokens~={token_count}")
            # Update history after full generation
            self._update_history(text, generated_text.strip()) # Ensure no trailing spaces

        except Exception as e:
             duration = time.time() - start_time
             logger.error(f"Gemma streaming generation failed after {duration:.3f}s: {e}", exc_info=True)
             if first_token_time:
                 yield " I encountered an error while generating the rest of the response."
             # Ensure thread is joinable/finished? Difficult if it hangs. Daemon=True helps exit.
             # thread.join(timeout=1.0) # Add timeout to join

# --- Kokoro TTS Processor ---

class KokoroTTSProcessor(metaclass=SingletonMeta):
    """Handles text-to-speech conversion using Kokoro."""
    def __init__(self):
        # Sync init in executor
        logger.info("Initializing KokoroTTSProcessor (sync part)...")
        try:
            # Synchronous call
            # --- FIX: Use the correct language code ---
            # self.pipeline = KPipeline(lang_code='en') # INCORRECT
            self.pipeline = KPipeline(lang_code='a')  # CORRECT for American English
            # --- END FIX ---

            # Ensure voice matches the selected language code 'a'
            self.default_voice = 'en_us_sarah' # This seems like a valid US voice name
            self.sample_rate = 24000 # Kokoro default
            logger.info(f"KokoroTTSProcessor sync initialization complete with lang_code='a', voice='{self.default_voice}'.")
            self.synthesis_count = 0
        except Exception as e:
            logger.error(f"Error during sync Kokoro TTS initialization: {e}", exc_info=True)
            self.pipeline = None # Ensure pipeline is None on error
            raise

    async def synthesize_speech_stream(self, text_iterator):
        """Synthesize speech sentence by sentence from a text iterator."""
        if not self.pipeline:
            logger.error("Kokoro TTS pipeline not available.")
            return # Stop iteration

        sentence_buffer = ""
        # Regex to split after sentence-ending punctuation followed by space or end-of-string
        sentence_end_pattern = re.compile(r'(?<=[.!?])(\s+|$)')

        try:
            async for text_chunk in text_iterator:
                sentence_buffer += text_chunk
                parts = sentence_end_pattern.split(sentence_buffer)

                # Process complete sentences found
                processed_len = 0
                # We expect parts like ['Sentence 1.', ' ', 'Sentence 2?', ' ', 'Partial sentence']
                # or ['Sentence 1.', '\n', 'Sentence 2!', '', ''] if ends with punctuation
                for i in range(0, len(parts) - 1, 2): # Step by 2 to get sentence + delimiter
                    sentence = (parts[i] + (parts[i+1] or '')).strip() # Re-add delimiter, strip spaces
                    if sentence:
                        start_time = time.time()
                        # Synthesize this sentence
                        audio_data = await self.synthesize_single_unit(sentence)
                        duration = time.time() - start_time
                        if audio_data is not None:
                            self.synthesis_count += 1
                            logger.info(f"TTS Stream #{self.synthesis_count}: Synth ('{sentence[:30]}...') took {duration:.3f}s")
                            yield audio_data
                        else:
                            logger.warning(f"TTS failed for unit: '{sentence[:30]}...'")
                        # Track how much of the buffer we processed
                        processed_len += len(parts[i]) + len(parts[i+1] or '')

                # Update buffer with remaining part
                sentence_buffer = sentence_buffer[processed_len:]
                # Yield control briefly
                await asyncio.sleep(0.01)

            # Process any remaining text in the buffer after the iterator finishes
            final_unit = sentence_buffer.strip()
            if final_unit:
                start_time = time.time()
                audio_data = await self.synthesize_single_unit(final_unit)
                duration = time.time() - start_time
                if audio_data is not None:
                    self.synthesis_count += 1
                    logger.info(f"TTS Stream #{self.synthesis_count}: Synth final ('{final_unit[:30]}...') took {duration:.3f}s")
                    yield audio_data
                else:
                    logger.warning(f"TTS failed for final unit: '{final_unit[:30]}...'")

        except asyncio.CancelledError:
            logger.info("TTS Synthesis stream cancelled.")
            raise # Propagate cancellation
        except Exception as e:
            logger.error(f"Kokoro TTS streaming synthesis error: {e}", exc_info=True)


    async def synthesize_single_unit(self, text_unit):
        """Synthesize a single unit of text (e.g., sentence)."""
        if not self.pipeline or not text_unit:
            return None
        loop = asyncio.get_running_loop() # Get loop when needed
        try:
            # Run synchronous Kokoro call in executor
            combined_audio = await loop.run_in_executor(
                 executor, self._sync_synthesize, text_unit
             )
            return combined_audio
        except Exception as e:
            logger.error(f"Kokoro synth unit failed for '{text_unit[:50]}...': {e}", exc_info=False)
            return None

    def _sync_synthesize(self, text):
        """Synchronous synthesis for executor."""
        # ... (no changes needed)
        audio_segments = []
        try:
            generator = self.pipeline(
                text,
                voice=self.default_voice,
                speed=1.0,
                split_pattern=None # Process as one chunk if possible
            )
            for _, _, audio in generator:
                if audio is not None and len(audio) > 0:
                    audio_segments.append(audio)

            if audio_segments:
                 combined = np.concatenate(audio_segments).astype(np.float32)
                 return combined
            else:
                 logger.warning(f"Kokoro returned no audio for text: '{text[:50]}...'")
                 return None
        except Exception as e:
             logger.error(f"Kokoro _sync_synthesize error for text '{text[:50]}...': {e}", exc_info=False)
             return None


# --- WebSocket Handler ---
class ClientHandler:
    def __init__(self, websocket, whisper, gemma, tts):
        self.websocket = websocket
        # Use pre-initialized singleton instances passed in
        self.detector = AudioSegmentDetector() # Detector is per-client stateful
        self.transcriber = whisper
        self.gemma_processor = gemma
        self.tts_processor = tts

        self._tts_playing = False
        self._tts_lock = asyncio.Lock()
        self._active_tasks = set()
        # Get loop when needed: self._loop = asyncio.get_running_loop()
        self.client_id = f"Client-{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"[{self.client_id}] Handler initialized.")

    async def _is_tts_playing(self):
        async with self._tts_lock:
            return self._tts_playing

    async def _set_tts_playing(self, playing):
        async with self._tts_lock:
            if self._tts_playing != playing:
                 logger.info(f"[{self.client_id}] Setting TTS playing state to: {playing}")
                 self._tts_playing = playing

    async def _cancel_active_tasks(self):
        # ... (rest of the method is okay, uses asyncio functions directly)
        if not self._active_tasks:
            return

        logger.warning(f"[{self.client_id}] Cancelling {len(self._active_tasks)} active task(s)...")
        try:
            interrupt_message = json.dumps({"interrupt": True})
            logger.info(f"[{self.client_id}] Sending interrupt message to client.")
            await self.websocket.send(interrupt_message)
        except websockets.exceptions.ConnectionClosed:
             logger.warning(f"[{self.client_id}] Connection closed while sending interrupt message.")
             self._active_tasks.clear()
             await self._set_tts_playing(False)
             return
        except Exception as e:
             logger.error(f"[{self.client_id}] Error sending interrupt message: {e}", exc_info=True)

        tasks_to_cancel = list(self._active_tasks)
        self._active_tasks.clear()

        cancelled_count = 0
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
                cancelled_count += 1
                # Give event loop a chance to process cancellation
                await asyncio.sleep(0)

        if cancelled_count > 0:
            logger.info(f"[{self.client_id}] {cancelled_count} active tasks cancellation requested.")
        else:
             logger.info(f"[{self.client_id}] No running tasks needed cancellation.")

        # Ensure TTS state is false after cancellation attempt
        await self._set_tts_playing(False)


    async def _process_segment(self, speech_segment):
        """Process a single detected speech segment."""
        # ... (rest of the method logic remains the same, uses asyncio directly)
        # 1. Transcribe
        transcription = await self.transcriber.transcribe(speech_segment)
        if not transcription or not any(c.isalnum() for c in transcription):
            logger.info(f"[{self.client_id}] Empty/non-alpha transcription: '{transcription}'. Skipping.")
            return

        # Filter common fillers
        filler_patterns = [
             r'^(um+|uh+|ah+|oh+|hm+|mhm+|hmm+)$',
             r'^(okay|yes|no|yeah|nah)$',
             r'^bye+$',
             r'^(thank you|thanks)$'
         ]
        if any(re.fullmatch(pattern, transcription.lower()) for pattern in filler_patterns):
             logger.info(f"[{self.client_id}] Skipping filler phrase: '{transcription}'")
             return

        logger.info(f"[{self.client_id}] User said: '{transcription}'")

        # Ensure previous tasks are cancelled
        await self._cancel_active_tasks()
        await self._set_tts_playing(True) # Mark TTS as active

        # Get loop for task creation
        loop = asyncio.get_running_loop()

        # Create tasks and add them to the active set
        gen_task = None
        synth_task = None
        playback_task = None
        tasks_created = False

        try:
            text_queue = asyncio.Queue()
            gen_task = loop.create_task(self._run_gemma_stream(transcription, text_queue))
            self._active_tasks.add(gen_task)

            audio_queue = asyncio.Queue()
            synth_task = loop.create_task(self._run_tts_stream(text_queue, audio_queue))
            self._active_tasks.add(synth_task)

            playback_task = loop.create_task(self._run_playback_stream(audio_queue))
            self._active_tasks.add(playback_task)
            tasks_created = True

            # Wait for all tasks in this chain to complete naturally
            await asyncio.gather(gen_task, synth_task, playback_task)
            logger.info(f"[{self.client_id}] Full response cycle completed for: '{transcription}'")

        except asyncio.CancelledError:
            logger.info(f"[{self.client_id}] Response cycle cancelled for: '{transcription}'")
            # Cancellation should be handled by _cancel_active_tasks called externally
            # Redundant check/removal below is okay

        except Exception as e:
            logger.error(f"[{self.client_id}] Error during response cycle for '{transcription}': {e}", exc_info=True)
            # Attempt to cancel any remaining tasks from this cycle
            await self._cancel_active_tasks() # Ensure cleanup on error

        finally:
            # --- Cleanup for this segment ---
            # Ensure TTS state is reset *if no cancellation happened*
            # _cancel_active_tasks already sets tts_playing to False
            # Check if the task set still contains these tasks (meaning they weren't cancelled by an interrupt)
            # and reset TTS state if they completed normally or errored out here.
            tasks_were_present = {gen_task, synth_task, playback_task}.issubset(self._active_tasks)

            if tasks_created: # Only try removal if tasks were added
                 self._active_tasks.discard(gen_task)
                 self._active_tasks.discard(synth_task)
                 self._active_tasks.discard(playback_task)

            # Only set TTS to false if tasks finished here (not interrupted)
            # This check might be flawed if cancellation happens between gather finishing and here.
            # Relying on _cancel_active_tasks setting it false is safer. Let's remove this redundant set.
            # if tasks_were_present and await self._is_tts_playing():
            #     await self._set_tts_playing(False)
            pass # Let _cancel_active_tasks manage the flag exclusively.


    async def _run_gemma_stream(self, text, output_queue):
        """Task to run Gemma generation and put text chunks into a queue."""
        # ... (rest of the method is okay, uses asyncio directly)
        try:
            async for chunk in self.gemma_processor.generate_response_stream(text):
                await output_queue.put(chunk)
                await asyncio.sleep(0) # Yield control
        except asyncio.CancelledError:
             logger.info(f"[{self.client_id}] Gemma generation task cancelled.")
             await output_queue.put(None) # Signal end/cancel downstream
             raise
        except Exception as e:
             logger.error(f"[{self.client_id}] Error in Gemma generation task: {e}", exc_info=True)
             await output_queue.put(None) # Signal error downstream
        finally:
             await output_queue.put(None) # Signal end of stream reliably

    async def _run_tts_stream(self, input_queue, output_queue):
        """Task to run TTS synthesis based on text chunks from a queue."""
        # ... (rest of the method is okay, uses asyncio directly)
        async def text_iterator():
            while True:
                chunk = await input_queue.get()
                if chunk is None: # End of stream signal
                    input_queue.task_done() # Mark sentinel as processed
                    break
                yield chunk
                input_queue.task_done() # Mark item as processed
        try:
            async for audio_chunk in self.tts_processor.synthesize_speech_stream(text_iterator()):
                if audio_chunk is not None:
                    await output_queue.put(audio_chunk)
                await asyncio.sleep(0) # Yield control
        except asyncio.CancelledError:
            logger.info(f"[{self.client_id}] TTS synthesis task cancelled.")
            await output_queue.put(None) # Signal end/cancel downstream
             # Ensure the input queue is drained if synthesis ends early/errors
            while not input_queue.empty():
                 item = await input_queue.get()
                 input_queue.task_done()
                 if item is None: break # Stop if sentinel found
            raise # Propagate cancellation
        except Exception as e:
            logger.error(f"[{self.client_id}] Error in TTS synthesis task: {e}", exc_info=True)
            await output_queue.put(None) # Signal error downstream
        finally:
             # Ensure the input queue is drained if synthesis ends early/errors
             while not input_queue.empty():
                 item = await input_queue.get()
                 input_queue.task_done()
                 if item is None: break # Stop if sentinel found
             await output_queue.put(None) # Signal end of audio stream reliably

    async def _run_playback_stream(self, input_queue):
        """Task to send synthesized audio chunks to the client."""
        # ... (rest of the method is okay, uses asyncio, np, base64 directly or via executor)
        stream_start_time = time.time()
        audio_sent_samples = 0
        loop = asyncio.get_running_loop() # Get loop for executor call
        try:
            while True:
                audio_chunk = await input_queue.get()
                if audio_chunk is None: # End of stream signal
                    input_queue.task_done() # Mark sentinel as processed
                    break

                # Convert numpy float32 audio to int16 bytes -> base64
                try:
                    if audio_chunk.dtype != np.float32: audio_chunk = audio_chunk.astype(np.float32)
                    np.clip(audio_chunk, -1.0, 1.0, out=audio_chunk)
                    audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                    # Run base64 encoding in executor as it can be CPU intensive for large chunks
                    base64_audio = await loop.run_in_executor(executor, base64.b64encode, audio_bytes)
                    base64_audio_str = base64_audio.decode('utf-8')

                    # Send to client
                    await self.websocket.send(json.dumps({
                        "audio": base64_audio_str,
                        "sample_rate": self.tts_processor.sample_rate
                    }))
                    audio_sent_samples += len(audio_chunk)
                except websockets.exceptions.ConnectionClosed:
                    logger.warning(f"[{self.client_id}] Connection closed during audio playback.")
                    await self._cancel_active_tasks() # Trigger cancellation upwards
                    # Drain queue after signalling cancellation
                    while not input_queue.empty():
                        item = await input_queue.get()
                        input_queue.task_done()
                        if item is None: break
                    break # Exit playback loop
                except Exception as e:
                    logger.error(f"[{self.client_id}] Error preparing/sending audio chunk: {e}", exc_info=True)
                    # Continue trying to send subsequent chunks? Or break? Let's continue for now.

                input_queue.task_done() # Mark item as processed
                await asyncio.sleep(0.01) # Small sleep to yield control

        except asyncio.CancelledError:
            logger.info(f"[{self.client_id}] Audio playback task cancelled.")
             # Drain queue on cancellation
            while not input_queue.empty():
                 item = await input_queue.get()
                 input_queue.task_done()
                 if item is None: break
            # Don't raise here, cancellation originated elsewhere
        except Exception as e:
             logger.error(f"[{self.client_id}] Error in playback task: {e}", exc_info=True)
        finally:
             # Drain queue on exit/error
             while not input_queue.empty():
                 item = await input_queue.get()
                 input_queue.task_done()
                 if item is None: break
             duration = time.time() - stream_start_time
             if audio_sent_samples > 0:
                 logger.info(f"[{self.client_id}] Audio playback finished. Sent {audio_sent_samples / self.tts_processor.sample_rate:.2f}s audio in {duration:.2f}s.")
             # Playback finished, TTS is no longer playing *from this cycle*
             # Rely on _cancel_active_tasks or natural completion flow to set flag.
             # await self._set_tts_playing(False) # Remove redundant flag set


    # --- Main Client Interaction Loops ---

    async def handle_connection(self):
        """Main handler for a single client connection."""
        # ... (rest of the method is okay, uses asyncio directly)
        logger.info(f"[{self.client_id}] Client connected.")
        loop = asyncio.get_running_loop() # Get loop for task creation
        try:
            await self.websocket.send(json.dumps({"status": "connected", "server_id": "GemmaKokoroV2"}))

            receive_task = loop.create_task(self.receive_messages())
            process_task = loop.create_task(self.process_speech_queue())
            keepalive_task = loop.create_task(self.send_keepalive())

            # Keep track of tasks specific to this handler's core loops
            handler_tasks = {receive_task, process_task, keepalive_task}

            done, pending = await asyncio.wait(
                handler_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Log which task finished first
            for task in done:
                 try:
                     await task # Check for exceptions in the completed task
                     logger.info(f"[{self.client_id}] Task {task.get_name()} completed normally.")
                 except Exception as e:
                     logger.error(f"[{self.client_id}] Task {task.get_name()} failed: {e}", exc_info=True)


            # If one task finishes (e.g., receive due to disconnect), cancel others
            logger.info(f"[{self.client_id}] One handler task finished, cancelling others ({len(pending)} pending).")
            for task in pending:
                if task and not task.done():
                     task.cancel()
            # Wait for pending tasks to finish cancellation
            if pending:
                 await asyncio.wait(pending)


        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"[{self.client_id}] Client disconnected gracefully.")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"[{self.client_id}] Client disconnected with error: {e}")
        except asyncio.CancelledError:
             logger.info(f"[{self.client_id}] Handler task cancelled.")
        except Exception as e:
            logger.error(f"[{self.client_id}] Unhandled error in client handler: {e}", exc_info=True)
        finally:
            logger.info(f"[{self.client_id}] Cleaning up handler resources...")
            # Ensure all *active* response tasks are cancelled on exit
            await self._cancel_active_tasks()
            # Also ensure the main handler tasks are cancelled if not already
            if 'handler_tasks' in locals():
                for task in handler_tasks:
                    if task and not task.done():
                        task.cancel()
            logger.info(f"[{self.client_id}] Client handler finished cleanup.")


    async def receive_messages(self):
        """Task to receive messages from the WebSocket client."""
        # ... (rest of the method is okay, uses asyncio directly)
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)

                    # Handle audio data chunks
                    if "audio_chunk" in data:
                        audio_data = base64.b64decode(data["audio_chunk"])
                        await self.detector.add_audio(
                            audio_data,
                            self._is_tts_playing, # Pass async function
                            self._cancel_active_tasks # Pass async function
                        )

                    # Handle image data
                    elif "image_chunk" in data:
                         image_data = base64.b64decode(data["image_chunk"])
                         await self.gemma_processor.set_image(image_data)
                         logger.info(f"[{self.client_id}] Received image chunk.")

                    # Handle client signals
                    elif "signal" in data:
                        if data["signal"] == "stop_tts":
                             logger.info(f"[{self.client_id}] Received 'stop_tts' signal.")
                             await self._cancel_active_tasks()

                except json.JSONDecodeError:
                    logger.warning(f"[{self.client_id}] Received invalid JSON: {message[:100]}...")
                except asyncio.CancelledError:
                    logger.info(f"[{self.client_id}] Receive task cancelled.")
                    break
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"[{self.client_id}] Connection closed during message receive.")
                    break
                except Exception as e:
                    logger.error(f"[{self.client_id}] Error processing received message: {e}", exc_info=True)
        finally:
            logger.info(f"[{self.client_id}] Receive messages task finished.")


    async def process_speech_queue(self):
        """Task to process detected speech segments from the queue."""
        # ... (rest of the method is okay, uses asyncio directly)
        try:
            while True:
                segment = await self.detector.get_next_segment(timeout=1.0)
                if segment:
                    # Process the segment
                    await self._process_segment(segment)

                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            logger.info(f"[{self.client_id}] Process speech queue task cancelled.")
        except Exception as e:
             logger.error(f"[{self.client_id}] Error in speech processing loop: {e}", exc_info=True)
        finally:
            logger.info(f"[{self.client_id}] Process speech queue task finished.")


    async def send_keepalive(self):
        """Task to send periodic pings to keep the connection alive."""
        # ... (rest of the method is okay, uses asyncio directly)
        try:
            while True:
                await asyncio.sleep(15)
                # Add timeout to ping
                try:
                    await asyncio.wait_for(self.websocket.ping(), timeout=10)
                except asyncio.TimeoutError:
                    logger.warning(f"[{self.client_id}] Ping timeout. Closing connection.")
                    await self.websocket.close(code=1011, reason="Ping timeout")
                    break
                except websockets.exceptions.ConnectionClosed:
                     logger.info(f"[{self.client_id}] Connection closed before ping response.")
                     break

        except asyncio.CancelledError:
            logger.info(f"[{self.client_id}] Keepalive task cancelled.")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"[{self.client_id}] Connection closed, stopping keepalive.")
        except Exception as e:
             logger.error(f"[{self.client_id}] Error in keepalive task: {e}", exc_info=True)
        finally:
            logger.info(f"[{self.client_id}] Keepalive task finished.")


# --- Server Entry Point ---
async def main():
    logger.info("Starting model pre-initialization using SingletonMeta.get_instance...")
    try:
        # Use the async get_instance method for each singleton
        whisper = await WhisperTranscriber.get_instance()
        gemma = await GemmaMultimodalProcessor.get_instance()
        tts = await KokoroTTSProcessor.get_instance()
        logger.info("Models pre-initialized successfully via get_instance.")
    except Exception as e:
        logger.error(f"Fatal error during model initialization: {e}", exc_info=True)
        sys.exit(1)


    async def client_connection_wrapper(websocket, path):
        """Wraps the client handling in its class, passing singletons."""
        # Pass the already initialized singletons to the handler
        handler = ClientHandler(websocket, whisper, gemma, tts)
        await handler.handle_connection()


    port = 9073
    host = "0.0.0.0"
    logger.info(f"Starting WebSocket server on {host}:{port}")

    try:
        server = await websockets.serve(
            client_connection_wrapper,
            host,
            port,
            ping_interval=20,
            ping_timeout=30, # Should be > ping_interval
            close_timeout=10,
            max_size=2**24, # 16MB limit
            max_queue=64
        )
        logger.info(f"WebSocket server running. Listening for connections...")
        # Keep server running indefinitely
        await asyncio.Future()

    except Exception as e:
        logger.error(f"Server failed to start or crashed: {e}", exc_info=True)
    finally:
        logger.info("Shutting down executor...")
        executor.shutdown(wait=True) # Wait for pending tasks in executor
        logger.info("Server shut down.")

if __name__ == "__main__":
    # Ensure loop exists for cleanup
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt).")
    finally:
        logger.info("Starting final cleanup...")
        # Clean up any remaining asyncio tasks gracefully
        tasks = [t for t in asyncio.all_tasks(loop=loop) if t is not asyncio.current_task(loop=loop)]
        if tasks:
             logger.info(f"Cancelling {len(tasks)} outstanding asyncio tasks...")
             [task.cancel() for task in tasks]
             # Allow tasks to finish cancelling
             try:
                 loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                 logger.info("Outstanding tasks cancelled.")
             except Exception as e:
                  logger.error(f"Error during final task cancellation: {e}")

        # Ensure loop is closed if it's still running
        if loop.is_running():
            loop.stop()
        if not loop.is_closed():
            loop.close()
            logger.info("Event loop closed.")
        logger.info("Cleanup complete.")
