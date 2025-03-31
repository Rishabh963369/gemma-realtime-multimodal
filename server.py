import asyncio
import json
import websockets
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, Gemma3ForConditionalGeneration, TextIteratorStreamer
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
logging.getLogger("transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# --- Global Thread Pool Executor for CPU-bound tasks ---
# To avoid blocking the asyncio event loop with synchronous library calls
executor = ThreadPoolExecutor(max_workers=os.cpu_count())

# --- Singleton Pattern for Models ---
# Ensures models are loaded only once

class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

# --- Audio Segment Detector ---
class AudioSegmentDetector:
    """Detects speech segments based on audio energy levels."""
    def __init__(self,
                 sample_rate=16000,
                 energy_threshold=0.015, # Adjusted threshold slightly, may need tuning
                 silence_duration=0.7,   # Slightly shorter silence duration
                 min_speech_duration=0.5, # Shorter min duration to catch quicker utterances
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
        self._loop = asyncio.get_event_loop()

        logger.info(f"AudioSegmentDetector initialized: "
                    f"Threshold={energy_threshold}, Silence={silence_duration}s, "
                    f"MinSpeech={min_speech_duration}s, MaxSpeech={max_speech_duration}s")

    async def _calculate_energy(self, audio_bytes):
        """Helper to calculate energy asynchronously."""
        if not audio_bytes:
            return 0.0
        # Run in executor to avoid blocking event loop with numpy potentially
        return await self._loop.run_in_executor(
            executor,
            self._sync_calculate_energy,
            audio_bytes
        )

    def _sync_calculate_energy(self, audio_bytes):
        """Synchronous energy calculation."""
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
        """
        Add audio data, detect segments, and trigger interrupt if needed.

        Args:
            audio_bytes: Raw PCM audio data.
            is_tts_playing_func: Async function to check if TTS is active.
            cancel_tasks_func: Async function to cancel ongoing tasks.
        """
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
                # If TTS is playing when new speech starts, interrupt it
                if await is_tts_playing_func():
                    logger.warning("New speech detected during TTS playback. Sending interrupt.")
                    await cancel_tasks_func() # Cancel ongoing TTS/Generation
                # --- END INTERRUPT LOGIC ---

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
                    speech_segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])
                    segment_len_samples = len(speech_segment_bytes) // self.bytes_per_sample

                    # Reset buffer and state
                    self.audio_buffer = self.audio_buffer[speech_end_idx:]
                    self.is_speech_active = False
                    self.silence_counter = 0

                    if segment_len_samples >= self.min_speech_samples:
                        self.segments_detected_count += 1
                        logger.info(f"Speech segment [Silence End] detected ({segment_len_samples / self.sample_rate:.2f}s). Queueing.")
                        await self.segment_queue.put(speech_segment_bytes)
                    else:
                         logger.info(f"Speech segment below min duration ({segment_len_samples / self.sample_rate:.2f}s). Discarding.")


            # Check for max speech duration limit
            current_speech_len_samples = (len(self.audio_buffer) - self.speech_start_idx) // self.bytes_per_sample
            if self.is_speech_active and current_speech_len_samples > self.max_speech_samples:
                logger.warning(f"Max speech duration ({self.max_speech_duration}s) exceeded.")
                speech_end_idx = self.speech_start_idx + (self.max_speech_samples * self.bytes_per_sample)
                speech_segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])

                # Keep the buffer after the max duration point
                self.audio_buffer = self.audio_buffer[speech_end_idx:]
                # Reset speech start index relative to the *new* buffer start
                self.speech_start_idx = 0
                # We are still potentially in speech, just forced a segment break
                self.silence_counter = 0 # Reset silence counter as we forced a break

                self.segments_detected_count += 1
                logger.info(f"Speech segment [Max Duration] detected ({self.max_speech_duration:.2f}s). Queueing.")
                await self.segment_queue.put(speech_segment_bytes)

                # --- INTERRUPT LOGIC ---
                # Also interrupt if max duration is hit and TTS was playing
                if await is_tts_playing_func():
                     logger.warning("Max speech duration hit during TTS playback. Sending interrupt.")
                     await cancel_tasks_func()
                # --- END INTERRUPT LOGIC ---

        # Keep buffer size manageable (e.g., keep last max_speech_duration + silence_duration)
        max_buffer_samples = (self.max_speech_samples + self.silence_samples)
        if len(self.audio_buffer) > max_buffer_samples * self.bytes_per_sample:
            keep_bytes = max_buffer_samples * self.bytes_per_sample
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
        logger.info("Initializing WhisperTranscriber...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda:0" else torch.float32
        # Using distil-whisper for potentially faster transcription with good accuracy
        # model_id = "distil-whisper/distil-large-v3"
        model_id = "openai/whisper-large-v3" # Reverted back based on original code - change if speed is paramount

        logger.info(f"Loading Whisper model: {model_id} on {self.device} with {self.torch_dtype}")
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                max_new_tokens=128, # Limit max tokens for ASR
                chunk_length_s=20, # Process in chunks
                batch_size=8,      # Batching for potential speedup
                return_timestamps=False, # Don't need timestamps for this use case
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            logger.info("WhisperTranscriber initialized successfully.")
            self.transcription_count = 0
            self._loop = asyncio.get_event_loop()
        except Exception as e:
            logger.error(f"Error initializing Whisper: {e}", exc_info=True)
            raise

    async def transcribe(self, audio_bytes, sample_rate=16000):
        """Transcribe audio bytes to text asynchronously."""
        if len(audio_bytes) < sample_rate * 0.2 * 2 : # Ignore very short segments (e.g., < 0.2s)
            logger.warning(f"Audio segment too short ({len(audio_bytes)} bytes), skipping transcription.")
            return ""
        start_time = time.time()
        try:
            # Prepare audio data (run in executor to avoid blocking)
            audio_input = await self._loop.run_in_executor(
                executor, self._prepare_audio, audio_bytes, sample_rate
            )
            if audio_input is None: return ""

            # Run pipeline in executor
            result = await self._loop.run_in_executor(
                executor,
                lambda: self.pipe(
                    audio_input,
                    generate_kwargs={"language": "english", "task": "transcribe", "temperature": 0.0}
                )
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
        logger.info("Initializing GemmaMultimodalProcessor...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_id = "google/gemma-3-4b-it" # Using 4B-IT as requested
        quantization_config = None
        torch_dtype = torch.bfloat16 # bfloat16 is generally good for Gemma on Ampere+ GPUs
        if torch.cuda.is_available():
             # Only quantize if CUDA is available
             from transformers import BitsAndBytesConfig
             quantization_config = BitsAndBytesConfig(
                 load_in_8bit=True, # Use 8-bit as requested
             )
             logger.info("Using 8-bit quantization for Gemma.")
        else:
             logger.info("Using bfloat16 for Gemma on CPU (quantization disabled).")


        logger.info(f"Loading Gemma model: {model_id} on {self.device}")
        try:
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quantization_config, # Use quantization config
                # device_map="auto", # device_map='auto' can be tricky with streaming
                torch_dtype=torch_dtype,
                 trust_remote_code=True # Needed for some Gemma versions
            )
            # Manually move to device if not using device_map='auto'
            if not quantization_config: # If not quantized, device_map wasn't used
                 self.model.to(self.device)

            self.processor = AutoProcessor.from_pretrained(model_id)
            self.tokenizer = self.processor.tokenizer # Explicitly get tokenizer for streamer
            logger.info("GemmaMultimodalProcessor initialized successfully.")

            self.last_image = None
            self.last_image_timestamp = 0
            self.message_history = []
            self.max_history_turns = 3 # Keep last 3 turns (User + Assistant pairs)
            self.generation_count = 0
            self._loop = asyncio.get_event_loop()

        except Exception as e:
            logger.error(f"Error initializing Gemma: {e}", exc_info=True)
            raise


    async def set_image(self, image_data):
        """Cache the most recent image received."""
        try:
            image = await self._loop.run_in_executor(
                executor, self._process_image, image_data
            )
            if image:
                self.last_image = image
                self.last_image_timestamp = time.time()
                # Clear history when a new image context is set
                self.message_history = []
                logger.info(f"New image set (resized to {image.size}). History cleared.")
                return True
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
        return False

    def _process_image(self, image_data):
        """Synchronous image processing (resizing)."""
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            # Resize to make it smaller for faster processing, adjust as needed
            # Keeping original size as resizing wasn't explicitly asked for fixing perf issues
            # max_size = (1024, 1024)
            # image.thumbnail(max_size, Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            logger.error(f"Pillow error processing image: {e}", exc_info=True)
            return None

    def _build_prompt(self, text):
        """Builds the prompt string including history for the model."""
        # System Prompt
        system_prompt = """You are a helpful, conversational AI assistant. You are looking at an image provided by the user. Respond concisely and naturally based on the user's spoken request. Keep responses suitable for text-to-speech (1-3 sentences usually). If the request is clearly about the image, describe what you see relevant to the request. If the request is conversational or unclear, respond naturally without forcing image description. Acknowledge the conversation history implicitly."""

        # Prepare history turns
        history = "\n".join([f"User: {turn['user']}\nAssistant: {turn['assistant']}" for turn in self.message_history])

        # Combine components
        full_prompt = f"{system_prompt}\n\n{history}\n\nUser: {text}\nAssistant:"
        return full_prompt

    def _update_history(self, user_text, assistant_response):
        """Update message history."""
        self.message_history.append({"user": user_text, "assistant": assistant_response})
        # Trim history
        if len(self.message_history) > self.max_history_turns:
            self.message_history = self.message_history[-self.max_history_turns:]

    async def generate_response_stream(self, text):
        """Generate response with streaming."""
        if self.last_image is None:
            logger.warning("Cannot generate response, no image available.")
            # Yield a fallback message
            yield "I don't have an image context right now. Could you provide one?"
            return

        start_time = time.time()
        prompt = self._build_prompt(text) # Use the simplified text prompt builder
        # Prepare inputs including the image
        try:
            inputs = self.processor(text=prompt, images=self.last_image, return_tensors="pt").to(self.model.device, dtype=self.model.dtype) # Ensure dtype matches
        except Exception as e:
            logger.error(f"Error processing Gemma inputs: {e}", exc_info=True)
            yield "Sorry, I had trouble processing that request with the image."
            return

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Generation arguments
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=150, # Max length of the generated response
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1 # Slight penalty for repeating
        )

        # Run generation in a separate thread via executor to avoid blocking asyncio loop
        # But the streamer needs to be accessed in the main thread/loop
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        token_count = 0
        first_token_time = None

        try:
            for new_text in streamer:
                if not first_token_time:
                    first_token_time = time.time()
                yield new_text
                generated_text += new_text
                token_count += 1 # Approximate token count
            thread.join() # Ensure thread finishes
            duration = time.time() - start_time
            ttft = (first_token_time - start_time) if first_token_time else duration
            self.generation_count += 1
            logger.info(f"Gemma Stream #{self.generation_count}: TTFT={ttft:.3f}s, Total={duration:.3f}s, Tokens~={token_count}")
            # Update history after full generation
            self._update_history(text, generated_text)

        except Exception as e:
             duration = time.time() - start_time
             logger.error(f"Gemma streaming generation failed after {duration:.3f}s: {e}", exc_info=True)
             # Yield an error message if streaming started, otherwise log handled it
             if first_token_time: # If we already started streaming, send error via stream
                 yield " I encountered an error while generating the rest of the response."
             thread.join() # Ensure thread is cleaned up even on error


# --- Kokoro TTS Processor ---
class KokoroTTSProcessor(metaclass=SingletonMeta):
    """Handles text-to-speech conversion using Kokoro."""
    def __init__(self):
        logger.info("Initializing KokoroTTSProcessor...")
        try:
            self.pipeline = KPipeline(lang_code='en') # Assuming English based on Whisper settings
            self.default_voice = 'en_us_sarah' # Example English voice
            logger.info(f"KokoroTTSProcessor initialized successfully with voice {self.default_voice}.")
            self.synthesis_count = 0
            self._loop = asyncio.get_event_loop()
            self.sample_rate = 24000 # Kokoro default sample rate
        except Exception as e:
            logger.error(f"Error initializing Kokoro TTS: {e}", exc_info=True)
            self.pipeline = None
            raise

    async def synthesize_speech_stream(self, text_iterator):
        """
        Convert text to speech chunk by chunk as text arrives.
        This is complex to do perfectly with Kokoro's current generator interface.
        A simpler approach: synthesize sentence by sentence.
        """
        if not self.pipeline:
            logger.error("Kokoro TTS pipeline not available.")
            return

        sentence_buffer = ""
        # Improved sentence splitting regex
        sentence_end_pattern = re.compile(r'(?<=[.!?])\s+')

        try:
            async for text_chunk in text_iterator:
                sentence_buffer += text_chunk
                # Try splitting into sentences
                sentences = sentence_end_pattern.split(sentence_buffer)

                # If we have at least one complete sentence (and possibly a partial one)
                if len(sentences) > 1:
                    # Process all complete sentences
                    for i in range(len(sentences) - 1):
                        sentence = sentences[i].strip()
                        if sentence:
                            start_time = time.time()
                            # Synthesize this sentence
                            audio_data = await self.synthesize_single_sentence(sentence)
                            duration = time.time() - start_time
                            if audio_data is not None:
                                self.synthesis_count += 1
                                logger.info(f"TTS Stream #{self.synthesis_count}: Synthesized sentence ('{sentence[:30]}...') took {duration:.3f}s")
                                yield audio_data
                            else:
                                logger.warning(f"TTS failed for sentence: '{sentence[:30]}...'")
                    # Keep the remaining partial sentence in the buffer
                    sentence_buffer = sentences[-1]
                # Add a small sleep to prevent tight loop if text comes very fast
                await asyncio.sleep(0.01)


            # Process any remaining text in the buffer after the iterator finishes
            final_sentence = sentence_buffer.strip()
            if final_sentence:
                start_time = time.time()
                audio_data = await self.synthesize_single_sentence(final_sentence)
                duration = time.time() - start_time
                if audio_data is not None:
                    self.synthesis_count += 1
                    logger.info(f"TTS Stream #{self.synthesis_count}: Synthesized final part ('{final_sentence[:30]}...') took {duration:.3f}s")
                    yield audio_data
                else:
                    logger.warning(f"TTS failed for final part: '{final_sentence[:30]}...'")


        except asyncio.CancelledError:
            logger.info("TTS Synthesis stream cancelled.")
            raise # Propagate cancellation
        except Exception as e:
            logger.error(f"Kokoro TTS streaming synthesis error: {e}", exc_info=True)
            # Don't yield here, error logged


    async def synthesize_single_sentence(self, sentence):
        """Synthesize a single sentence using Kokoro."""
        if not self.pipeline or not sentence:
            return None
        try:
            # Run synchronous Kokoro call in executor
            combined_audio = await self._loop.run_in_executor(
                 executor, self._sync_synthesize, sentence
             )
            return combined_audio
        except Exception as e:
            # Log specific sentence failure
            logger.error(f"Kokoro synthesis failed for sentence '{sentence[:50]}...': {e}", exc_info=False) # Reduce log noise
            return None

    def _sync_synthesize(self, text):
        """Synchronous synthesis for executor."""
        # Kokoro's pipeline returns a generator even for single calls
        audio_segments = []
        try:
             # Use split_pattern=None to treat the input as a single unit ideally
             # Kokoro might still internally split based on its own logic
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
                 # Ensure output is float32 numpy array
                 combined = np.concatenate(audio_segments).astype(np.float32)
                 # Normalize audio slightly to prevent clipping if needed, but be careful
                 # max_val = np.max(np.abs(combined))
                 # if max_val > 1.0:
                 #     combined = combined / max_val
                 return combined
            else:
                 logger.warning(f"Kokoro returned no audio for text: '{text[:50]}...'")
                 return None
        except Exception as e:
             logger.error(f"Kokoro _sync_synthesize error for text '{text[:50]}...': {e}", exc_info=False)
             return None


# --- WebSocket Handler ---
class ClientHandler:
    def __init__(self, websocket):
        self.websocket = websocket
        self.detector = AudioSegmentDetector()
        self.transcriber = WhisperTranscriber() # Get singleton instance
        self.gemma_processor = GemmaMultimodalProcessor() # Get singleton instance
        self.tts_processor = KokoroTTSProcessor() # Get singleton instance

        self._tts_playing = False
        self._tts_lock = asyncio.Lock()
        self._active_tasks = set()
        self._loop = asyncio.get_event_loop()
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
        if not self._active_tasks:
            return

        logger.warning(f"[{self.client_id}] Cancelling {len(self._active_tasks)} active task(s)...")
        # Send interrupt message *before* cancelling server-side tasks
        try:
            interrupt_message = json.dumps({"interrupt": True})
            logger.info(f"[{self.client_id}] Sending interrupt message to client.")
            await self.websocket.send(interrupt_message)
        except websockets.exceptions.ConnectionClosed:
             logger.warning(f"[{self.client_id}] Connection closed while sending interrupt message.")
             # Don't try to cancel tasks if connection is gone
             self._active_tasks.clear()
             await self._set_tts_playing(False) # Ensure state is reset
             return
        except Exception as e:
             logger.error(f"[{self.client_id}] Error sending interrupt message: {e}", exc_info=True)


        tasks_to_cancel = list(self._active_tasks)
        self._active_tasks.clear() # Clear immediately

        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
                # Give event loop a chance to process cancellation
                await asyncio.sleep(0)

        # Optionally wait briefly for tasks to finish cancelling
        try:
            # Use wait_for with a short timeout on each task
            # This is tricky as tasks might not handle cancellation instantly
            # A simple sleep might be sufficient
             await asyncio.sleep(0.1) # Short delay to allow cancellations to process
        except asyncio.CancelledError:
            pass # Expected if the handler itself is cancelled

        logger.info(f"[{self.client_id}] Active tasks cancellation requested.")
        # Ensure TTS state is false after cancellation
        await self._set_tts_playing(False)


    async def _process_segment(self, speech_segment):
        """Process a single detected speech segment."""
        # 1. Transcribe
        transcription = await self.transcriber.transcribe(speech_segment)
        if not transcription or not any(c.isalnum() for c in transcription):
            logger.info(f"[{self.client_id}] Empty or non-alphanumeric transcription: '{transcription}'. Skipping.")
            return

        # Filter common fillers (optional, based on previous logic)
        filler_patterns = [
             r'^(um+|uh+|ah+|oh+|hm+|mhm+|hmm+)$',
             r'^(okay|yes|no|yeah|nah)$',
             r'^bye+$',
             r'^thank you$',
             r'^thanks$'
         ]
        if any(re.fullmatch(pattern, transcription.lower()) for pattern in filler_patterns):
             logger.info(f"[{self.client_id}] Skipping likely filler phrase: '{transcription}'")
             return

        logger.info(f"[{self.client_id}] User said: '{transcription}'")

        # --- Start Generation and TTS ---
        # Ensure previous tasks are cancelled (belt-and-suspenders)
        await self._cancel_active_tasks()
        await self._set_tts_playing(True) # Mark TTS as active for the duration of generation + playback

        # Create tasks and add them to the active set
        generation_stream = None
        tts_stream = None
        playback_task = None

        try:
            # 2. Generate Response Stream from Gemma
            # Use an internal queue to bridge generation and TTS
            text_queue = asyncio.Queue()
            gen_task = self._loop.create_task(self._run_gemma_stream(transcription, text_queue))
            self._active_tasks.add(gen_task)

            # 3. Synthesize Speech Stream from Kokoro based on Gemma's output
            # The synth task reads from the text_queue
            audio_queue = asyncio.Queue()
            synth_task = self._loop.create_task(self._run_tts_stream(text_queue, audio_queue))
            self._active_tasks.add(synth_task)

            # 4. Playback Audio Stream (send to client)
            # The playback task reads from the audio_queue
            playback_task = self._loop.create_task(self._run_playback_stream(audio_queue))
            self._active_tasks.add(playback_task)

            # Wait for all tasks in this chain to complete naturally
            # If any task raises an exception (including CancelledError), it will propagate here
            await asyncio.gather(gen_task, synth_task, playback_task)
            logger.info(f"[{self.client_id}] Full response cycle completed for: '{transcription}'")

        except asyncio.CancelledError:
            logger.info(f"[{self.client_id}] Response cycle cancelled for: '{transcription}'")
            # Cancellation is handled by _cancel_active_tasks, just log here
            # Ensure tasks are removed (should be handled by cancel function)
            self._active_tasks.discard(gen_task)
            self._active_tasks.discard(synth_task)
            if playback_task: self._active_tasks.discard(playback_task)

        except Exception as e:
            logger.error(f"[{self.client_id}] Error during response cycle for '{transcription}': {e}", exc_info=True)
            # Attempt to cancel any remaining tasks from this cycle
            await self._cancel_active_tasks() # Ensure cleanup on error

        finally:
            # --- Cleanup for this segment ---
            # Ensure TTS state is reset *unless* cancellation happened due to new speech
            # _cancel_active_tasks already sets tts_playing to False
            if not await self._is_tts_playing(): # Check if it wasn't already reset by an interrupt
                 await self._set_tts_playing(False)

            # Ensure all tasks related to *this specific segment* are removed if not already
            if 'gen_task' in locals() and gen_task in self._active_tasks: self._active_tasks.remove(gen_task)
            if 'synth_task' in locals() and synth_task in self._active_tasks: self._active_tasks.remove(synth_task)
            if 'playback_task' in locals() and playback_task in self._active_tasks: self._active_tasks.remove(playback_task)


    async def _run_gemma_stream(self, text, output_queue):
        """Task to run Gemma generation and put text chunks into a queue."""
        try:
            async for chunk in self.gemma_processor.generate_response_stream(text):
                await output_queue.put(chunk)
                await asyncio.sleep(0) # Yield control
        except asyncio.CancelledError:
             logger.info(f"[{self.client_id}] Gemma generation task cancelled.")
             raise
        except Exception as e:
             logger.error(f"[{self.client_id}] Error in Gemma generation task: {e}", exc_info=True)
             await output_queue.put(None) # Signal error downstream
        finally:
             await output_queue.put(None) # Signal end of stream

    async def _run_tts_stream(self, input_queue, output_queue):
        """Task to run TTS synthesis based on text chunks from a queue."""
        async def text_iterator():
            while True:
                chunk = await input_queue.get()
                if chunk is None: # End of stream signal
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
            raise # Propagate cancellation
        except Exception as e:
            logger.error(f"[{self.client_id}] Error in TTS synthesis task: {e}", exc_info=True)
            await output_queue.put(None) # Signal error downstream
        finally:
             # Ensure the input queue is drained if synthesis ends early/errors
             while not input_queue.empty():
                 await input_queue.get()
                 input_queue.task_done()
             await output_queue.put(None) # Signal end of audio stream

    async def _run_playback_stream(self, input_queue):
        """Task to send synthesized audio chunks to the client."""
        stream_start_time = time.time()
        audio_sent_samples = 0
        try:
            while True:
                audio_chunk = await input_queue.get()
                if audio_chunk is None: # End of stream signal
                    break

                # Convert numpy float32 audio to int16 bytes -> base64
                try:
                    # Ensure it's float32 before conversion
                    if audio_chunk.dtype != np.float32:
                        audio_chunk = audio_chunk.astype(np.float32)
                    # Clamp values to avoid issues during int16 conversion
                    np.clip(audio_chunk, -1.0, 1.0, out=audio_chunk)
                    audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                    base64_audio = await self._loop.run_in_executor(executor, base64.b64encode, audio_bytes)
                    base64_audio_str = base64_audio.decode('utf-8')

                    # Send to client
                    await self.websocket.send(json.dumps({
                        "audio": base64_audio_str,
                        "sample_rate": self.tts_processor.sample_rate # Send sample rate
                    }))
                    audio_sent_samples += len(audio_chunk)
                    # logger.debug(f"Sent audio chunk: {len(audio_bytes)} bytes") # Debug level
                except websockets.exceptions.ConnectionClosed:
                    logger.warning(f"[{self.client_id}] Connection closed during audio playback.")
                    # If connection closed, cancel upstream tasks (TTS, Gen)
                    await self._cancel_active_tasks() # Trigger cancellation upwards
                    break # Exit playback loop
                except Exception as e:
                    logger.error(f"[{self.client_id}] Error preparing/sending audio chunk: {e}", exc_info=True)

                input_queue.task_done() # Mark item as processed
                await asyncio.sleep(0.01) # Small sleep to yield control

        except asyncio.CancelledError:
            logger.info(f"[{self.client_id}] Audio playback task cancelled.")
            # Don't raise here, cancellation originated elsewhere
        except Exception as e:
             logger.error(f"[{self.client_id}] Error in playback task: {e}", exc_info=True)
        finally:
             # Drain queue on exit/error
             while not input_queue.empty():
                 await input_queue.get()
                 input_queue.task_done()
             duration = time.time() - stream_start_time
             if audio_sent_samples > 0:
                 logger.info(f"[{self.client_id}] Audio playback finished. Sent {audio_sent_samples / self.tts_processor.sample_rate:.2f}s of audio in {duration:.2f}s.")
             # Playback finished, TTS is no longer playing *from this cycle*
             # Note: _cancel_active_tasks handles the main flag, this is redundant but safe
             await self._set_tts_playing(False)


    # --- Main Client Interaction Loops ---

    async def run():
        """Static method to initialize models before starting server."""
        logger.info("Pre-initializing models...")
        WhisperTranscriber()
        GemmaMultimodalProcessor()
        KokoroTTSProcessor()
        logger.info("Model pre-initialization complete.")

    async def handle_connection(self):
        """Main handler for a single client connection."""
        logger.info(f"[{self.client_id}] Client connected.")
        try:
            # Send initial config or confirmation if needed by client
            await self.websocket.send(json.dumps({"status": "connected", "server_id": "GemmaKokoroV1"}))

            # Start concurrent tasks for this client
            receive_task = self._loop.create_task(self.receive_messages())
            process_task = self._loop.create_task(self.process_speech_queue())
            keepalive_task = self._loop.create_task(self.send_keepalive())

            # Keep tasks running
            done, pending = await asyncio.wait(
                [receive_task, process_task, keepalive_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # If one task finishes (e.g., receive due to disconnect), cancel others
            for task in pending:
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
            logger.info(f"[{self.client_id}] Cleaning up resources...")
            # Ensure all tasks are cancelled on exit
            await self._cancel_active_tasks()
            # Clean up any remaining tasks specifically tracked here if needed
            logger.info(f"[{self.client_id}] Client handler finished.")


    async def receive_messages(self):
        """Task to receive messages from the WebSocket client."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)

                    # Handle audio data chunks
                    if "audio_chunk" in data:
                        audio_data = base64.b64decode(data["audio_chunk"])
                        # Pass callbacks for state checking and cancellation
                        await self.detector.add_audio(
                            audio_data,
                            self._is_tts_playing,
                            self._cancel_active_tasks
                        )

                    # Handle image data
                    # Process image even if TTS is playing, but maybe delay generation if needed?
                    # Current Gemma logic handles this (won't generate if no image)
                    elif "image_chunk" in data:
                         image_data = base64.b64decode(data["image_chunk"])
                         await self.gemma_processor.set_image(image_data)
                         logger.info(f"[{self.client_id}] Received image chunk.")


                    # Handle client signals (e.g., explicit stop)
                    elif "signal" in data:
                        if data["signal"] == "stop_tts":
                             logger.info(f"[{self.client_id}] Received 'stop_tts' signal from client.")
                             await self._cancel_active_tasks()


                except json.JSONDecodeError:
                    logger.warning(f"[{self.client_id}] Received invalid JSON: {message[:100]}...")
                except asyncio.CancelledError:
                    logger.info(f"[{self.client_id}] Receive task cancelled.")
                    break # Exit loop cleanly on cancellation
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"[{self.client_id}] Connection closed during message receive.")
                    break # Exit loop
                except Exception as e:
                    logger.error(f"[{self.client_id}] Error processing received message: {e}", exc_info=True)
                    # Decide if we should break or continue
                    # continue
        finally:
            logger.info(f"[{self.client_id}] Receive messages task finished.")


    async def process_speech_queue(self):
        """Task to process detected speech segments from the queue."""
        try:
            while True:
                segment = await self.detector.get_next_segment(timeout=1.0) # Check queue periodically
                if segment:
                    # Check if TTS is currently active *before* processing
                    # If an interrupt happened via add_audio, _cancel_active_tasks would have run
                    # if await self._is_tts_playing():
                    #     logger.warning("Ignoring new segment because TTS is still marked as active (potential race condition?).")
                    #     continue

                    # Process the segment (transcribe -> generate -> synthesize -> play)
                    await self._process_segment(segment)

                # Yield control even if no segment found
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            logger.info(f"[{self.client_id}] Process speech queue task cancelled.")
        except Exception as e:
             logger.error(f"[{self.client_id}] Error in speech processing loop: {e}", exc_info=True)
        finally:
            logger.info(f"[{self.client_id}] Process speech queue task finished.")


    async def send_keepalive(self):
        """Task to send periodic pings to keep the connection alive."""
        try:
            while True:
                await asyncio.sleep(15) # Send ping every 15 seconds
                await self.websocket.ping()
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
    # Pre-initialize models to load them before accepting connections
    logger.info("Starting model pre-initialization...")
    try:
        # Run synchronous model loading in executor to avoid blocking startup
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, WhisperTranscriber)
        await loop.run_in_executor(executor, GemmaMultimodalProcessor)
        await loop.run_in_executor(executor, KokoroTTSProcessor)
        logger.info("Models pre-initialized successfully.")
    except Exception as e:
        logger.error(f"Fatal error during model initialization: {e}", exc_info=True)
        sys.exit(1) # Exit if models can't load


    async def client_connection_wrapper(websocket, path):
        """Wraps the client handling in its class."""
        handler = ClientHandler(websocket)
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
            ping_timeout=30,
            close_timeout=10,
            # Increase max message size if large images are expected
            max_size=2**24, # 16MB limit for messages (adjust if needed)
            # Increase queue size if server needs to buffer more messages under load
            max_queue=64
        )
        logger.info(f"WebSocket server running.")
        await server.wait_closed() # Keep server running until stopped
    except Exception as e:
        logger.error(f"Server failed to start or crashed: {e}", exc_info=True)
    finally:
        logger.info("Shutting down executor...")
        executor.shutdown(wait=True)
        logger.info("Server shut down.")

if __name__ == "__main__":
    # Add handler to gracefully shut down on SIGINT/SIGTERM
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt).")
    finally:
        # Clean up any remaining asyncio tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if tasks:
             logger.info(f"Cancelling {len(tasks)} outstanding tasks...")
             [task.cancel() for task in tasks]
             # Allow tasks to finish cancelling
             loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.close()
        logger.info("Event loop closed.")
