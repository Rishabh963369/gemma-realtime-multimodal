# -*- coding: utf-8 -*-
import asyncio
import json
import websockets
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, Gemma3ForConditionalGeneration, TextIteratorStreamer
from transformers.utils import is_flash_attn_2_available
import numpy as np
import logging
import sys
import io
from PIL import Image
import time
import os
from datetime import datetime
import re
from threading import Thread
import librosa # <-- Import librosa

# Import Kokoro TTS library
# Ensure kokoro is installed: pip install kokoro-tts
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

# --- Configuration ---
WEBSOCKET_PORT = 9073
AUDIO_SAMPLE_RATE = 16000 # Target sample rate for VAD and Whisper
WHISPER_MODEL_ID = "openai/whisper-large-v3"
GEMMA_MODEL_ID = "google/gemma-3-4b-it"
KOKORO_VOICE = 'en_tina' # Example English voice
KOKORO_LANG = 'e'       # English language code for Kokoro
KOKORO_INTERNAL_SAMPLE_RATE = 24000 # <-- Assume Kokoro's output rate

# --- Constants ---
PCM_CHUNK_SIZE = 1024 * 2
IMAGE_RESIZE_FACTOR = 0.75
MAX_HISTORY_MESSAGES = 6
MAX_MESSAGE_SIZE = 10 * 1024 * 1024 # 10 MB limit

# --- Flash Attention Check ---
attn_implementation = None
if is_flash_attn_2_available():
    attn_implementation = "flash_attention_2"
    logger.info("Flash Attention 2 is available. Using it for models.")
else:
    logger.info("Flash Attention 2 not available. Using default attention.")


# ==============================
# Audio Segment Detector (VAD)
# ==============================
class AudioSegmentDetector:
    """Detects speech segments based on audio energy levels."""

    def __init__(self,
                 sample_rate=AUDIO_SAMPLE_RATE, # Use the target rate
                 energy_threshold=0.015,
                 silence_duration=0.7,
                 min_speech_duration=0.5,
                 max_speech_duration=15.0):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_samples = int(silence_duration * sample_rate)
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.max_speech_samples = int(max_speech_duration * sample_rate)
        self.bytes_per_sample = 2 # 16-bit PCM

        self.audio_buffer = bytearray()
        self.is_speech_active = False
        self.silence_counter = 0
        self.speech_start_idx = 0
        self.segment_queue = asyncio.Queue()
        self.lock = asyncio.Lock()
        self.segments_detected_count = 0
        logger.info(f"AudioSegmentDetector initialized with SR={sample_rate}Hz")


    async def add_audio(self, audio_bytes: bytes):
        """Add audio data and detect speech segments."""
        async with self.lock:
            self.audio_buffer.extend(audio_bytes)

            # Process in chunks
            while len(self.audio_buffer) >= self.speech_start_idx + PCM_CHUNK_SIZE * self.bytes_per_sample:
                current_chunk_start = self.speech_start_idx if self.is_speech_active else len(self.audio_buffer) - (PCM_CHUNK_SIZE * self.bytes_per_sample)
                current_chunk_end = current_chunk_start + (PCM_CHUNK_SIZE * self.bytes_per_sample)
                chunk_bytes = self.audio_buffer[current_chunk_start:current_chunk_end]

                if not chunk_bytes:
                     break

                audio_array = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                energy = np.sqrt(np.mean(audio_array**2)) if len(audio_array) > 0 else 0.0

                if not self.is_speech_active:
                    if energy > self.energy_threshold:
                        self.is_speech_active = True
                        self.speech_start_idx = max(0, current_chunk_start - int(0.1 * self.sample_rate * self.bytes_per_sample))
                        self.silence_counter = 0
                        logger.info(f"Speech START detected (energy: {energy:.4f})")
                    else:
                         self.speech_start_idx = len(self.audio_buffer) - (PCM_CHUNK_SIZE * self.bytes_per_sample)
                else:
                    current_speech_len_bytes = len(self.audio_buffer) - self.speech_start_idx
                    current_speech_len_samples = current_speech_len_bytes // self.bytes_per_sample

                    if energy > self.energy_threshold:
                        self.silence_counter = 0
                    else:
                        self.silence_counter += len(audio_array)

                        if self.silence_counter >= self.silence_samples:
                            speech_end_idx = len(self.audio_buffer) - (self.silence_counter * self.bytes_per_sample)
                            await self._finalize_segment(self.speech_start_idx, speech_end_idx, "Silence")
                            self.is_speech_active = False
                            self.silence_counter = 0
                            self.audio_buffer = self.audio_buffer[speech_end_idx:]
                            self.speech_start_idx = 0
                            break

                    if self.is_speech_active and current_speech_len_samples >= self.max_speech_samples:
                        logger.warning(f"Max duration reached ({self.max_speech_duration}s). Finalizing segment.")
                        speech_end_idx = self.speech_start_idx + self.max_speech_samples * self.bytes_per_sample
                        await self._finalize_segment(self.speech_start_idx, speech_end_idx, "Max Duration")
                        # is_speech_active remains True potentially
                        self.silence_counter = 0
                        # Advance start index, DON'T trim buffer here
                        self.speech_start_idx = speech_end_idx
                        break # Re-evaluate next chunk from new start index


            if not self.is_speech_active and len(self.audio_buffer) > self.max_speech_samples * self.bytes_per_sample * 2:
                 keep_bytes = self.silence_samples * self.bytes_per_sample
                 self.audio_buffer = self.audio_buffer[-keep_bytes:]
                 self.speech_start_idx = 0


    async def _finalize_segment(self, start_idx, end_idx, reason=""):
        """Extracts the segment, validates, and puts it on the queue."""
        # Clamp indices to buffer bounds just in case
        start_idx = max(0, start_idx)
        end_idx = min(len(self.audio_buffer), end_idx)
        if start_idx >= end_idx:
             logger.warning(f"Attempted to finalize zero-length segment ({reason}).")
             return

        speech_segment_bytes = bytes(self.audio_buffer[start_idx:end_idx])
        segment_len_samples = len(speech_segment_bytes) // self.bytes_per_sample

        if segment_len_samples >= self.min_speech_samples:
            self.segments_detected_count += 1
            duration = segment_len_samples / self.sample_rate
            logger.info(f"Speech segment DETECTED ({reason}): {duration:.2f}s (Total: {self.segments_detected_count})")
            await self.segment_queue.put(speech_segment_bytes)
        else:
            duration = segment_len_samples / self.sample_rate
            logger.info(f"Speech segment IGNORED ({reason}, too short: {duration:.2f}s < {self.min_speech_duration:.2f}s)")


    async def get_next_segment(self, timeout=0.1):
        """Get the next available speech segment with a timeout."""
        try:
            return await asyncio.wait_for(self.segment_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

# ==============================
# Whisper Transcriber
# ==============================
class WhisperTranscriber:
    """Handles speech transcription using Whisper model."""
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        logger.info("Initializing Whisper Transcriber...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda:0" else torch.float32
        logger.info(f"Whisper using device: {self.device}, dtype: {self.torch_dtype}")
        self.bytes_per_sample = 2 # 16-bit PCM

        logger.info(f"Loading Whisper model: {WHISPER_MODEL_ID}...")
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                WHISPER_MODEL_ID,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation=attn_implementation
            )
            self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                # max_new_tokens=128, # Pass in generate_kwargs
                chunk_length_s=30,
                batch_size=16,
                return_timestamps=False,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            logger.info("Whisper pipeline ready.")
            self.transcription_count = 0
        except Exception as e:
            logger.error(f"FATAL: Failed to load Whisper model: {e}", exc_info=True)
            raise

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text."""
        self.transcription_count += 1
        start_time = time.monotonic()
        try:
             # Check based on feature extractor's expected input size
            min_samples_needed = self.processor.feature_extractor.hop_length if hasattr(self.processor.feature_extractor, 'hop_length') else 160 # fallback guess
            if len(audio_bytes) < min_samples_needed * self.bytes_per_sample:
                logger.warning(f"Transcription skipped: Audio segment too short ({len(audio_bytes)} bytes).")
                return ""

            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipe(
                    audio_array,
                    # Ensure sampling rate matches what Whisper expects (usually 16k)
                    # The input audio_array is already at AUDIO_SAMPLE_RATE (16k)
                    generate_kwargs={
                        "language": "english",
                        "task": "transcribe",
                        "temperature": 0.0,
                        "max_new_tokens": 128
                        }
                )
            )

            text = result.get("text", "").strip()
            duration = time.monotonic() - start_time
            logger.info(f"Transcription #{self.transcription_count} ({duration:.2f}s): '{text}'")

            if not text or not any(c.isalnum() for c in text):
                logger.info("Transcription invalid (empty or punctuation only).")
                return ""
            words = text.lower().split()
            if len(words) <= 1 and words[0] in ["okay", "yes", "no", "yeah", "nah", "bye", "uh", "um", "hmm", "thanks", "thank you", "hi", "hello"]:
                 logger.info(f"Skipping single-word filler/greeting: '{text}'")
                 return ""
            if "fuck" in text.lower() or "shit" in text.lower():
                 logger.warning(f"Potential profanity detected in transcription: '{text}'. Replacing.")
                 return "Sorry, I cannot process that phrase."

            return text

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return ""

# ==============================
# Gemma Multimodal Processor
# ==============================
class GemmaMultimodalProcessor:
    """Handles multimodal generation using Gemma model."""
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        logger.info("Initializing Gemma Multimodal Processor...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        logger.info(f"Gemma using device: {self.device}, dtype: {self.torch_dtype}")

        logger.info(f"Loading Gemma model: {GEMMA_MODEL_ID}...")
        try:
             # Explicitly set max_memory if accelerate gives warnings
            max_memory = {0: torch.cuda.get_device_properties(0).total_memory * 0.9} if self.device == "cuda:0" else None # 90%
            if max_memory:
                logger.info(f"Setting max_memory: {max_memory}")

            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                GEMMA_MODEL_ID,
                device_map="auto",
                torch_dtype=self.torch_dtype,
                attn_implementation=attn_implementation,
                max_memory=max_memory,
            )
            self.processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)
            logger.info("Gemma model ready.")

            self.last_image = None
            self.last_image_timestamp = 0
            self.image_lock = asyncio.Lock()
            self.message_history = []
            self.generation_count = 0

        except Exception as e:
            logger.error(f"FATAL: Failed to load Gemma model: {e}", exc_info=True)
            raise

    async def set_image(self, image_data: bytes):
        """Cache the most recent image, resizing it."""
        async with self.image_lock:
            try:
                img_pil = Image.open(io.BytesIO(image_data)).convert("RGB") # Ensure RGB

                if IMAGE_RESIZE_FACTOR != 1.0:
                    orig_size = img_pil.size
                    new_size = (int(orig_size[0] * IMAGE_RESIZE_FACTOR), int(orig_size[1] * IMAGE_RESIZE_FACTOR))
                    img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)
                    logger.info(f"Resized image from {orig_size} to {new_size}")

                self.last_image = img_pil
                self.last_image_timestamp = time.monotonic()
                logger.info("New image set.")
                return True
            except Exception as e:
                logger.error(f"Error processing image: {e}", exc_info=True)
                self.last_image = None
                return False

    def _build_prompt(self, text: str) -> str:
        """Builds the prompt string including history and image context."""
        system_prompt = """You are a helpful, conversational AI assistant.
Engage naturally with the user. If the user asks about the image provided, describe it concisely.
If the user's query isn't about the image, respond conversationally without forcing the image description.
Keep responses brief (1-3 sentences), clear, and suitable for spoken delivery. Avoid lists or complex formatting.
If the user's request is unclear, ask for clarification politely.
Use simple, everyday language. Avoid jargon. Be friendly."""

        prompt_parts = [f"<|system|>\n{system_prompt}"]
        for msg in self.message_history:
             role = "user" if msg["role"] == "user" else "model"
             prompt_parts.append(f"<|{role}|>\n{msg['content']}")

        prompt_parts.append("<|user|>")
        if self.last_image:
            prompt_parts.append("<image>")
        prompt_parts.append(text)
        prompt_parts.append("<|model|>")

        return "\n".join(prompt_parts)


    def _update_history(self, user_text: str, assistant_response: str):
        """Update message history."""
        # Basic check to avoid adding empty/error responses to history
        if user_text and assistant_response and "sorry, an error occurred" not in assistant_response.lower():
            self.message_history.append({"role": "user", "content": user_text})
            self.message_history.append({"role": "assistant", "content": assistant_response})

            if len(self.message_history) > MAX_HISTORY_MESSAGES * 2:
                self.message_history = self.message_history[-(MAX_HISTORY_MESSAGES * 2):]
            logger.debug(f"History updated. Length: {len(self.message_history)}")
        else:
             logger.debug(f"Skipping history update due to empty or error response.")


    async def generate_streaming(self, text: str):
        """Generate response token by token, yielding text chunks."""
        self.generation_count += 1
        start_time = time.monotonic()
        logger.info(f"Gemma generation #{self.generation_count} starting for: '{text[:50]}...'")

        async with self.image_lock:
            current_image = self.last_image

        if not current_image:
            logger.warning("No image available for multimodal generation.")
            # Optionally yield a message indicating no image context?
            # yield "I don't have an image context right now. "

        # --- Prepare inputs ---
        try:
            prompt = self._build_prompt(text)
            inputs = self.processor(text=prompt, images=current_image, return_tensors="pt").to(self.model.device, dtype=self.torch_dtype)

            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=5 # Timeout for streamer iteration
            )

            generation_kwargs = dict(
                inputs.to(self.model.device), # Ensure inputs are on the correct device again just before generate
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                streamer=streamer,
            )

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True)
            thread.start()
            logger.info(f"Gemma generation thread started.")

        except Exception as e:
            logger.error(f"Gemma input preparation/thread start error: {e}", exc_info=True)
            yield f"Sorry, I encountered an error preparing your request."
            return

        # --- Stream output ---
        accumulated_text = ""
        chunk_buffer = ""
        try:
            # Use asyncio.to_thread if available (Python 3.9+) for better async integration with iterator
            # Otherwise, iterate directly (might block slightly more)
            async for new_text in self._iterate_streamer_async(streamer):
                chunk_buffer += new_text
                accumulated_text += new_text

                # Sentence splitting logic (same as before)
                parts = re.split(r'([.!?]+["\']?\s+|\n\n)', chunk_buffer)
                processed_chunk = False
                for i in range(0, len(parts) - 1, 2):
                    sentence = parts[i] + (parts[i+1] if i + 1 < len(parts) else '')
                    sentence = sentence.strip()
                    if sentence:
                        #logger.debug(f"Yielding sentence chunk: '{sentence}'")
                        yield sentence
                        processed_chunk = True

                # Update buffer with the remaining part
                if processed_chunk:
                     chunk_buffer = parts[-1] if len(parts) % 2 == 1 else ""
                # If no sentence end found but buffer is getting long, yield it
                elif len(chunk_buffer) > 80: # Yield if chunk gets long without punctuation
                     logger.debug("Yielding long chunk without sentence end.")
                     yield chunk_buffer
                     chunk_buffer = ""


            if chunk_buffer.strip():
                #logger.debug(f"Yielding final chunk: '{chunk_buffer.strip()}'")
                yield chunk_buffer.strip()

            duration = time.monotonic() - start_time
            logger.info(f"Gemma generation #{self.generation_count} completed ({duration:.2f}s). Total length: {len(accumulated_text)}")
            self._update_history(text, accumulated_text)

        except Exception as e:
            logger.error(f"Gemma streaming error: {e}", exc_info=True)
            yield "Sorry, an error occurred while generating the response."
        finally:
             if thread.is_alive():
                 # No forceful join needed usually with TextIteratorStreamer if handled correctly
                 # streamer should signal end when thread finishes.
                 logger.debug("Gemma generation thread assumed finished with streamer.")
                 pass


    async def _iterate_streamer_async(self, streamer: TextIteratorStreamer):
        """Helper to iterate the streamer in an async context without blocking the event loop excessively."""
        while True:
            try:
                # Use run_in_executor to get the next item without blocking
                next_item = await asyncio.get_event_loop().run_in_executor(
                    None,  # Use default executor
                    lambda: next(streamer, None) # Use None as sentinel for end
                )
                if next_item is None: # Stream finished
                    break
                yield next_item
            except StopIteration: # Should be caught by next(streamer, None)
                 break
            # Add a small sleep to prevent tight loop if streamer is slow/empty
            await asyncio.sleep(0.005)


# ==============================
# Kokoro TTS Processor
# ==============================
class KokoroTTSProcessor:
    """Handles text-to-speech conversion using Kokoro model."""
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        logger.info("Initializing Kokoro TTS Processor...")
        try:
            self.pipeline = KPipeline(lang_code=KOKORO_LANG)
            self.default_voice = KOKORO_VOICE
            # --- FIX: Assume Kokoro's sample rate ---
            self.native_sample_rate = KOKORO_INTERNAL_SAMPLE_RATE
            # --- Store the target sample rate for resampling ---
            self.target_sample_rate = AUDIO_SAMPLE_RATE

            logger.info(f"Kokoro TTS initialized. Voice: {self.default_voice}, Native SR: {self.native_sample_rate}Hz, Target SR: {self.target_sample_rate}Hz")
            self.synthesis_count = 0
        except Exception as e:
            # --- FIX: Log error correctly, assign None to pipeline, store fallback SR ---
            logger.error(f"FATAL: Error initializing Kokoro TTS: {e}", exc_info=True)
            self.pipeline = None
            self.native_sample_rate = KOKORO_INTERNAL_SAMPLE_RATE # Still store assumed rate
            self.target_sample_rate = AUDIO_SAMPLE_RATE
            raise # Propagate error

    async def synthesize_speech_streaming(self, text_chunk: str):
        """Synthesize a chunk of text and resample to the target rate."""
        if not text_chunk or not self.pipeline:
            logger.warning("TTS skipped: Empty text or pipeline unavailable.")
            return None

        self.synthesis_count += 1
        start_time = time.monotonic()
        # logger.info(f"Kokoro TTS #{self.synthesis_count} synthesizing chunk: '{text_chunk[:60]}...'")

        try:
            audio_segments = []
            # --- FIX: Use run_in_executor for blocking Kokoro call ---
            generator = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipeline(
                    text_chunk,
                    voice=self.default_voice,
                    speed=1.0,
                    split_pattern=None
                )
            )

            for _, _, audio_np in generator:
                if audio_np is not None and audio_np.size > 0:
                    audio_segments.append(audio_np.astype(np.float32)) # Ensure float32 for librosa

            if not audio_segments:
                logger.warning(f"Kokoro TTS #{self.synthesis_count} produced no audio for chunk.")
                return None

            combined_audio = np.concatenate(audio_segments)

            # --- FIX: Resample if necessary ---
            if self.native_sample_rate != self.target_sample_rate:
                # logger.debug(f"Resampling TTS audio from {self.native_sample_rate} Hz to {self.target_sample_rate} Hz")
                try:
                     # Use run_in_executor for potentially CPU-intensive resampling
                     resampled_audio = await asyncio.get_event_loop().run_in_executor(
                          None,
                          lambda: librosa.resample(
                               combined_audio,
                               orig_sr=self.native_sample_rate,
                               target_sr=self.target_sample_rate
                          )
                     )
                     final_audio = resampled_audio
                     output_sr = self.target_sample_rate
                except Exception as resample_err:
                     logger.error(f"Librosa resampling failed: {resample_err}. Sending audio at native rate {self.native_sample_rate}Hz.", exc_info=True)
                     final_audio = combined_audio # Fallback to original audio
                     output_sr = self.native_sample_rate

            else:
                final_audio = combined_audio
                output_sr = self.native_sample_rate

            # Convert to 16-bit PCM bytes
            audio_bytes = (final_audio * 32767).astype(np.int16).tobytes()
            duration = time.monotonic() - start_time
            logger.info(f"Kokoro TTS #{self.synthesis_count} synthesis complete ({duration:.2f}s). Samples: {len(final_audio)} @ {output_sr}Hz")
            return audio_bytes

        except Exception as e:
            logger.error(f"Kokoro TTS #{self.synthesis_count} synthesis error: {e}", exc_info=True)
            return None


# ==============================
# WebSocket Handler State & Logic (No changes needed below this line from previous version)
# ==============================

class ClientState:
    """Manages state for a single client connection."""
    def __init__(self):
        self.tts_playing = False
        self.tts_lock = asyncio.Lock() # Lock for tts_playing flag and task cancellation
        self.current_processing_task = None # Stores the asyncio.Task for the current process_speech_segment call
        self.client_id = os.urandom(4).hex() # Unique ID for logging
        logger.info(f"[{self.client_id}] New client state created.")

    async def set_tts_playing(self, is_playing: bool):
        async with self.tts_lock:
            # Only log if state changes to reduce noise
            if self.tts_playing != is_playing:
                 logger.info(f"[{self.client_id}] TTS playing state set to: {is_playing}")
                 self.tts_playing = is_playing


    async def is_tts_playing(self) -> bool:
        return self.tts_playing

    async def cancel_current_task(self):
        """Cancels the currently active processing task if it exists."""
        async with self.tts_lock:
            task_to_cancel = self.current_processing_task
            if task_to_cancel and not task_to_cancel.done():
                logger.info(f"[{self.client_id}] Requesting cancellation of processing task: {task_to_cancel.get_name()}")
                task_to_cancel.cancel()
                # Set task to None immediately within the lock
                self.current_processing_task = None
                self.tts_playing = False # Also reset TTS playing state on cancellation
                 # Await outside the lock to avoid holding it during potential long waits
                try:
                    await asyncio.wait_for(task_to_cancel, timeout=1.5) # Increased timeout slightly
                except asyncio.CancelledError:
                    logger.info(f"[{self.client_id}] Processing task cancelled successfully.")
                except asyncio.TimeoutError:
                    logger.warning(f"[{self.client_id}] Processing task did not finish cancellation within timeout.")
                except Exception as e:
                     logger.error(f"[{self.client_id}] Error waiting for cancelled task: {e}", exc_info=True)

            elif self.tts_playing:
                 # If no task exists but TTS is marked as playing, reset the state
                 logger.warning(f"[{self.client_id}] TTS marked as playing, but no active task found. Resetting state.")
                 self.tts_playing = False
                 self.current_processing_task = None


    async def set_current_task(self, task: asyncio.Task | None):
         async with self.tts_lock:
             self.current_processing_task = task
             # logger.debug(f"[{self.client_id}] Current processing task set (exists: {task is not None}).")


async def handle_client(websocket, path):
    """Handles a single WebSocket client connection."""
    state = ClientState()
    client_addr = websocket.remote_address
    logger.info(f"[{state.client_id}] Client connected from {client_addr}")

    try:
        detector = AudioSegmentDetector()
        transcriber = WhisperTranscriber.get_instance()
        gemma_processor = GemmaMultimodalProcessor.get_instance()
        tts_processor = KokoroTTSProcessor.get_instance()
        # Optional check: Ensure VAD sample rate matches TTS target rate after potential resampling
        if detector.sample_rate != tts_processor.target_sample_rate:
            logger.warning(f"[{state.client_id}] VAD SR ({detector.sample_rate}Hz) != TTS Target SR ({tts_processor.target_sample_rate}Hz). Ensure client expects {tts_processor.target_sample_rate}Hz audio.")

    except Exception as e:
         logger.error(f"[{state.client_id}] Failed to initialize processors: {e}. Closing connection.", exc_info=True)
         await websocket.close(code=1011, reason="Server initialization failed")
         return

    async def process_speech_segment(speech_segment: bytes):
        """Transcribe, generate, synthesize, and stream the response."""
        try:
            await state.set_tts_playing(True) # Mark TTS as potentially active

            transcription = await transcriber.transcribe(speech_segment)
            if not transcription:
                logger.info(f"[{state.client_id}] Empty or filtered transcription, skipping response.")
                return # Exit early

            full_response_text = ""
            async for text_chunk in gemma_processor.generate_streaming(transcription):
                # Check for cancellation before synthesis
                if asyncio.current_task().cancelled(): # More direct check
                     logger.info(f"[{state.client_id}] Processing task cancelled during generation stream.")
                     raise asyncio.CancelledError

                # Sanitize chunk slightly before TTS
                clean_chunk = text_chunk.replace("*", "").strip() # Remove common markdown like '*'
                if not clean_chunk:
                     continue

                full_response_text += text_chunk # Accumulate original text for history

                audio_bytes = await tts_processor.synthesize_speech_streaming(clean_chunk)

                if audio_bytes:
                    # Check for cancellation before sending
                    if asyncio.current_task().cancelled():
                        logger.info(f"[{state.client_id}] Processing task cancelled before sending audio chunk.")
                        raise asyncio.CancelledError

                    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                    try:
                        # Send audio and the corresponding text chunk
                        await websocket.send(json.dumps({"audio": base64_audio, "text_chunk": clean_chunk}))
                        # logger.debug(f"[{state.client_id}] Sent audio chunk ({len(audio_bytes)} bytes).")
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning(f"[{state.client_id}] Connection closed while sending audio chunk.")
                        raise # Re-raise to stop processing
                else:
                    logger.warning(f"[{state.client_id}] TTS failed for chunk: '{clean_chunk[:50]}...'")
                await asyncio.sleep(0.01) # Yield control


        except asyncio.CancelledError:
            logger.info(f"[{state.client_id}] Speech processing task was cancelled.")
        except websockets.exceptions.ConnectionClosed as e:
             logger.warning(f"[{state.client_id}] Connection closed during speech processing: {e}")
        except Exception as e:
            logger.error(f"[{state.client_id}] Error processing segment: {e}", exc_info=True)
            try:
                 await websocket.send(json.dumps({"error": "Sorry, an internal error occurred processing your request."}))
            except websockets.exceptions.ConnectionClosed:
                 pass
        finally:
            await state.set_tts_playing(False) # Ensure TTS state is reset
            # Task reference is cleared when a new task starts or on cancellation


    async def detect_speech_segments():
        """Listens for VAD segments and manages processing tasks."""
        while True:
            try:
                speech_segment = await detector.get_next_segment()
                if speech_segment:
                    logger.debug(f"[{state.client_id}] Speech segment pulled ({len(speech_segment)} bytes).")

                    # --- Interruption Logic ---
                    # Cancel any existing task *before* starting the new one
                    await state.cancel_current_task() # Handles lock internally

                    # --- Start New Processing Task ---
                    new_task = asyncio.create_task(
                        process_speech_segment(speech_segment),
                        name=f"process_segment_{state.client_id}_{detector.segments_detected_count}"
                        )
                    await state.set_current_task(new_task)
                else:
                     await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                 logger.info(f"[{state.client_id}] Speech detection loop cancelled.")
                 break
            except websockets.exceptions.ConnectionClosed:
                 logger.info(f"[{state.client_id}] Connection closed during speech detection.")
                 break
            except Exception as e:
                 logger.error(f"[{state.client_id}] Error in speech detection loop: {e}", exc_info=True)
                 await asyncio.sleep(1)


    async def receive_data():
        """Handles incoming WebSocket messages."""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    is_tts_active = await state.is_tts_playing() # Check TTS state once per message

                    # Simplified message handling
                    msg_type = data.get("type")
                    msg_data_b64 = data.get("data")

                    if msg_type == "audio" and msg_data_b64:
                        try:
                            audio_data = base64.b64decode(msg_data_b64)
                            await detector.add_audio(audio_data)
                        except (base64.binascii.Error, ValueError) as decode_err:
                            logger.warning(f"[{state.client_id}] Invalid base64 audio data: {decode_err}")
                    elif msg_type == "image" and msg_data_b64:
                        # logger.debug(f"[{state.client_id}] Received image message.")
                        if not is_tts_active:
                            try:
                                image_data = base64.b64decode(msg_data_b64)
                                await gemma_processor.set_image(image_data)
                            except (base64.binascii.Error, ValueError) as decode_err:
                                logger.warning(f"[{state.client_id}] Invalid base64 image data: {decode_err}")
                            except Exception as img_err:
                                logger.error(f"[{state.client_id}] Error processing received image: {img_err}", exc_info=True)
                        else:
                            logger.info(f"[{state.client_id}] Image received but TTS active, update deferred.")
                    # Handle legacy format if necessary
                    elif "realtime_input" in data and "media_chunks" in data["realtime_input"]:
                         # logger.warning(f"[{state.client_id}] Processing legacy 'realtime_input'.") # Reduce noise
                         for chunk in data["realtime_input"]["media_chunks"]:
                              mime_type = chunk.get("mime_type")
                              chunk_data_b64 = chunk.get("data")
                              if not chunk_data_b64: continue
                              try:
                                  chunk_data = base64.b64decode(chunk_data_b64)
                                  if mime_type == "audio/pcm":
                                      await detector.add_audio(chunk_data)
                                  elif mime_type == "image/jpeg":
                                      if not is_tts_active:
                                           await gemma_processor.set_image(chunk_data)
                                      # else: logger.info(f"[{state.client_id}] Legacy image chunk deferred.") # Reduce noise
                              except Exception as legacy_err:
                                  logger.error(f"[{state.client_id}] Error legacy chunk ({mime_type}): {legacy_err}")

                    else:
                         # Log other message types if needed, or ignore
                         # logger.debug(f"[{state.client_id}] Received unhandled message structure: {list(data.keys())}")
                         pass


                except json.JSONDecodeError:
                    logger.warning(f"[{state.client_id}] Received non-JSON message.")
                except Exception as e:
                    logger.error(f"[{state.client_id}] Error processing received message: {e}", exc_info=True)

        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"[{state.client_id}] Client closed connection normally.")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"[{state.client_id}] Client connection closed with error: {e}")
        except asyncio.CancelledError:
             logger.info(f"[{state.client_id}] Data reception loop cancelled.")
        except Exception as e:
            logger.error(f"[{state.client_id}] Unexpected error in data reception loop: {e}", exc_info=True)


    async def send_keepalive():
        """Sends periodic pings."""
        while True:
            try:
                await websocket.ping()
                # logger.debug(f"[{state.client_id}] Sent ping.") # Reduce noise
                await asyncio.sleep(20)
            except asyncio.CancelledError:
                 logger.info(f"[{state.client_id}] Keepalive loop cancelled.")
                 break
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"[{state.client_id}] Connection closed, stopping keepalive.")
                break
            except Exception as e:
                 logger.error(f"[{state.client_id}] Keepalive error: {e}. Stopping.", exc_info=True)
                 break


    # --- Run background tasks ---
    detection_task = asyncio.create_task(detect_speech_segments(), name=f"detect_speech_{state.client_id}")
    reception_task = asyncio.create_task(receive_data(), name=f"receive_data_{state.client_id}")
    keepalive_task = asyncio.create_task(send_keepalive(), name=f"keepalive_{state.client_id}")

    tasks = [detection_task, reception_task, keepalive_task]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    # --- Cleanup ---
    logger.info(f"[{state.client_id}] First background task finished ({[t.get_name() for t in done]}). Initiating cleanup...")

    # Cancel remaining background tasks
    for task in pending:
        if not task.done():
             # logger.debug(f"[{state.client_id}] Cancelling pending background task: {task.get_name()}")
             task.cancel()

    # Cancel the current processing task, if any
    await state.cancel_current_task()

    # Await all tasks (including already done ones and newly cancelled ones) to ensure cleanup
    all_tasks = list(done) + list(pending)
    try:
         await asyncio.wait(all_tasks, timeout=5.0)
         logger.debug(f"[{state.client_id}] All background tasks finished or timed out.")
    except asyncio.TimeoutError:
         logger.warning(f"[{state.client_id}] Timeout waiting for background tasks to complete cleanup.")


    # Close websocket if not already closed
    if websocket.open:
        logger.debug(f"[{state.client_id}] Closing WebSocket connection.")
        await websocket.close(code=1000, reason="Client handler shutting down")

    logger.info(f"[{state.client_id}] Client handler finished cleanup for {client_addr}.")


# ==============================
# Main Server Function
# ==============================
async def main():
    """Initializes models and starts the WebSocket server."""
    try:
        logger.info("--- Initializing Models (this may take a while) ---")
        WhisperTranscriber.get_instance()
        GemmaMultimodalProcessor.get_instance()
        KokoroTTSProcessor.get_instance()
        logger.info("--- Models Initialized ---")
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize one or more models during startup: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Starting WebSocket server on 0.0.0.0:{WEBSOCKET_PORT}")

    try:
        # Add exception_handler for server-level errors
        async def server_exception_handler(exc: Exception):
             logger.error(f"Unhandled exception in WebSocket server: {exc}", exc_info=True)

        async with websockets.serve(
            handle_client,
            "0.0.0.0",
            WEBSOCKET_PORT,
            ping_interval=20,
            ping_timeout=30,
            close_timeout=10,
            max_size=MAX_MESSAGE_SIZE,
            # logger=websockets.logging.getLogger("websockets.server"), # More verbose websocket logging if needed
            # exception_handler=server_exception_handler # Handle server errors more gracefully
        ):
            logger.info(f"WebSocket server running and listening on ws://0.0.0.0:{WEBSOCKET_PORT}")
            await asyncio.Future() # Run forever
    except OSError as e:
         logger.error(f"Failed to start server, likely port {WEBSOCKET_PORT} is already in use or insufficient permissions: {e}")
    except Exception as e:
        logger.error(f"Server encountered an unrecoverable error during startup: {e}", exc_info=True)
    finally:
         logger.info("WebSocket server shutting down.")


if __name__ == "__main__":
    try:
        # Consider adding uvloop for potential performance boost if running on Linux
        # import uvloop
        # uvloop.install()
        # logger.info("Using uvloop event loop.")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt).")
