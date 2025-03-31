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
AUDIO_SAMPLE_RATE = 16000
WHISPER_MODEL_ID = "openai/whisper-large-v3" # Using large-v3 as requested (turbo version doesn't exist for seq2seq)
GEMMA_MODEL_ID = "google/gemma-3-4b-it"
KOKORO_VOICE = 'en_tina' # Example English voice, adjust if needed. Check Kokoro docs for available voices.
KOKORO_LANG = 'e'       # English language code for Kokoro

# --- Constants ---
PCM_CHUNK_SIZE = 1024 * 2 # How many bytes to process at once in VAD
IMAGE_RESIZE_FACTOR = 0.75
MAX_HISTORY_MESSAGES = 6 # Keep last 3 user/assistant pairs

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
                 sample_rate=AUDIO_SAMPLE_RATE,
                 energy_threshold=0.015, # Might need tuning
                 silence_duration=0.7,   # Reduced slightly
                 min_speech_duration=0.5, # Reduced slightly
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


    async def add_audio(self, audio_bytes: bytes):
        """Add audio data and detect speech segments."""
        async with self.lock:
            self.audio_buffer.extend(audio_bytes)

            # Process in chunks to avoid recalculating energy over the whole buffer repeatedly
            while len(self.audio_buffer) >= self.speech_start_idx + PCM_CHUNK_SIZE * self.bytes_per_sample:
                # Analyze the chunk just after the current speech start (or from the beginning if not speaking)
                current_chunk_start = self.speech_start_idx if self.is_speech_active else len(self.audio_buffer) - (PCM_CHUNK_SIZE * self.bytes_per_sample)
                current_chunk_end = current_chunk_start + (PCM_CHUNK_SIZE * self.bytes_per_sample)
                chunk_bytes = self.audio_buffer[current_chunk_start:current_chunk_end]

                if not chunk_bytes: # Should not happen with the while loop condition, but safety check
                     break

                audio_array = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                energy = np.sqrt(np.mean(audio_array**2)) if len(audio_array) > 0 else 0.0

                # --- Finite State Machine for Speech Detection ---
                if not self.is_speech_active:
                    if energy > self.energy_threshold:
                        # Speech start detected
                        self.is_speech_active = True
                        # Start speech slightly before the chunk that triggered it, clamping to buffer start
                        self.speech_start_idx = max(0, current_chunk_start - int(0.1 * self.sample_rate * self.bytes_per_sample)) # backtrack slightly
                        self.silence_counter = 0
                        logger.info(f"Speech START detected (energy: {energy:.4f})")
                    else:
                         # Keep shifting start index forward if buffer grows while silent
                         self.speech_start_idx = len(self.audio_buffer) - (PCM_CHUNK_SIZE * self.bytes_per_sample)

                else: # is_speech_active == True
                    current_speech_len_bytes = len(self.audio_buffer) - self.speech_start_idx

                    if energy > self.energy_threshold:
                        # Continued speech - reset silence counter
                        self.silence_counter = 0
                    else:
                        # Potential end of speech - increment silence counter
                        self.silence_counter += len(audio_array) # Count samples, not bytes

                        if self.silence_counter >= self.silence_samples:
                             # Silence duration exceeded - finalize segment
                            speech_end_idx = len(self.audio_buffer) - (self.silence_counter * self.bytes_per_sample) # Adjust end index based on samples
                            await self._finalize_segment(self.speech_start_idx, speech_end_idx, "Silence")
                            # Reset state AFTER finalizing
                            self.is_speech_active = False
                            self.silence_counter = 0
                            # Important: Trim buffer *after* finalizing segment
                            self.audio_buffer = self.audio_buffer[speech_end_idx:]
                            self.speech_start_idx = 0 # Reset relative start index for next detection
                            # Break inner loop as buffer was modified
                            break


                    # Check for max duration even if not silent
                    if self.is_speech_active and current_speech_len_bytes >= self.max_speech_samples * self.bytes_per_sample:
                        logger.warning(f"Max duration reached ({self.max_speech_duration}s). Finalizing segment.")
                        # Finalize at max duration
                        speech_end_idx = self.speech_start_idx + self.max_speech_samples * self.bytes_per_sample
                        await self._finalize_segment(self.speech_start_idx, speech_end_idx, "Max Duration")
                        # Reset state AFTER finalizing
                        # Keep is_speech_active=True (could be continuing speech)
                        self.silence_counter = 0
                        # Important: Update buffer start index *after* finalizing
                        # Don't trim the buffer here, just advance the start index
                        # The next loop iteration will process the remaining audio
                        self.speech_start_idx = speech_end_idx
                        # Break inner loop as state changed significantly
                        break

            # Keep buffer size manageable if no speech is active for a long time
            if not self.is_speech_active and len(self.audio_buffer) > self.max_speech_samples * self.bytes_per_sample * 2:
                 keep_bytes = self.silence_samples * self.bytes_per_sample # Keep silence duration worth of audio
                 self.audio_buffer = self.audio_buffer[-keep_bytes:]
                 self.speech_start_idx = 0


    async def _finalize_segment(self, start_idx, end_idx, reason=""):
        """Extracts the segment, validates, and puts it on the queue."""
        speech_segment_bytes = bytes(self.audio_buffer[start_idx:end_idx])
        segment_len_samples = len(speech_segment_bytes) // self.bytes_per_sample

        if segment_len_samples >= self.min_speech_samples:
            self.segments_detected_count += 1
            duration = segment_len_samples / self.sample_rate
            logger.info(f"Speech segment DETECTED ({reason}): {duration:.2f}s (Total: {self.segments_detected_count})")
            await self.segment_queue.put(speech_segment_bytes)
        else:
            logger.info(f"Speech segment IGNORED (too short: {segment_len_samples / self.sample_rate:.2f}s < {self.min_speech_duration:.2f}s)")


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

        logger.info(f"Loading Whisper model: {WHISPER_MODEL_ID}...")
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                WHISPER_MODEL_ID,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation=attn_implementation # Use Flash Attention if available
            )
            self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30, # Process audio in chunks
                batch_size=16,     # Batch inference for potential speedup
                return_timestamps=False, # Don't need timestamps for this use case
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
            if len(audio_bytes) < self.processor.feature_extractor.hop_length * self.bytes_per_sample: # Need at least one frame
                logger.warning("Transcription skipped: Audio segment too short.")
                return ""

            # Convert PCM bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Run transcription in executor thread
            result = await asyncio.get_event_loop().run_in_executor(
                None,  # Use default executor
                lambda: self.pipe(
                    audio_array,
                    generate_kwargs={"language": "english", "task": "transcribe", "temperature": 0.0}
                )
            )

            text = result.get("text", "").strip()
            duration = time.monotonic() - start_time
            logger.info(f"Transcription #{self.transcription_count} ({duration:.2f}s): '{text}'")

            # Basic filtering
            if not text or not any(c.isalnum() for c in text):
                logger.info("Transcription invalid (empty or punctuation only).")
                return ""
            # Filter very short utterances that are likely noise or common fillers
            words = text.lower().split()
            if len(words) <= 1 and words[0] in ["okay", "yes", "no", "yeah", "nah", "bye", "uh", "um", "hmm", "thanks", "thank you"]:
                 logger.info(f"Skipping single-word filler: '{text}'")
                 return ""
            # Simple profanity filter example (can be expanded)
            if "fuck" in text.lower() or "shit" in text.lower():
                 logger.warning(f"Potential profanity detected in transcription: '{text}'. Replacing.")
                 return "Sorry, I cannot process that phrase."


            return text

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return "" # Return empty string on error

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
        # Gemma generally prefers bfloat16 if available on GPU
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        logger.info(f"Gemma using device: {self.device}, dtype: {self.torch_dtype}")

        logger.info(f"Loading Gemma model: {GEMMA_MODEL_ID}...")
        try:
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                GEMMA_MODEL_ID,
                device_map="auto", # Let HF determine device placement
                # load_in_8bit=True, # 8-bit can sometimes be slower, let's try bf16/fp16 first
                torch_dtype=self.torch_dtype,
                attn_implementation=attn_implementation # Use Flash Attention if available

            )
            self.processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)
            logger.info("Gemma model ready.")

            self.last_image = None
            self.last_image_timestamp = 0
            self.image_lock = asyncio.Lock()
            self.message_history = [] # Store as {"role": role, "content": content} list
            self.generation_count = 0

        except Exception as e:
            logger.error(f"FATAL: Failed to load Gemma model: {e}", exc_info=True)
            raise

    async def set_image(self, image_data: bytes):
        """Cache the most recent image, resizing it."""
        async with self.image_lock:
            try:
                img_pil = Image.open(io.BytesIO(image_data)).convert("RGB") # Ensure RGB

                # Resize
                if IMAGE_RESIZE_FACTOR != 1.0:
                    new_size = (int(img_pil.width * IMAGE_RESIZE_FACTOR), int(img_pil.height * IMAGE_RESIZE_FACTOR))
                    img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)
                    logger.info(f"Resized image to {new_size}")

                self.last_image = img_pil
                self.last_image_timestamp = time.monotonic()
                # Clear history when a new image is explicitly set? Maybe not, allow conversation continuation.
                # self.message_history = []
                # logger.info("New image set, conversation history cleared.")
                logger.info("New image set.")
                return True
            except Exception as e:
                logger.error(f"Error processing image: {e}", exc_info=True)
                self.last_image = None # Clear image on error
                return False

    def _build_prompt(self, text: str) -> str:
        """Builds the prompt string including history and image context (if available)."""
        # System Prompt (adjust as needed)
        system_prompt = """You are a helpful, conversational AI assistant.
Engage naturally with the user. If the user asks about the image provided, describe it concisely.
If the user's query isn't about the image, respond conversationally without forcing the image description.
Keep responses brief (1-3 sentences), clear, and suitable for spoken delivery. Avoid lists or complex formatting.
If the user's request is unclear, ask for clarification politely.
"""
        # Start with the system prompt
        prompt_parts = [f"<|system|>\n{system_prompt}"]

        # Add conversation history
        for msg in self.message_history:
             role = "user" if msg["role"] == "user" else "model" # map to gemma roles
             prompt_parts.append(f"<|{role}|>\n{msg['content']}")

        # Add current user query (including image placeholder if image exists)
        prompt_parts.append("<|user|>")
        if self.last_image:
            prompt_parts.append("<image>") # Placeholder for image
        prompt_parts.append(text)

        # Add the prompt for the model to start generating
        prompt_parts.append("<|model|>") # Signal for model to respond

        return "\n".join(prompt_parts)


    def _update_history(self, user_text: str, assistant_response: str):
        """Update message history, keeping it within limits."""
        self.message_history.append({"role": "user", "content": user_text})
        self.message_history.append({"role": "assistant", "content": assistant_response})

        # Trim history
        if len(self.message_history) > MAX_HISTORY_MESSAGES * 2: # *2 because each turn has user+assistant
            self.message_history = self.message_history[-(MAX_HISTORY_MESSAGES * 2):]
        logger.debug(f"History updated. Length: {len(self.message_history)}")

    async def generate_streaming(self, text: str):
        """Generate response token by token using the latest image and text, yielding text chunks."""
        self.generation_count += 1
        start_time = time.monotonic()
        logger.info(f"Gemma generation #{self.generation_count} starting for: '{text[:50]}...'")

        async with self.image_lock: # Ensure image doesn't change during processing
            current_image = self.last_image # Use the image available at the start

        if not current_image:
            logger.warning("No image available for multimodal generation. Proceeding text-only.")
            # Handle text-only generation if needed, or return a specific message
            # For now, let's allow it but the prompt won't have <image>
            # yield "I don't have an image to look at right now. " # Prepend clarification

        # --- Prepare inputs ---
        try:
            prompt = self._build_prompt(text)
            # Note: Pass the image separately if using processor. V3 uses text prompt with <image> placeholder.
            inputs = self.processor(text=prompt, images=current_image, return_tensors="pt").to(self.model.device, dtype=self.torch_dtype)
            input_len = inputs["input_ids"].shape[-1]

            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = dict(
                **inputs,
                max_new_tokens=150, # Increased slightly
                do_sample=True,
                temperature=0.7,
                top_p=0.9,          # Add top_p sampling
                # use_cache=True,   # Cache is default
                streamer=streamer,
            )

            # Run generation in a separate thread as it's blocking
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True)
            thread.start()

            logger.info(f"Gemma generation thread started.")

        except Exception as e:
            logger.error(f"Gemma input preparation error: {e}", exc_info=True)
            yield f"Sorry, I encountered an error preparing your request." # Yield error message
            return # Stop generation


        # --- Stream output ---
        accumulated_text = ""
        sentence_end_pattern = re.compile(r'[.!?]\s*$') # End of sentence marker
        chunk_buffer = ""

        try:
            for new_text in streamer:
                chunk_buffer += new_text
                accumulated_text += new_text

                # Try to yield complete sentences or clauses for better TTS chunking
                # Split on sentence-ending punctuation followed by space, or double newline
                parts = re.split(r'([.!?]+["\']?\s+|\n\n)', chunk_buffer)

                # Process complete parts (sentence + delimiter)
                for i in range(0, len(parts) - 1, 2): # Iterate in steps of 2 (part + delimiter)
                    sentence = parts[i] + (parts[i+1] if i + 1 < len(parts) else '')
                    sentence = sentence.strip()
                    if sentence:
                        logger.debug(f"Yielding sentence chunk: '{sentence}'")
                        yield sentence
                        chunk_buffer = "" # Reset buffer after yielding

                # If the last part is not empty, it's an incomplete sentence, keep it in buffer
                if len(parts) % 2 == 1 and parts[-1]:
                     chunk_buffer = parts[-1]
                else: # If split resulted in empty last part, clear buffer
                     chunk_buffer = ""


            # Yield any remaining text in the buffer after the stream ends
            if chunk_buffer.strip():
                logger.debug(f"Yielding final chunk: '{chunk_buffer.strip()}'")
                yield chunk_buffer.strip()

            duration = time.monotonic() - start_time
            logger.info(f"Gemma generation #{self.generation_count} completed ({duration:.2f}s). Total length: {len(accumulated_text)}")

            # Update history AFTER successful generation of the full response
            self._update_history(text, accumulated_text)

        except Exception as e:
            logger.error(f"Gemma streaming error: {e}", exc_info=True)
            yield "Sorry, an error occurred while generating the response." # Yield error message
        finally:
             # Ensure thread finishes, though streamer should handle this
             if thread.is_alive():
                 # This might be risky if generation is stuck, but necessary
                 # Ideally, streamer completion means thread is done. Add timeout?
                 thread.join(timeout=1.0)
                 if thread.is_alive():
                     logger.warning("Gemma generation thread did not exit cleanly after streaming.")


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
            # Initialize Kokoro TTS pipeline
            self.pipeline = KPipeline(lang_code=KOKORO_LANG) # Use configured lang code
            self.default_voice = KOKORO_VOICE
            self.sample_rate = self.pipeline.hps.data.sampling_rate # Get SR from Kokoro
            logger.info(f"Kokoro TTS processor initialized successfully. Voice: {self.default_voice}, Sample Rate: {self.sample_rate}")
            self.synthesis_count = 0
        except Exception as e:
            logger.error(f"FATAL: Error initializing Kokoro TTS: {e}", exc_info=True)
            self.pipeline = None
            self.sample_rate = AUDIO_SAMPLE_RATE # Fallback SR
            raise # Propagate error

    async def synthesize_speech_streaming(self, text_chunk: str):
        """Synthesize a chunk of text (e.g., a sentence) to audio bytes."""
        if not text_chunk or not self.pipeline:
            logger.warning("TTS skipped: Empty text or pipeline unavailable.")
            return None

        self.synthesis_count += 1
        start_time = time.monotonic()
        logger.info(f"Kokoro TTS #{self.synthesis_count} synthesizing chunk: '{text_chunk[:60]}...'")

        try:
            # Use run_in_executor for the potentially blocking TTS call
            # Kokoro's pipeline might yield multiple internal segments for a single input text chunk.
            # We need to concatenate them for the given chunk.
            audio_segments = []
            generator = await asyncio.get_event_loop().run_in_executor(
                None, # Use default executor
                lambda: self.pipeline(
                    text_chunk,
                    voice=self.default_voice,
                    speed=1.0,
                     # Minimal splitting within the chunk itself, as we assume chunks are meaningful units
                     split_pattern=None
                )
            )

            # Collect all audio parts generated for this chunk
            for _, _, audio_np in generator:
                if audio_np is not None and audio_np.size > 0:
                    audio_segments.append(audio_np)

            if not audio_segments:
                logger.warning(f"Kokoro TTS #{self.synthesis_count} produced no audio for chunk.")
                return None

            # Combine segments and convert to bytes
            combined_audio = np.concatenate(audio_segments)
            audio_bytes = (combined_audio * 32767).astype(np.int16).tobytes()
            duration = time.monotonic() - start_time
            logger.info(f"Kokoro TTS #{self.synthesis_count} synthesis complete ({duration:.2f}s). Samples: {len(combined_audio)}")
            return audio_bytes

        except Exception as e:
            logger.error(f"Kokoro TTS #{self.synthesis_count} synthesis error: {e}", exc_info=True)
            return None

# ==============================
# WebSocket Handler State
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
            self.tts_playing = is_playing
            logger.debug(f"[{self.client_id}] TTS playing state set to: {is_playing}")

    async def is_tts_playing(self) -> bool:
         # Don't necessarily need lock for read, but good practice if writes happen often
         # async with self.tts_lock:
        return self.tts_playing

    async def cancel_current_task(self):
        """Cancels the currently active processing task if it exists."""
        async with self.tts_lock:
            if self.current_processing_task and not self.current_processing_task.done():
                logger.info(f"[{self.client_id}] Requesting cancellation of current processing task.")
                self.current_processing_task.cancel()
                try:
                    # Give the task a moment to handle cancellation
                    await asyncio.wait_for(self.current_processing_task, timeout=1.0)
                except asyncio.CancelledError:
                    logger.info(f"[{self.client_id}] Current processing task cancelled successfully.")
                except asyncio.TimeoutError:
                    logger.warning(f"[{self.client_id}] Current processing task did not cancel within timeout.")
                except Exception as e:
                     # Log other potential errors during cancellation wait
                     logger.error(f"[{self.client_id}] Error waiting for cancelled task: {e}", exc_info=True)
                finally:
                    # Ensure it's marked as None even if wait failed
                    self.current_processing_task = None
                    self.tts_playing = False # Force TTS off after cancellation attempt
            else:
                 logger.debug(f"[{self.client_id}] No active processing task to cancel.")
                 self.tts_playing = False # Ensure state consistency


    async def set_current_task(self, task: asyncio.Task | None):
         async with self.tts_lock:
             self.current_processing_task = task
             logger.debug(f"[{self.client_id}] Current processing task set (exists: {task is not None}).")


# ==============================
# WebSocket Client Handler
# ==============================
async def handle_client(websocket, path):
    """Handles a single WebSocket client connection."""
    state = ClientState()
    client_addr = websocket.remote_address
    logger.info(f"[{state.client_id}] Client connected from {client_addr}")

    # Get singleton instances of processors
    try:
        detector = AudioSegmentDetector() # New detector per client
        transcriber = WhisperTranscriber.get_instance()
        gemma_processor = GemmaMultimodalProcessor.get_instance()
        tts_processor = KokoroTTSProcessor.get_instance()
        if tts_processor.sample_rate != detector.sample_rate:
             logger.warning(f"Mismatch between TTS sample rate ({tts_processor.sample_rate}) and Detector/Whisper rate ({detector.sample_rate}). Ensure consistency.")

    except Exception as e:
         logger.error(f"[{state.client_id}] Failed to initialize processors: {e}. Closing connection.", exc_info=True)
         await websocket.close(code=1011, reason="Server initialization failed")
         return


    # --- Core Processing Logic ---
    async def process_speech_segment(speech_segment: bytes):
        """Transcribe, generate response, synthesize, and send audio."""
        try:
            await state.set_tts_playing(True) # Signal start of processing/potential TTS

            # 1. Transcribe
            transcription = await transcriber.transcribe(speech_segment)
            if not transcription:
                logger.info(f"[{state.client_id}] Empty or filtered transcription, skipping response generation.")
                return # Exit early if transcription failed or was filtered

            # 2. Generate Response (Streaming) & Synthesize/Send (Streaming)
            full_response_text = ""
            async for text_chunk in gemma_processor.generate_streaming(transcription):
                # Check for cancellation *before* synthesizing each chunk
                if state.current_processing_task and state.current_processing_task.cancelled():
                     logger.info(f"[{state.client_id}] Processing task cancelled during generation stream. Stopping.")
                     raise asyncio.CancelledError # Propagate cancellation

                full_response_text += text_chunk # Accumulate full text for history
                audio_bytes = await tts_processor.synthesize_speech_streaming(text_chunk)

                if audio_bytes:
                    # Check for cancellation *before* sending audio
                    if state.current_processing_task and state.current_processing_task.cancelled():
                        logger.info(f"[{state.client_id}] Processing task cancelled before sending audio chunk. Stopping.")
                        raise asyncio.CancelledError

                    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                    try:
                        await websocket.send(json.dumps({"audio": base64_audio, "text": text_chunk})) # Send text chunk too
                        logger.debug(f"[{state.client_id}] Sent audio chunk ({len(audio_bytes)} bytes).")
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning(f"[{state.client_id}] Connection closed while sending audio chunk.")
                        raise # Re-raise to stop processing
                else:
                    logger.warning(f"[{state.client_id}] TTS failed for chunk: '{text_chunk[:50]}...'")
                 # Small sleep to allow other tasks (like receiving audio) to run
                await asyncio.sleep(0.01)


            # Note: History update is now handled within generate_streaming after full response

        except asyncio.CancelledError:
            logger.info(f"[{state.client_id}] Speech processing task was cancelled.")
            # History update might be partial or non-existent if cancelled early. Gemma handles its own history update on completion.
        except websockets.exceptions.ConnectionClosed as e:
             logger.warning(f"[{state.client_id}] Connection closed during speech processing: {e}")
             # No need to send close frame, connection is already gone.
        except Exception as e:
            logger.error(f"[{state.client_id}] Error processing segment: {e}", exc_info=True)
            try:
                 # Send error message back to client if possible
                 await websocket.send(json.dumps({"error": "Sorry, an internal error occurred."}))
            except websockets.exceptions.ConnectionClosed:
                 pass # Ignore if connection is already closed
        finally:
            # IMPORTANT: Always ensure tts_playing is reset and task reference is cleared
            await state.set_tts_playing(False)
            await state.set_current_task(None) # Clear the task reference


    # --- Background Task: Detect Speech Segments ---
    async def detect_speech_segments():
        """Listens for VAD segments and triggers processing."""
        while True:
            try:
                speech_segment = await detector.get_next_segment()
                if speech_segment:
                    logger.debug(f"[{state.client_id}] Speech segment pulled from queue ({len(speech_segment)} bytes).")

                    # --- Interruption Logic ---
                    # Check if TTS was playing *before* processing the new segment
                    # Use lock to ensure atomicity of check and cancellation
                    async with state.tts_lock:
                        if state.tts_playing:
                            logger.info(f"[{state.client_id}] New speech detected while TTS playing. INTERRUPTING.")
                            # Cancel the *previous* task (which should be state.current_processing_task)
                            if state.current_processing_task and not state.current_processing_task.done():
                                state.current_processing_task.cancel()
                                # Don't await here, let the task handle cancellation in its own flow
                            # Ensure state reflects cancellation immediately within the lock
                            state.tts_playing = False
                            state.current_processing_task = None

                    # --- Start New Processing Task ---
                    # Create the new task and store its reference
                    new_task = asyncio.create_task(
                        process_speech_segment(speech_segment),
                        name=f"process_segment_{state.client_id}_{detector.segments_detected_count}"
                        )
                    await state.set_current_task(new_task) # Store reference for potential future cancellation
                else:
                     # No segment, brief sleep to yield control
                     await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                 logger.info(f"[{state.client_id}] Speech detection loop cancelled.")
                 break
            except websockets.exceptions.ConnectionClosed:
                 logger.info(f"[{state.client_id}] Connection closed during speech detection loop.")
                 break # Exit loop if connection is closed
            except Exception as e:
                 logger.error(f"[{state.client_id}] Error in speech detection loop: {e}", exc_info=True)
                 await asyncio.sleep(1) # Avoid rapid spinning on unexpected errors

    # --- Background Task: Receive Audio and Images ---
    async def receive_data():
        """Handles incoming WebSocket messages (audio, images)."""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)

                    # Handle simple image message
                    if "image" in data:
                        image_data_b64 = data.get("image")
                        if image_data_b64:
                            logger.debug(f"[{state.client_id}] Received standalone image message.")
                            try:
                                image_data = base64.b64decode(image_data_b64)
                                # Set image only if TTS is not active to avoid interrupting speech for an image update
                                if not await state.is_tts_playing():
                                     await gemma_processor.set_image(image_data)
                                else:
                                     logger.info(f"[{state.client_id}] Image received but TTS is active, skipping update for now.")
                            except (base64.binascii.Error, ValueError) as decode_err:
                                logger.warning(f"[{state.client_id}] Received invalid base64 image data: {decode_err}")
                            except Exception as img_err:
                                logger.error(f"[{state.client_id}] Error processing received image: {img_err}", exc_info=True)
                        continue # Process next message

                    # Handle potential chunked data (if needed, adapt format)
                    # Example structure: {"type": "audio", "data": "base64..."}
                    # Example structure: {"type": "image", "data": "base64..."}
                    if "type" in data and "data" in data:
                         msg_type = data["type"]
                         msg_data_b64 = data["data"]

                         if msg_type == "audio":
                              try:
                                   audio_data = base64.b64decode(msg_data_b64)
                                   await detector.add_audio(audio_data)
                              except (base64.binascii.Error, ValueError) as decode_err:
                                   logger.warning(f"[{state.client_id}] Received invalid base64 audio data: {decode_err}")
                         elif msg_type == "image":
                              logger.debug(f"[{state.client_id}] Received chunked image message.")
                              try:
                                   image_data = base64.b64decode(msg_data_b64)
                                   if not await state.is_tts_playing():
                                        await gemma_processor.set_image(image_data)
                                   else:
                                        logger.info(f"[{state.client_id}] Image received but TTS is active, skipping update.")
                              except (base64.binascii.Error, ValueError) as decode_err:
                                   logger.warning(f"[{state.client_id}] Received invalid base64 image data: {decode_err}")
                              except Exception as img_err:
                                   logger.error(f"[{state.client_id}] Error processing received image: {img_err}", exc_info=True)
                         else:
                              logger.warning(f"[{state.client_id}] Received unknown message type: {msg_type}")

                    # Handle legacy `realtime_input` format (if still used by client)
                    elif "realtime_input" in data and "media_chunks" in data["realtime_input"]:
                        logger.warning(f"[{state.client_id}] Received legacy 'realtime_input' format. Consider updating client.")
                        for chunk in data["realtime_input"]["media_chunks"]:
                             mime_type = chunk.get("mime_type")
                             chunk_data_b64 = chunk.get("data")
                             if not chunk_data_b64: continue

                             try:
                                 chunk_data = base64.b64decode(chunk_data_b64)
                                 if mime_type == "audio/pcm":
                                     await detector.add_audio(chunk_data)
                                 elif mime_type == "image/jpeg":
                                     if not await state.is_tts_playing():
                                          await gemma_processor.set_image(chunk_data)
                                     else:
                                          logger.info(f"[{state.client_id}] Image received but TTS is active, skipping update.")
                             except (base64.binascii.Error, ValueError) as decode_err:
                                 logger.warning(f"[{state.client_id}] Invalid base64 in legacy chunk ({mime_type}): {decode_err}")
                             except Exception as legacy_err:
                                 logger.error(f"[{state.client_id}] Error processing legacy chunk ({mime_type}): {legacy_err}", exc_info=True)

                except json.JSONDecodeError:
                    logger.warning(f"[{state.client_id}] Received non-JSON message: {message[:100]}...") # Log beginning of message
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

    # --- Background Task: Keepalive ---
    async def send_keepalive():
        """Sends pings periodically to keep the connection alive."""
        while True:
            try:
                await websocket.ping()
                logger.debug(f"[{state.client_id}] Sent ping.")
                await asyncio.sleep(20)  # Send ping every 20 seconds
            except asyncio.CancelledError:
                 logger.info(f"[{state.client_id}] Keepalive loop cancelled.")
                 break
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"[{state.client_id}] Connection closed, stopping keepalive pings.")
                break # Exit loop if connection is closed
            except Exception as e:
                 logger.error(f"[{state.client_id}] Error in keepalive loop: {e}. Stopping pings.", exc_info=True)
                 break


    # --- Run background tasks ---
    detection_task = asyncio.create_task(detect_speech_segments(), name=f"detect_speech_{state.client_id}")
    reception_task = asyncio.create_task(receive_data(), name=f"receive_data_{state.client_id}")
    keepalive_task = asyncio.create_task(send_keepalive(), name=f"keepalive_{state.client_id}")

    done, pending = await asyncio.wait(
        [detection_task, reception_task, keepalive_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # --- Cleanup ---
    logger.info(f"[{state.client_id}] First background task finished. Initiating cleanup...")

    # Cancel pending tasks
    for task in pending:
        logger.debug(f"[{state.client_id}] Cancelling pending task: {task.get_name()}")
        task.cancel()

    # Cancel any processing task that might still be running
    await state.cancel_current_task()

    # Await cancelled tasks to allow cleanup
    if pending:
        await asyncio.wait(pending, timeout=5.0) # Give pending tasks time to finish cancellation

    # Ensure websocket is closed
    if not websocket.closed:
        await websocket.close(code=1000, reason="Server shutting down client handler")

    logger.info(f"[{state.client_id}] Client handler finished cleanup for {client_addr}.")


# ==============================
# Main Server Function
# ==============================
async def main():
    """Initializes models and starts the WebSocket server."""
    # Initialize models proactively on startup
    try:
        logger.info("--- Initializing Models (this may take a while) ---")
        WhisperTranscriber.get_instance()
        GemmaMultimodalProcessor.get_instance()
        KokoroTTSProcessor.get_instance()
        logger.info("--- Models Initialized ---")
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize one or more models: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Starting WebSocket server on 0.0.0.0:{WEBSOCKET_PORT}")

    # Set higher limits for message size if needed (e.g., for large images)
    # Adjust based on expected max image size + audio buffer
    MAX_MESSAGE_SIZE = 10 * 1024 * 1024 # 10 MB limit

    try:
        async with websockets.serve(
            handle_client,
            "0.0.0.0",
            WEBSOCKET_PORT,
            ping_interval=20,    # Send ping every 20 seconds
            ping_timeout=30,     # Wait up to 30 seconds for pong
            close_timeout=10,    # Wait up to 10 seconds for close handshake
            max_size=MAX_MESSAGE_SIZE, # Set max message size
            # compression=None # Consider enabling compression if bandwidth is an issue (websockets.Compression.default)
        ):
            logger.info(f"WebSocket server running and listening on ws://0.0.0.0:{WEBSOCKET_PORT}")
            await asyncio.Future()  # Run forever until interrupted
    except OSError as e:
         logger.error(f"Failed to start server, likely port {WEBSOCKET_PORT} is already in use: {e}")
    except Exception as e:
        logger.error(f"Server encountered an unrecoverable error: {e}", exc_info=True)
    finally:
         logger.info("WebSocket server shutting down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt).")
