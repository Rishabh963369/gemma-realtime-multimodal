import asyncio
import json
import websockets
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, GemmaForConditionalGeneration # Corrected Gemma model import
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

WHISPER_MODEL_ID = "openai/whisper-large-v3" # Changed back to large-v3 as requested (note potential speed impact)
GEMMA_MODEL_ID = "google/gemma-1.1-2b-it" # Using smaller Gemma 1.1 2B for potentially faster response
KOKORO_VOICE = 'en_us_sarah' # Assuming English voice is desired based on Whisper language

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
                self.current_generation_task.cancel()
                try:
                    await self.current_generation_task
                except asyncio.CancelledError:
                    logger.info("Generation task cancelled successfully.")
                except Exception as e:
                    logger.error(f"Error awaiting cancelled generation task: {e}")
                cancelled_gen = True
            self.current_generation_task = None # Clear reference

            cancelled_tts = False
            if self.current_tts_task and not self.current_tts_task.done():
                self.current_tts_task.cancel()
                try:
                    await self.current_tts_task
                except asyncio.CancelledError:
                    logger.info("TTS task cancelled successfully.")
                except Exception as e:
                    logger.error(f"Error awaiting cancelled TTS task: {e}")
                cancelled_tts = True
            self.current_tts_task = None # Clear reference

            if cancelled_gen or cancelled_tts:
                 logger.info("Ongoing tasks cancelled.")
            # TTS playing state will be reset by the caller or in the main loop finally block

    async def set_current_tasks(self, generation_task=None, tts_task=None):
        """Set current generation and TTS tasks safely"""
        async with self.task_lock:
            self.current_generation_task = generation_task
            self.current_tts_task = tts_task
            logger.debug(f"Set current tasks: Gen={generation_task is not None}, TTS={tts_task is not None}")

    async def add_audio(self, audio_bytes):
        """Add audio data, detect speech, and handle interruptions."""
        async with self.lock:
            self.audio_buffer.extend(audio_bytes)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_array) == 0:
                return None

            energy = np.sqrt(np.mean(audio_array**2))

            # --- Interruption Logic ---
            async with self.tts_lock:
                is_currently_playing = self.tts_playing

            if is_currently_playing and energy > self.energy_threshold / 2 and not self.interrupt_pending: # More sensitive threshold for interruption
                logger.info(f"Interrupt detected! Energy {energy:.4f} > threshold/2 while TTS playing.")
                self.interrupt_pending = True # Set flag to prevent spamming
                await self.cancel_current_tasks()
                # Send interrupt message immediately
                try:
                    interrupt_message = json.dumps({"interrupt": True})
                    await self.websocket.send(interrupt_message)
                    logger.info("Sent interrupt signal to client.")
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Connection closed, cannot send interrupt.")
                except Exception as e:
                    logger.error(f"Failed to send interrupt signal: {e}")
                # Don't reset tts_playing here; let the main loop handle it in finally
                # Do not process this audio chunk further for speech detection if interrupting
                return None
            # --- End Interruption Logic ---

            # --- Speech Detection Logic ---
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
                        # Ensure start index is not past end index
                        if self.speech_start_idx < speech_end_idx:
                            speech_segment = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])
                            segment_len_samples = len(speech_segment) // 2 # 2 bytes per sample (int16)

                            # Reset state
                            self.is_speech_active = False
                            self.silence_counter = 0
                            self.audio_buffer = self.audio_buffer[speech_end_idx:] # Trim buffer

                            if segment_len_samples >= self.min_speech_samples:
                                self.segments_detected += 1
                                logger.info(f"Speech segment detected: {segment_len_samples / self.sample_rate:.2f}s")
                                await self.segment_queue.put(speech_segment)
                                return speech_segment # Return immediately after queuing
                            else:
                                logger.info(f"Speech segment too short ({segment_len_samples / self.sample_rate:.2f}s), discarding.")
                        else:
                             # Reset state even if segment is invalid
                            logger.warning(f"Invalid segment indices: start={self.speech_start_idx}, end={speech_end_idx}. Resetting.")
                            self.is_speech_active = False
                            self.silence_counter = 0
                            self.audio_buffer = bytearray() # Clear buffer on error


                # Check max duration regardless of silence
                current_segment_len_bytes = len(self.audio_buffer) - self.speech_start_idx
                if self.is_speech_active and current_segment_len_bytes >= self.max_speech_samples * 2:
                    speech_segment = bytes(self.audio_buffer[self.speech_start_idx : self.speech_start_idx + self.max_speech_samples * 2])
                    # Keep the buffer, just move the start index for the *next* potential segment
                    self.speech_start_idx += self.max_speech_samples * 2
                    self.silence_counter = 0 # Reset silence counter as we forced a split

                    self.segments_detected += 1
                    logger.info(f"Max duration speech segment extracted: {self.max_speech_samples / self.sample_rate:.2f}s")
                    await self.segment_queue.put(speech_segment)
                    return speech_segment # Return immediately

            # Clean up buffer if it gets too large without speech detection
            max_buffer_samples = (self.max_speech_duration + self.silence_duration) * self.sample_rate * 5 # Allow larger buffer
            if not self.is_speech_active and len(self.audio_buffer) > max_buffer_samples * 2:
                 logger.warning(f"Audio buffer trimming (idle): {len(self.audio_buffer)} bytes")
                 keep_bytes = int(self.silence_samples * 2 * 1.5) # Keep a bit more than silence duration
                 self.audio_buffer = self.audio_buffer[-keep_bytes:]


            return None

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
            self.transcription_count += 1
            logger.info(f"Transcription #{self.transcription_count} ({(len(audio_array)/sample_rate):.2f}s audio -> {elapsed:.2f}s): '{text}'")

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
            # Try loading with bfloat16 if available on GPU, otherwise float32
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
            logger.info(f"Using dtype: {dtype} for Gemma")

            self.model = GemmaForConditionalGeneration.from_pretrained( # Corrected import
                GEMMA_MODEL_ID,
                device_map="auto", # Automatically distribute across available GPUs/CPU
                torch_dtype=dtype,
                # load_in_8bit=True, # Keep 8-bit for memory saving if needed, but bfloat16 might be faster
                # attn_implementation="flash_attention_2" # Optional
            )
            self.processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)
            logger.info(f"Gemma model loaded in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to load Gemma model: {e}", exc_info=True)
            raise

        self.last_image = None
        self.last_image_timestamp = 0
        self.image_lock = asyncio.Lock() # Separate lock for image updates

        self.message_history = []
        self.max_history_len_tokens = 1024 # Limit history by tokens instead of messages
        self.history_lock = asyncio.Lock() # Separate lock for history

        self.generation_count = 0

    async def set_image(self, image_data):
        """Cache the most recent image received, resizing it."""
        async with self.image_lock:
            try:
                image = Image.open(io.BytesIO(image_data)).convert("RGB") # Ensure RGB

                # Optional: Resize for faster processing / less VRAM, adjust as needed
                max_size = (1024, 1024)
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

    async def _build_prompt(self, text):
        """Build the prompt string including history and the latest image."""
        async with self.image_lock:
            current_image = self.last_image

        if not current_image:
             # If no image, handle as text-only conversation
             async with self.history_lock:
                 # Simplified text-only history for now
                history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.message_history])
                prompt = f"<start_of_turn>system\nYou are a helpful conversational assistant. Respond concisely and naturally.<end_of_turn>\n"
                if history_text:
                     prompt += history_text + "\n"
                prompt += f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
             return prompt, None # No image


        # Build multimodal prompt
        async with self.history_lock:
            # Convert history to Gemma format "<role>\n<content><end_of_turn>"
            # Naive token counting (approximation)
            current_token_count = 0
            history_for_prompt = []
            for msg in reversed(self.message_history):
                 # Simple estimate: len(text) / 4 ≈ tokens
                 msg_tokens = len(msg.get("content", "")) // 4
                 if current_token_count + msg_tokens < self.max_history_len_tokens:
                      history_for_prompt.append(f"<start_of_turn>{msg['role']}\n{msg['content']}<end_of_turn>")
                      current_token_count += msg_tokens
                 else:
                      break
            history_text = "\n".join(reversed(history_for_prompt))

        # Construct the final prompt for the model (Gemma 1.1 format)
        # Note: Image handling might require specific tokens or formatting depending on the exact processor/model version.
        # This is a general structure. The AutoProcessor should handle the details.
        # The image is passed separately to the processor usually.
        prompt = f"<start_of_turn>system\nYou are a helpful assistant analyzing an image and conversing. Keep responses concise (1-3 sentences) and conversational.<end_of_turn>\n"
        if history_text:
            prompt += history_text + "\n"
        prompt += f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n" # The processor expects the image separately

        return prompt, current_image


    async def _update_history(self, user_text, assistant_response):
        """Update message history, trimming old messages based on token count."""
        async with self.history_lock:
            self.message_history.append({"role": "user", "content": user_text})
            self.message_history.append({"role": "assistant", "content": assistant_response})

            # Trim history based on approximate token count
            total_tokens = 0
            trimmed_history = []
            for msg in reversed(self.message_history):
                 msg_tokens = len(msg.get("content", "")) // 4
                 if total_tokens + msg_tokens < self.max_history_len_tokens:
                     trimmed_history.append(msg)
                     total_tokens += msg_tokens
                 else:
                     break # Stop adding older messages
            self.message_history = list(reversed(trimmed_history))
            logger.debug(f"History updated. Approx tokens: {total_tokens}, Messages: {len(self.message_history)}")


    async def generate(self, text):
        """Generate a response using the latest image and text input (non-streaming)."""
        prompt_text, image_input = await self._build_prompt(text)

        if not prompt_text: # Should not happen with current logic but safe check
             return "Error: Could not build prompt."

        try:
            start_time = time.time()

            # Prepare inputs using the processor
            # The processor handles combining text and image appropriately for the model
            inputs = self.processor(text=prompt_text, images=image_input, return_tensors="pt").to(self.model.device, self.model.dtype)

            # Generate response
            # Use run_in_executor for the blocking model call
            generate_ids = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate(
                    **inputs,
                    max_new_tokens=100, # Shorter max length for faster response
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id # Set pad_token_id
                )
            )

            # Decode the generated tokens, excluding the input prompt tokens
            # Slicing ensures we only decode the newly generated part
            output_ids = generate_ids[0][inputs['input_ids'].shape[1]:]
            generated_text = self.processor.decode(output_ids, skip_special_tokens=True).strip()

            elapsed = time.time() - start_time
            self.generation_count += 1

            # Update conversation history
            await self._update_history(text, generated_text)

            logger.info(f"Gemma generation #{self.generation_count} ({elapsed:.2f}s): '{generated_text[:100]}...'")
            return generated_text

        except Exception as e:
            logger.error(f"Gemma generation error: {e}", exc_info=True)
            # Attempt to clear cache on error? Might help if it's an OOM.
            if self.device == "cuda:0":
                torch.cuda.empty_cache()
                logger.warning("Cleared CUDA cache after Gemma error.")
            return f"Sorry, I encountered an error while generating a response."


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
            # Use 'a' for automatic language detection if mixing, or specify 'en' etc.
            self.pipeline = KPipeline(lang_code='a') # Auto-detect language
            self.default_voice = KOKORO_VOICE # Use constant
            logger.info(f"Kokoro TTS processor initialized in {time.time() - start_time:.2f}s.")
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
            logger.info(f"Synthesizing speech for text: '{text[:50]}...'")
            start_time = time.time()
            audio_segments = []

            # Use run_in_executor for the blocking TTS call
            # The generator itself might do I/O or CPU work
            generator = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipeline(
                    text,
                    voice=self.default_voice,
                    speed=1.0, # Default speed
                    # Split reasonably for natural pauses, adjust pattern as needed
                    split_pattern=r'[.!?]+|[,\uff0c\u3002\uff1f\uff01]+' # English and common CJK punctuation
                )
            )

            # Consume the generator (which yields audio chunks)
            # This part might still involve some waiting if Kokoro processes chunks sequentially
            for _, _, audio_chunk in generator:
                 if audio_chunk is not None and len(audio_chunk) > 0:
                      audio_segments.append(audio_chunk)


            if not audio_segments:
                 logger.warning(f"TTS produced no audio for text: '{text[:50]}...'")
                 return None

            combined_audio = np.concatenate(audio_segments)
            elapsed = time.time() - start_time
            self.synthesis_count += 1
            logger.info(f"Speech synthesis #{self.synthesis_count} complete ({elapsed:.2f}s): {len(combined_audio)/self.pipeline.sr:.2f}s audio")
            # Ensure output is float32 between -1 and 1
            if combined_audio.dtype != np.float32:
                 combined_audio = combined_audio.astype(np.float32)
                 # Normalize if necessary (assuming Kokoro might output int16 sometimes)
                 max_val = np.max(np.abs(combined_audio))
                 if max_val > 1.0:
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

    try:
        # Receive initial configuration (optional, depends on client)
        # await websocket.recv() # Uncomment if client sends config first

        # Initialize components specific to this client connection
        detector = AudioSegmentDetector(websocket=websocket) # Pass websocket for interrupts
        transcriber = await WhisperTranscriber.get_instance()
        gemma_processor = await GemmaMultimodalProcessor.get_instance()
        tts_processor = await KokoroTTSProcessor.get_instance()

        # --- Main Processing Loop Task ---
        async def process_speech_queue():
            while True:
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
                    filler_patterns = [r'^(um+|uh+|ah+|oh+|hm+|mhm+|hmm+|huh)$', r'^(okay|yes|no|yeah|nah|bye|hi|hello)$']
                    if any(re.fullmatch(pattern, transcription.lower()) for pattern in filler_patterns):
                        logger.info(f"Skipping likely filler phrase: '{transcription}'")
                        continue

                    logger.info(f"Processing valid transcription: '{transcription}'")

                    # --- Generation & TTS ---
                    generation_task = None
                    tts_task = None
                    try:
                        # Set TTS playing flag only *before* potentially long operations
                        await detector.set_tts_playing(True)

                        # Create and store generation task
                        generation_task = asyncio.create_task(gemma_processor.generate(transcription))
                        await detector.set_current_tasks(generation_task=generation_task)
                        generated_text = await generation_task
                        await detector.set_current_tasks(generation_task=None) # Clear gen task after completion

                        if generated_text:
                            # Create and store TTS task
                            tts_task = asyncio.create_task(tts_processor.synthesize_speech(generated_text))
                            await detector.set_current_tasks(tts_task=tts_task)
                            synthesized_audio = await tts_task
                            await detector.set_current_tasks(tts_task=None) # Clear tts task after completion

                            if synthesized_audio is not None and len(synthesized_audio) > 0:
                                # Convert float32 audio back to int16 bytes
                                audio_int16 = (synthesized_audio * 32767).astype(np.int16)
                                audio_bytes = audio_int16.tobytes()
                                base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

                                # Send audio response
                                await websocket.send(json.dumps({"audio": base64_audio}))
                                logger.info(f"Sent {len(audio_bytes)/ (SAMPLE_RATE*2):.2f}s audio response to client.")
                            else:
                                 logger.warning("TTS synthesis produced no audio, skipping send.")
                        else:
                             logger.warning("Gemma generation produced no text, skipping TTS.")

                    except asyncio.CancelledError:
                        logger.info("Processing task (Gen/TTS) was cancelled due to interruption.")
                        # The cancel_current_tasks method already handles awaiting
                        # TTS state will be reset in the finally block
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("Connection closed during generation/TTS sending.")
                        break # Exit loop if connection is closed
                    except Exception as e:
                        logger.error(f"Error during generation/TTS processing: {e}", exc_info=True)
                        # Attempt to send an error message?
                        try:
                             await websocket.send(json.dumps({"error": "Processing failed"}))
                        except Exception:
                             pass # Ignore send errors if connection is broken
                    finally:
                        # Crucial: Always reset the playing flag and clear tasks if we exit processing
                        await detector.set_tts_playing(False)
                        # Clear tasks just in case they weren't cleared properly on exit/error
                        async with detector.task_lock:
                             detector.current_generation_task = None
                             detector.current_tts_task = None


                except asyncio.CancelledError:
                     logger.info("Speech processing queue task cancelled.")
                     break # Exit loop if task is cancelled externally
                except Exception as e:
                    logger.error(f"Error in speech processing loop: {e}", exc_info=True)
                    await asyncio.sleep(1) # Avoid tight loop on unexpected errors

        # --- Input Receiving Task ---
        async def receive_input():
            async for message in websocket:
                try:
                    data = json.loads(message)

                    # Handle audio chunks for VAD
                    if "realtime_input" in data:
                        for chunk in data["realtime_input"]["media_chunks"]:
                            if chunk["mime_type"] == "audio/pcm":
                                audio_data = base64.b64decode(chunk["data"])
                                if detector:
                                     await detector.add_audio(audio_data)
                            elif chunk["mime_type"] == "image/jpeg":
                                 image_data = base64.b64decode(chunk["data"])
                                 if gemma_processor:
                                     await gemma_processor.set_image(image_data) # Update image context

                    # Handle standalone image (less common for real-time)
                    elif "image" in data:
                        image_data = base64.b64decode(data["image"])
                        if gemma_processor:
                            await gemma_processor.set_image(image_data)

                    # Handle client confirmation that TTS playback finished (optional)
                    elif "tts_finished" in data and data["tts_finished"]:
                         logger.info("Client indicated TTS finished.")
                         if detector:
                             # We already set tts_playing to False after sending audio,
                             # but this confirms client acknowledges it.
                             # Could potentially reset interrupt_pending here if needed.
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
        processing_task = asyncio.create_task(process_speech_queue())
        receiving_task = asyncio.create_task(receive_input())

        # Wait for either task to complete (e.g., due to connection close)
        done, pending = await asyncio.wait(
            {processing_task, receiving_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel any remaining tasks
        for task in pending:
            logger.info(f"Cancelling pending task: {task.get_name()}")
            task.cancel()
            try:
                await task # Allow cancellation to propagate
            except asyncio.CancelledError:
                 pass # Expected
            except Exception as e:
                 logger.error(f"Error awaiting cancelled task {task.get_name()}: {e}")

        logger.info(f"Finished waiting for tasks for client {client_id}.")


    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Client {client_id} disconnected: {e.code} {e.reason}")
    except Exception as e:
        logger.error(f"Unhandled error in handle_client for {client_id}: {e}", exc_info=True)
    finally:
        logger.info(f"Cleaning up resources for client {client_id}.")
        # Ensure any lingering tasks associated with detector are cancelled
        if detector:
            logger.info(f"Performing final cancellation check for detector tasks of client {client_id}")
            await detector.cancel_current_tasks()
            await detector.set_tts_playing(False) # Ensure flag is off
        # Singletons are not cleaned up here, they persist for the server lifetime

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
    # Increase open file limit if necessary: ulimit -n <number>
    try:
        async with websockets.serve(
            handle_client,
            host,
            port,
            ping_interval=20,    # Keep connection alive
            ping_timeout=60,     # Timeout for pong response
            close_timeout=10,    # Timeout for closing handshake
            max_size=2**22,      # Allow larger messages (e.g., 4MB, adjust as needed for images/long audio)
            # read_limit=2**22,    # Limit for single frame read
            # write_limit=2**22    # Limit for single frame write
        ):
            logger.info(f"WebSocket server running on ws://{host}:{port}")
            await asyncio.Future()  # Run forever until interrupted

    except OSError as e:
         logger.critical(f"Failed to start server, possibly port {port} is already in use: {e}")
    except Exception as e:
        logger.critical(f"Server encountered critical error: {e}", exc_info=True)

if __name__ == "__main__":
    # Consider adding argument parsing here for config if needed
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
