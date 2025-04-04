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
            if generation_task is not None or (generation_task is None and remaining_text_task is None): # Clear if both are None
                 self.current_generation_task = generation_task
            if remaining_text_task is not None:
                self.current_remaining_text_task = remaining_text_task
            if tts_task is not None or tts_task is None: # Clear if explicitly None
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
                                self.audio_buffer = self.audio_buffer[trim_index:]
                                self.is_speech_active = False
                                self.silence_counter = 0
                                # Don't return anything, just reset and continue

                        # Check if speech segment exceeds maximum duration (regardless of silence)
                        elif (len(self.audio_buffer) - self.speech_start_idx) >= self.max_speech_samples * 2:
                            segment_end_idx = self.speech_start_idx + self.max_speech_samples * 2
                            segment_to_process = bytes(self.audio_buffer[self.speech_start_idx : segment_end_idx])

                            # Update start index for the *next* potential segment immediately after this one
                            self.speech_start_idx = segment_end_idx
                            # Don't reset silence counter or is_speech_active here, as speech might be continuing
                            # Don't trim the buffer here, as the next segment starts immediately

                            logger.info(f"Max duration speech segment detected: {len(segment_to_process)/2/self.sample_rate:.2f}s")
                            # reset_speech_state_after_processing remains False

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
                        else:
                             logger.warning("Reset state requested but trim_index was invalid.")
                        # Reset VAD state
                        self.is_speech_active = False
                        self.silence_counter = 0

                    return segment_to_process # Indicate a segment was processed and queued

            return None # No segment processed in this call

    async def get_next_segment(self):
        """Get the next available speech segment from the queue."""
        try:
            # Use a small timeout to prevent blocking indefinitely if the queue is empty
            return await asyncio.wait_for(self.segment_queue.get(), timeout=0.1)
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
                use_safetensors=True
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
            logger.error(f"FATAL: Failed to load Whisper model: {e}")
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
                    # Return timestamps can be useful for debugging, but adds overhead
                    # return_timestamps=False,
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
                # Quantization can sometimes cause issues or slow down inference depending on the setup
                # Try without 8-bit first for stability. Enable if VRAM is tight.
                # load_in_8bit=True,
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

    def _build_messages(self, text):
        """Build messages array with history for the model"""
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history (make sure it's serializable if needed later)
        messages.extend(self.message_history)

        # Add current user message with image (if available)
        content = []
        if self.last_image:
             # Use the processor's expected format if it differs
             # Assuming PIL Image is directly usable here based on Gemma usage
             content.append({"type": "image"}) # Placeholder, actual image added in processor call
        content.append({"type": "text", "text": text})

        messages.append({"role": "user", "content": content})

        return messages

    def _update_history(self, user_text, assistant_response):
        """Update message history with new exchange, trimming old turns."""
        # Add user message (text only for history)
        self.message_history.append({"role": "user", "content": user_text}) # Store text only

        # Add assistant response
        self.message_history.append({"role": "assistant", "content": assistant_response})

        # Trim history: Keep last `max_history_turns` exchanges (user + assistant pairs)
        max_messages = self.max_history_turns * 2
        if len(self.message_history) > max_messages:
            self.message_history = self.message_history[-max_messages:]
            logger.debug(f"Trimmed message history to {len(self.message_history)} messages.")

    async def generate_streaming(self, text):
        """Generate a response using the latest image and text input with streaming."""
        async with self.lock: # Protect image and history during processing
            if not self.last_image:
                logger.warning("No image available for multimodal generation, responding text-only.")
                # Fallback: Text-only response (or predefined message)
                # For simplicity, return None here, handle upstream.
                # Could implement a text-only Gemma call if needed.
                return None, f"Sorry, I don't have an image to look at right now. Can you describe what you want to talk about?"


            try:
                messages = self._build_messages(text)

                # Use processor to prepare inputs including the image
                # Note: PIL Image is passed directly here for Gemma processor
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True, # Important for Gemma-IT
                    tokenize=True,
                    return_tensors="pt",
                    # Pass the actual image here
                    images=self.last_image
                ).to(self.model.device)

                # input_len = inputs["input_ids"].shape[-1] # Needed if not skipping prompt in streamer

                streamer = TextIteratorStreamer(
                    self.processor.tokenizer,
                    skip_prompt=True, # Don't stream the input prompt
                    skip_special_tokens=True
                )

                generation_kwargs = dict(
                    inputs, # Pass processed inputs directly
                    streamer=streamer,
                    max_new_tokens=150, # Max length of the generated response
                    do_sample=True,    # Enable sampling
                    temperature=0.7,   # Control randomness (lower = more deterministic)
                    top_p=0.9,         # Nucleus sampling
                    eos_token_id=self.processor.tokenizer.eos_token_id # Ensure generation stops correctly
                )

                # Run model.generate in a separate thread to avoid blocking asyncio loop
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start() # Generation starts here

                # Return the streamer immediately for the calling function to iterate
                self.generation_count += 1
                logger.info(f"Gemma generation #{self.generation_count} started for: '{text}'")

                # History update will happen *after* the full response is streamed
                return streamer, text # Return streamer and original user text for history

            except Exception as e:
                logger.error(f"Gemma streaming generation error: {e}", exc_info=True)
                # Return a user-friendly error message
                error_response = "Sorry, I encountered an error trying to process that."
                return None, error_response


    # This method might not be needed if streaming is always used, but keep for potential non-streaming use cases
    async def generate(self, text):
        """Generate a response using the latest image and text input (non-streaming)."""
        async with self.lock:
            if not self.last_image:
                 logger.warning("No image available for multimodal generation.")
                 return f"No image context available to answer: {text}"

            try:
                messages = self._build_messages(text)

                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    images=self.last_image
                ).to(self.model.device)

                input_len = inputs["input_ids"].shape[-1]

                generation = await asyncio.get_event_loop().run_in_executor(
                     None,
                     lambda: self.model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        eos_token_id=self.processor.tokenizer.eos_token_id
                    )
                )

                generated_tokens = generation[0][input_len:]
                generated_text = self.processor.decode(
                    generated_tokens,
                    skip_special_tokens=True
                ).strip()

                self._update_history(text, generated_text)
                self.generation_count += 1
                logger.info(f"Gemma generation #{self.generation_count} complete (non-streaming).")
                logger.debug(f"Generated Text: {generated_text}")

                return generated_text

            except Exception as e:
                logger.error(f"Gemma non-streaming generation error: {e}", exc_info=True)
                return f"Error processing: {text}"

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
        try:
            # Assuming 'en' is the code for English in Kokoro, adjust if needed
            self.pipeline = KPipeline(lang_code='en')
            # Find an English voice, 'en_script_1' is often a default/common one
            # List available voices if unsure: print(self.pipeline.backend.util.SUPPORTED_VOICES)
            self.default_voice = 'en_script_1' # Example, change if needed
            if self.default_voice not in self.pipeline.backend.util.SUPPORTED_VOICES:
                logger.warning(f"Voice '{self.default_voice}' not found, trying first available English voice.")
                # Fallback to the first available English voice
                available_en_voices = [v for v in self.pipeline.backend.util.SUPPORTED_VOICES if v.startswith('en_')]
                if available_en_voices:
                    self.default_voice = available_en_voices[0]
                    logger.info(f"Using fallback English voice: '{self.default_voice}'")
                else:
                    raise Exception("No suitable English voice found for Kokoro TTS.")

            logger.info(f"Kokoro TTS processor initialized successfully with voice '{self.default_voice}'")
            self.synthesis_count = 0
            self.sample_rate = self.pipeline.backend.sr # Get sample rate from Kokoro
            logger.info(f"Kokoro TTS sample rate: {self.sample_rate} Hz")

        except Exception as e:
            logger.error(f"FATAL: Error initializing Kokoro TTS: {e}", exc_info=True)
            self.pipeline = None
            raise # Stop server startup

    async def _synthesize_internal(self, text, split_pattern=None):
        """Internal synthesis logic shared by initial and remaining synthesis"""
        if not text or not self.pipeline:
            logger.warning("Skipping synthesis - empty text or no TTS pipeline.")
            return None

        try:
            # Kokoro's pipeline might be synchronous, run in executor
            # The generator pattern seems inherent to Kokoro, process it fully.
            audio_segments = []
            full_command = lambda: list(self.pipeline( # Use list() to consume the generator
                    text,
                    voice=self.default_voice,
                    speed=1.0, # Adjust speed if desired
                    split_pattern=split_pattern
                ))

            # Run the potentially blocking TTS generation in a thread
            generator_output = await asyncio.get_event_loop().run_in_executor(
                None, full_command
            )

            # Extract audio from the consumed generator results
            for _gs, _ps, audio_data in generator_output:
                if audio_data is not None and len(audio_data) > 0:
                    audio_segments.append(audio_data)
                else:
                     logger.warning("Kokoro generated an empty audio segment.")


            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                logger.info(f"Synthesis complete: generated {len(combined_audio)} samples ({len(combined_audio)/self.sample_rate:.2f}s)")
                return combined_audio
            else:
                logger.warning(f"Synthesis resulted in no audio data for text: '{text[:50]}...'")
                return None

        except Exception as e:
            # Log detailed error, including the text if possible (be mindful of PII)
            logger.error(f"Speech synthesis error for text '{text[:50]}...': {e}", exc_info=True)
            return None


    async def synthesize_initial_speech(self, text):
        """Convert initial text chunk to speech rapidly (minimal splitting)."""
        logger.info(f"Synthesizing initial speech for: '{text}'")
        # Less aggressive splitting for the first chunk to get audio out faster
        # Split only on major punctuation that indicates a pause.
        # Adjust pattern as needed based on typical LLM first chunk output.
        split_pattern = r'[.!?]+'
        # split_pattern = None # Option: No splitting for maximum speed on first chunk
        audio_data = await self._synthesize_internal(text, split_pattern=split_pattern)
        if audio_data is not None:
             self.synthesis_count += 1
             logger.info(f"Initial speech synthesis count: {self.synthesis_count}")
        return audio_data


    async def synthesize_remaining_speech(self, text):
        """Convert remaining text to speech with more natural splitting."""
        if not text: return None # Handle empty remaining text
        logger.info(f"Synthesizing remaining speech for: '{text[:50]}...'")
        # More comprehensive splitting for potentially longer remaining text
        split_pattern = r'[.!?,\-;:]+' # Split on more punctuation types
        audio_data = await self._synthesize_internal(text, split_pattern=split_pattern)
        # Don't increment count here, or adjust logic if needed
        # self.synthesis_count += 1
        return audio_data

# --- WebSocket Handler ---
async def handle_client(websocket):
    """Handles WebSocket client connection, orchestrating the components."""
    client_id = websocket.remote_address
    logger.info(f"Client connected: {client_id}")

    # Initialize per-connection state
    detector = AudioSegmentDetector() # Each client gets its own detector/state machine
    # Get singleton instances for models
    transcriber = WhisperTranscriber.get_instance()
    gemma_processor = GemmaMultimodalProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance()

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

            if streamer:
                try:
                    async for chunk in streamer: # Async iteration over TextIteratorStreamer
                        if chunk:
                            remaining_text_list.append(chunk)
                            # Optional: Synthesize smaller chunks of remaining text as they arrive?
                            # This adds complexity but could further reduce latency.
                            # For now, collect all remaining text first.

                except asyncio.CancelledError:
                    logger.info("Remaining text collection was cancelled.")
                    # Process whatever was collected before cancellation
                    pass # Allow finally block to handle synthesis/history
                except Exception as e:
                    logger.error(f"Error collecting remaining text: {e}", exc_info=True)
                    # Attempt to process what was collected
                    pass # Allow finally block to handle synthesis/history
                finally:
                    remaining_text = "".join(remaining_text_list)
                    complete_response += remaining_text
                    logger.info(f"Collected remaining text ({len(remaining_text)} chars). Full response len: {len(complete_response)} chars.")

                    # Update history with the *complete* response *now* that we have it all
                    # (or as much as we got before cancellation)
                    if complete_response and original_user_text:
                         await gemma_processor._update_history(original_user_text, complete_response) # Use await if _update_history becomes async


                    if remaining_text:
                        # Synthesize the collected remaining text
                        tts_remaining_task = asyncio.create_task(
                             tts_processor.synthesize_remaining_speech(remaining_text)
                        )
                        # Store this specific task handle
                        await detector.set_current_tasks(tts_task=tts_remaining_task) # Store remaining TTS task

                        try:
                            remaining_audio = await tts_remaining_task
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
                        finally:
                            # Clear only the TTS task handle after completion/cancellation
                             await detector.set_current_tasks(tts_task=None)


            # Send remaining audio if synthesized successfully
            if remaining_audio is not None:
                try:
                    audio_bytes = (remaining_audio * 32767).astype(np.int16).tobytes()
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

            # --- Final Cleanup after processing remaining text/audio ---
            # Signal that TTS *might* be fully done (initial + remaining)
            # This is tricky because the audio playback is client-side.
            # We set tts_playing = False here assuming the client starts playing immediately.
            # If there's a delay client-side, new speech might interrupt too early.
            # A more robust solution involves client-side acknowledgements.
            logger.info("Finished processing initial and remaining TTS flow.")
            await detector.set_tts_playing(False) # Assume TTS playback finishes after sending last chunk
            # Clear remaining text task handle specifically
            await detector.set_current_tasks(remaining_text_task=None)


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

                        # Basic filtering (adjust as needed)
                        cleaned_transcription = transcription.strip().lower()
                        if not any(c.isalnum() for c in cleaned_transcription):
                            logger.info(f"Skipping transcription with no alphanumeric chars: '{transcription}'")
                            continue

                        # Filter short/common fillers (customize list)
                        filler_patterns = [
                            r'^(um|uh|ah|oh|hm|mhm|hmm)$',
                            r'^(okay|ok|yes|no|yeah|nah|got it)$',
                            r'^(bye|goodbye)$'
                        ]
                        # Check word count AFTER potential cleaning
                        words = [w for w in cleaned_transcription.split() if any(c.isalnum() for c in w)]
                        if len(words) <= 1 and any(re.fullmatch(pattern, cleaned_transcription) for pattern in filler_patterns):
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

                        # 2. Mark TTS as potentially starting (blocks image updates) & Store Tasks
                        # This happens *before* awaiting generation/TTS
                        await detector.set_tts_playing(True)

                        # 3. Start LLM Generation (Streaming)
                        generation_stream_task = asyncio.create_task(
                             gemma_processor.generate_streaming(transcription)
                        )
                        # Store generation task handle immediately
                        await detector.set_current_tasks(generation_task=generation_stream_task)

                        try:
                            # Await the *start* of generation to get the streamer
                            streamer, user_text_for_history = await generation_stream_task
                            # Now the generation_stream_task itself is completed (it returned the streamer)
                            # We no longer need to track it directly. The streamer is the active part.
                            await detector.set_current_tasks(generation_task=None)

                            if streamer is None:
                                logger.warning(f"Generation failed or returned no streamer for '{transcription}'. User text: '{user_text_for_history}'")
                                # If generation failed, maybe send the error message back as TTS
                                if user_text_for_history != transcription: # Check if it's an error message
                                    error_audio_task = asyncio.create_task(tts_processor.synthesize_initial_speech(user_text_for_history))
                                    await detector.set_current_tasks(tts_task=error_audio_task)
                                    try:
                                        error_audio = await error_audio_task
                                        if error_audio is not None:
                                            audio_bytes = (error_audio * 32767).astype(np.int16).tobytes()
                                            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                                            await websocket.send(json.dumps({
                                                "audio": base64_audio,
                                                "sample_rate": tts_processor.sample_rate
                                            }))
                                            logger.info("Sent TTS for generation error message.")
                                        # Wait a bit before allowing new input after error TTS
                                        await asyncio.sleep(1)
                                    except asyncio.CancelledError:
                                        logger.info("Error TTS task was cancelled.")
                                    except Exception as e:
                                        logger.error(f"Failed to send error TTS: {e}")
                                    finally:
                                         await detector.set_current_tasks(tts_task=None) # Clear TTS task handle
                                # Reset TTS playing state as generation failed
                                await detector.set_tts_playing(False)
                                continue # Skip to next VAD segment

                            # 4. Process Initial Text from Streamer
                            initial_text = ""
                            min_chars_for_initial_tts = 35 # Chars to accumulate for first TTS chunk
                            sentence_end_pattern = re.compile(r'[.!?,\-;:]') # Include comma etc.
                            found_sentence_end = False

                            async for first_chunk in streamer: # Use async for here
                                initial_text += first_chunk
                                if sentence_end_pattern.search(first_chunk):
                                    found_sentence_end = True
                                # Break if we have a sentence end AND enough characters, OR just enough characters
                                if (found_sentence_end and len(initial_text) >= min_chars_for_initial_tts // 2) or \
                                   len(initial_text) >= min_chars_for_initial_tts:
                                    break
                                # Safety break if it generates a lot without punctuation
                                if len(initial_text) > min_chars_for_initial_tts * 3:
                                    logger.warning("Breaking initial text collection early due to length.")
                                    break

                            logger.info(f"Collected initial text ({len(initial_text)} chars): '{initial_text}'")

                            # 5. Synthesize and Send Initial Audio
                            if initial_text:
                                initial_tts_task = asyncio.create_task(
                                    tts_processor.synthesize_initial_speech(initial_text)
                                )
                                # Store the initial TTS task handle
                                await detector.set_current_tasks(tts_task=initial_tts_task)

                                try:
                                    initial_audio = await initial_tts_task
                                    if initial_audio is not None:
                                        audio_bytes = (initial_audio * 32767).astype(np.int16).tobytes()
                                        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                                        await websocket.send(json.dumps({
                                            "audio": base64_audio,
                                            "sample_rate": tts_processor.sample_rate
                                        }))
                                        logger.info(f"Sent initial audio chunk ({len(audio_bytes)} bytes)")

                                        # 6. Start Collecting Remaining Text & Audio IN PARALLEL
                                        # Pass the streamer, initial text, and original user query
                                        remaining_task = asyncio.create_task(
                                            collect_remaining_text_and_synthesize(streamer, initial_text, user_text_for_history)
                                        )
                                        # Store handle for *this specific task* for potential cancellation
                                        await detector.set_current_tasks(remaining_text_task=remaining_task)
                                        # Note: The initial TTS task handle is cleared implicitly now
                                        # Let this task run in the background. The main loop continues.

                                    else:
                                        logger.warning("Initial TTS synthesis returned None. Cannot proceed with this turn.")
                                        # Reset state as TTS failed
                                        await detector.set_tts_playing(False)
                                        await detector.set_current_tasks() # Clear all task handles

                                except asyncio.CancelledError:
                                    logger.info("Initial TTS task was cancelled (likely by new speech).")
                                    # State (_tts_playing=False) should have been handled by cancel_current_tasks
                                    # streamer might still contain text, but we discard it as the turn was interrupted.
                                    # Clear any remaining task handles just in case
                                    await detector.set_current_tasks()
                                    # History was not updated, which is correct.
                                except websockets.exceptions.ConnectionClosed:
                                    logger.warning("Connection closed while sending initial audio.")
                                    break # Exit loop
                                except Exception as e:
                                    logger.error(f"Error during initial TTS processing: {e}", exc_info=True)
                                    await detector.set_tts_playing(False) # Reset state on error
                                    await detector.set_current_tasks() # Clear task handles
                            else:
                                logger.warning("No initial text was collected from the streamer.")
                                # Reset state as we didn't get anything to speak
                                await detector.set_tts_playing(False)
                                await detector.set_current_tasks() # Clear task handles
                                # Update history with empty response? Or just log? Logged above.

                        except asyncio.CancelledError:
                             logger.info("LLM Generation task was cancelled before streamer obtained.")
                             # State should have been handled by cancel_current_tasks
                             await detector.set_current_tasks() # Clear task handles
                        except Exception as e:
                             logger.error(f"Error awaiting generation task/streamer: {e}", exc_info=True)
                             await detector.set_tts_playing(False) # Reset state on error
                             await detector.set_current_tasks() # Clear task handles

                    else:
                        # No segment detected, sleep briefly to yield control
                        await asyncio.sleep(0.02) # Adjust sleep time as needed

                except websockets.exceptions.ConnectionClosed:
                    logger.info("Connection closed during speech processing loop.")
                    break
                except Exception as e:
                    logger.error(f"Error in detect_speech_and_process loop: {e}", exc_info=True)
                    # Attempt to reset state on unexpected errors
                    try:
                        await detector.cancel_current_tasks()
                        await detector.set_tts_playing(False)
                        await detector.set_current_tasks()
                    except Exception as E:
                        logger.error(f"Error during error recovery: {E}")
                    await asyncio.sleep(1) # Pause briefly after an error

        async def receive_messages():
            """Receives audio and image messages from the client."""
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_timestamp = time.time()

                    # Handle image data (only if TTS is not active)
                    # Check tts_playing status *before* processing image
                    is_tts_playing = await detector.tts_playing
                    if not is_tts_playing:
                        if "image" in data:
                            logger.debug("Received standalone image frame.")
                            image_data = base64.b64decode(data["image"])
                            await gemma_processor.set_image(image_data)
                        elif "realtime_input" in data and "media_chunks" in data["realtime_input"]:
                             for chunk in data["realtime_input"]["media_chunks"]:
                                 if chunk.get("mime_type") == "image/jpeg":
                                     logger.debug("Received image chunk in realtime_input.")
                                     image_data = base64.b64decode(chunk["data"])
                                     await gemma_processor.set_image(image_data)
                                     break # Process only one image per message if multiple sent


                    # Handle audio data regardless of TTS state
                    if "realtime_input" in data and "media_chunks" in data["realtime_input"]:
                        for chunk in data["realtime_input"]["media_chunks"]:
                            if chunk.get("mime_type") == "audio/pcm":
                                # logger.debug("Received audio chunk.") # Can be very verbose
                                audio_data = base64.b64decode(chunk["data"])
                                await detector.add_audio(audio_data) # Add audio to VAD buffer

                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON message: {message[:100]}...")
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Connection closed while receiving messages.")
                    break
                except Exception as e:
                    logger.error(f"Error processing received message: {e}", exc_info=True)

        # --- Run Tasks Concurrently ---
        logger.info("Starting worker tasks for client...")
        speech_task = asyncio.create_task(detect_speech_and_process())
        receive_task = asyncio.create_task(receive_messages())
        keepalive_task = asyncio.create_task(send_keepalive())

        # Wait for tasks to complete (normally receive_task or speech_task exits on connection close)
        done, pending = await asyncio.wait(
            [speech_task, receive_task, keepalive_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
             logger.info(f"Cancelling pending task: {task.get_name()}")
             task.cancel()
             try:
                 await task # Await cancellation
             except asyncio.CancelledError:
                 pass
             except Exception as e:
                  logger.error(f"Error during task cancellation: {e}")

        logger.info(f"Finished waiting for tasks for client {client_id}.")

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"Client {client_id} disconnected normally.")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"Client {client_id} connection closed error: {e}")
    except Exception as e:
        logger.error(f"Unhandled error in handle_client for {client_id}: {e}", exc_info=True)
    finally:
        logger.info(f"Cleaning up resources for client {client_id}")
        # Ensure any lingering tasks associated with this client are cancelled
        if speech_task and not speech_task.done(): speech_task.cancel()
        if receive_task and not receive_task.done(): receive_task.cancel()
        if keepalive_task and not keepalive_task.done(): keepalive_task.cancel()
        # Explicitly cancel any detector tasks that might still be referenced
        await detector.cancel_current_tasks()
        logger.info(f"Client {client_id} handler finished.")


# --- Main Server ---
async def main():
    """Initializes models and starts the WebSocket server."""
    # Initialize models eagerly at startup
    logger.info("--- Initializing Models ---")
    try:
        WhisperTranscriber.get_instance()
        GemmaMultimodalProcessor.get_instance()
        KokoroTTSProcessor.get_instance()
        logger.info("--- All Models Initialized ---")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}. Server cannot start.", exc_info=True)
        sys.exit(1) # Exit if essential models fail to load


    # Start the WebSocket server
    host = "0.0.0.0"
    port = 9074
    logger.info(f"Starting WebSocket server on {host}:{port}")

    # Set higher limits for message size if needed (e.g., for large images)
    # Adjust max_size based on expected image/audio data size
    MAX_WEBSOCKET_MESSAGE_SIZE = 10 * 1024 * 1024 # 10 MB example

    try:
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
         logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True)
