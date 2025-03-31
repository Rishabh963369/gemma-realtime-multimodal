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
# Import Kokoro TTS library
from kokoro import KPipeline
import re
from threading import Thread

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s', # Added funcName
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000
ENERGY_THRESHOLD = 0.015
SILENCE_DURATION = 0.8
MIN_SPEECH_DURATION = 0.8
MAX_SPEECH_DURATION = 15
WEBSOCKET_PORT = 9073
PING_INTERVAL = 20
PING_TIMEOUT = 60
MAX_HISTORY_MESSAGES = 6 # Increased history slightly

# --- Removed the faulty singleton decorator function ---

class AudioSegmentDetector:
    """Detects speech segments based on audio energy levels and manages interaction flow"""
    # ... (Keep the AudioSegmentDetector class exactly as it was in the previous corrected version) ...
    def __init__(self,
                 sample_rate=SAMPLE_RATE,
                 energy_threshold=ENERGY_THRESHOLD,
                 silence_duration=SILENCE_DURATION,
                 min_speech_duration=MIN_SPEECH_DURATION,
                 max_speech_duration=MAX_SPEECH_DURATION):

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
        self.segment_queue = asyncio.Queue()
        self.buffer_lock = asyncio.Lock() # Lock specifically for audio_buffer access

        # Counters
        self.segments_detected = 0

        # TTS playback and generation control state
        self._tts_playing = False # Use property for safer access
        self._interrupt_requested = False
        self.state_lock = asyncio.Lock() # Lock for state variables (_tts_playing, _interrupt_requested)
        self.current_processing_task = None # Single task to manage the whole process cycle
        self.processing_task_lock = asyncio.Lock() # Lock for managing current_processing_task

    @property
    async def tts_playing(self):
        async with self.state_lock:
            return self._tts_playing

    async def set_tts_playing(self, is_playing: bool):
        async with self.state_lock:
            if self._tts_playing != is_playing:
                logger.info(f"Setting TTS playing state to: {is_playing}")
                self._tts_playing = is_playing

    async def request_interrupt(self):
        async with self.state_lock:
            if not self._interrupt_requested:
                logger.info("Interrupt requested.")
                self._interrupt_requested = True

    async def is_interrupt_requested(self):
        async with self.state_lock:
            return self._interrupt_requested

    async def clear_interrupt_request(self):
         async with self.state_lock:
            if self._interrupt_requested:
                logger.info("Clearing interrupt request.")
                self._interrupt_requested = False

    async def cancel_current_processing(self):
        """Cancel any ongoing generation and TTS task chain."""
        async with self.processing_task_lock:
            task_to_cancel = self.current_processing_task
            self.current_processing_task = None # Clear immediately to prevent race conditions
            if task_to_cancel and not task_to_cancel.done():
                logger.warning("Cancelling current processing task.")
                task_to_cancel.cancel()
                try:
                    # Give cancellation a moment to propagate and handle cleanup within the task
                    await asyncio.wait_for(task_to_cancel, timeout=0.5)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    logger.info("Current processing task cancelled or timed out during wait.")
                except Exception as e:
                    logger.error(f"Error waiting for cancelled task: {e}")

            # Ensure state is reset after cancellation attempt
            await self.set_tts_playing(False)
            await self.clear_interrupt_request() # Clear interrupt flag after cancellation

    async def set_current_processing_task(self, task):
        """Set the current main processing task."""
        async with self.processing_task_lock:
            # It's the caller's responsibility to ensure the previous task was handled/cancelled
            # This prevents potential double-cancellation issues.
            self.current_processing_task = task
            if task:
                 await self.set_tts_playing(True) # Mark as busy when a task starts

    async def add_audio(self, audio_bytes):
        """Add audio data and detect speech segments."""
        async with self.buffer_lock:
            self.audio_buffer.extend(audio_bytes)
            # Quick check if buffer is too large (e.g., > 30 seconds) - might indicate missed processing
            max_buffer_samples = 30 * self.sample_rate * 2 # 30 seconds * 16kHz * 16-bit
            if len(self.audio_buffer) > max_buffer_samples:
                logger.warning(f"Audio buffer growing large ({len(self.audio_buffer)} bytes). Trimming.")
                # Trim retaining only the last part roughly corresponding to max speech duration + silence
                keep_bytes = (self.max_speech_samples + self.silence_samples) * 2
                self.audio_buffer = self.audio_buffer[-keep_bytes:]
                # Reset detection state if buffer was trimmed drastically
                self.is_speech_active = False
                self.silence_counter = 0
                self.speech_start_idx = 0


            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_array) == 0:
                return

            # Use a small epsilon to avoid log(0) or sqrt(0) issues if audio is pure silence
            energy = np.sqrt(np.mean(audio_array**2) + 1e-10)

            segment_to_process = None

            # --- State Machine Logic ---
            if not self.is_speech_active:
                if energy > self.energy_threshold:
                    # Transition: Silence -> Speech
                    self.is_speech_active = True
                    # Estimate start slightly before the current chunk for better capture
                    self.speech_start_idx = max(0, len(self.audio_buffer) - len(audio_bytes) - int(0.1 * self.sample_rate * 2)) # Look back 100ms
                    self.silence_counter = 0
                    logger.info(f"Speech start detected (energy: {energy:.6f})")
                    if await self.tts_playing:
                        logger.info("New speech detected while TTS playing. Requesting interrupt.")
                        await self.request_interrupt() # Signal the main processing loop to cancel
            else: # self.is_speech_active == True
                current_speech_duration_bytes = len(self.audio_buffer) - self.speech_start_idx
                # Check Max Duration First
                if current_speech_duration_bytes >= self.max_speech_samples * 2:
                     # Transition: Speech -> Speech (Max Duration Cut)
                    speech_end_idx = self.speech_start_idx + self.max_speech_samples * 2
                    speech_segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])

                    # Reset start index for the *next* segment which continues immediately
                    self.speech_start_idx = speech_end_idx
                    self.silence_counter = 0 # Reset silence as speech continues

                    self.segments_detected += 1
                    logger.info(f"Speech segment [max duration]: {len(speech_segment_bytes)/2/self.sample_rate:.2f}s")
                    segment_to_process = speech_segment_bytes
                    # State remains is_speech_active = True

                # Check Energy for Silence
                elif energy <= self.energy_threshold:
                    # Accumulate Silence
                    self.silence_counter += len(audio_array)
                    if self.silence_counter >= self.silence_samples:
                         # Transition: Speech -> Silence (End of Segment)
                        # Calculate end index *before* the silence started
                        speech_end_idx = len(self.audio_buffer) - self.silence_counter
                        speech_segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])

                        # Reset state fully for next detection
                        self.is_speech_active = False
                        self.silence_counter = 0
                        # Trim buffer *after* extracting segment
                        self.audio_buffer = self.audio_buffer[speech_end_idx:]
                        self.speech_start_idx = 0 # Reset start index

                        if len(speech_segment_bytes) >= self.min_speech_samples * 2: # Check min duration
                            self.segments_detected += 1
                            logger.info(f"Speech segment [silence end]: {len(speech_segment_bytes)/2/self.sample_rate:.2f}s")
                            segment_to_process = speech_segment_bytes
                        else:
                            logger.info(f"Skipping short segment (silence end): {len(speech_segment_bytes)/2/self.sample_rate:.2f}s")
                        # State becomes is_speech_active = False
                else:
                     # Still Speech, Reset Silence Counter
                     self.silence_counter = 0
                     # State remains is_speech_active = True


            # --- Queue Segment ---
            if segment_to_process:
                 # Check again if TTS is playing *before* putting into queue
                 # This prevents queuing up segments if an interrupt was just requested
                if await self.tts_playing:
                    logger.info("Segment detected while TTS playing. Requesting interrupt (if not already).")
                    await self.request_interrupt()
                    # Don't queue the segment, let the interrupt handle the current process
                else:
                    # Only queue if not currently processing/playing TTS
                    logger.info("Queuing detected speech segment.")
                    try:
                        self.segment_queue.put_nowait(segment_to_process)
                    except asyncio.QueueFull:
                         logger.warning("Segment queue full. Discarding oldest segment.")
                         # Discard oldest and try again - prevents blocking detector
                         try:
                             self.segment_queue.get_nowait()
                             self.segment_queue.put_nowait(segment_to_process)
                         except asyncio.QueueEmpty: # Should not happen if full, but handle anyway
                             pass
                         except asyncio.QueueFull: # Still full? Log error, discard new.
                              logger.error("Segment queue remained full after discarding. Discarding new segment.")


    async def get_next_segment(self):
        """Get the next available speech segment non-blockingly."""
        try:
            return self.segment_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

# CORRECTED SINGLETON IMPLEMENTATION WITHIN CLASSES

class WhisperTranscriber:
    """Handles speech transcription using Whisper."""
    _instance = None

    @classmethod
    def get_instance(cls):
        """Gets the singleton instance of the WhisperTranscriber."""
        if cls._instance is None:
            logger.info("Creating WhisperTranscriber singleton instance.")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initializes the Whisper model and pipeline. Should only be called once."""
        # Check if instance already exists to prevent re-initialization (optional but good practice)
        # Note: This check makes the class not strictly reusable if you *wanted* multiple instances later,
        # but for a singleton pattern, it enforces the single instance idea.
        if hasattr(self.__class__, '_instance_initialized') and self.__class__._instance_initialized:
             return # Already initialized

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Whisper using device: {self.device}")
        self.torch_dtype = torch.float16 if self.device == "cuda:0" else torch.float32
        model_id = "openai/whisper-large-v3"
        logger.info(f"Loading Whisper model: {model_id}...")
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
                max_new_tokens=128, # Set max tokens in pipeline config
                chunk_length_s=30,  # Process in 30s chunks
                batch_size=16,      # Batching for potential speedup if needed
                return_timestamps=False, # Don't need timestamps here
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            logger.info("Whisper model ready.")
            self.transcription_count = 0
            self.__class__._instance_initialized = True # Mark as initialized
        except Exception as e:
            logger.critical(f"FATAL: Failed to load Whisper model: {e}", exc_info=True)
            # Set flag or state indicating failure? For now, rely on exception propagation.
            self.__class__._instance_initialized = False
            raise # Re-raise the exception to halt server startup

    async def transcribe(self, audio_bytes):
        """Transcribe audio bytes to text."""
        # Ensure model is ready before proceeding
        if not getattr(self.__class__, '_instance_initialized', False) or not self.pipe:
             logger.error("Whisper transcriber not initialized properly. Cannot transcribe.")
             return ""

        if len(audio_bytes) < 200: # Check length in bytes (100 samples * 2 bytes/sample)
            logger.warning("Skipping transcription for very short audio segment.")
            return ""
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            # logger.info(f"Starting transcription for segment of {len(audio_array)/SAMPLE_RATE:.2f}s...")
            start_time = time.time()

            # Run blocking pipeline call in executor
            # The pipeline itself handles chunking if audio is long
            result = await asyncio.get_event_loop().run_in_executor(
                None, # Default executor
                lambda: self.pipe(
                    audio_array, # Pass numpy array directly
                    # generate_kwargs can override pipeline defaults if needed
                    generate_kwargs={
                        "task": "transcribe",
                        "language": "english",
                        "temperature": 0.0 # Be deterministic
                    }
                )
            )
            text = result.get("text", "").strip()
            duration = time.time() - start_time
            self.transcription_count += 1
            logger.info(f"Transcription #{self.transcription_count} completed in {duration:.2f}s: '{text}'")
            return text
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return ""


class GemmaMultimodalProcessor:
    """Handles multimodal generation using Gemma."""
    _instance = None

    @classmethod
    def get_instance(cls):
        """Gets the singleton instance of the GemmaMultimodalProcessor."""
        if cls._instance is None:
            logger.info("Creating GemmaMultimodalProcessor singleton instance.")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initializes the Gemma model and processor. Should only be called once."""
        if hasattr(self.__class__, '_instance_initialized') and self.__class__._instance_initialized:
             return # Already initialized

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Gemma using device: {self.device}")
        model_id = "google/gemma-3-4b-it"
        logger.info(f"Loading Gemma model: {model_id}...")
        try:
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32
            logger.info(f"Gemma using dtype: {dtype}")

            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=dtype,
                attn_implementation="flash_attention_2" if dtype != torch.float32 and torch.cuda.is_available() else None # Check cuda avail again
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            logger.info("Gemma model ready.")
            self.last_image = None
            self.last_image_timestamp = 0
            self.image_lock = asyncio.Lock()
            self.message_history = []
            self.generation_count = 0
            self.__class__._instance_initialized = True
        except Exception as e:
            logger.critical(f"FATAL: Failed to load Gemma model: {e}", exc_info=True)
            self.__class__._instance_initialized = False
            raise # Re-raise the exception

    async def set_image(self, image_data):
        """Cache the most recent image, resizing it."""
        if not getattr(self.__class__, '_instance_initialized', False): return False # Don't process if not ready

        async with self.image_lock:
            try:
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                max_size = (1024, 1024)
                image.thumbnail(max_size, Image.Resampling.LANCZOS)

                logger.info(f"New image set (resized to {image.size}). Clearing conversation history.")
                self.last_image = image
                self.last_image_timestamp = time.time()
                self.message_history = [] # Reset history on new image
                return True
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                self.last_image = None
                return False

    def _build_prompt(self, text):
        """Builds the prompt string including history and system message."""
        # Check if instance is initialized before proceeding
        if not getattr(self.__class__, '_instance_initialized', False) or not self.processor:
             logger.error("Gemma processor not initialized. Cannot build prompt.")
             # Return a minimal prompt or raise an error, returning empty might be safer
             return ""

        system_prompt = """You are a helpful assistant describing images and chatting naturally. Keep responses concise (2-3 sentences), conversational, and directly related to the user's question about the image if applicable. If the question isn't about the image, respond normally. Avoid unnecessary introductions or closings."""

        chat_template_input = [{"role": "system", "content": system_prompt}]
        chat_template_input.extend(self.message_history)
        chat_template_input.append({"role": "user", "content": text})

        try:
            prompt_text = self.processor.tokenizer.apply_chat_template(
                chat_template_input,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt_text
        except Exception as e:
            logger.error(f"Error applying chat template: {e}", exc_info=True)
            # Fallback to simpler prompt if template fails
            return f"User: {text}\nAssistant:"


    async def generate_streaming(self, text):
        """Generate response stream using the latest image and text."""
         # Check initialization status
        if not getattr(self.__class__, '_instance_initialized', False) or not self.model or not self.processor:
            logger.error("Gemma model/processor not initialized. Cannot generate.")
            async def error_streamer(): yield "Error: Model not ready." # Create a simple async generator
            return error_streamer(), "Error: Model not ready."


        # Use a separate lock for generation to prevent concurrent calls if needed,
        # though the main control flow should prevent this.
        # async with self.generation_lock: # Optional: If strict single generation needed
        async with self.image_lock: # Ensure image doesn't change mid-generation
            if not self.last_image:
                logger.warning("No image available for generation.")
                async def fallback_streamer():
                    yield "Sorry, I don't have an image to look at right now."
                return fallback_streamer(), "Sorry, I don't have an image to look at right now."

            prompt = self._build_prompt(text)
            if not prompt: # Handle case where prompt building failed
                 logger.error("Prompt generation failed. Aborting generation.")
                 async def empty_streamer(): yield "" # Empty stream
                 return empty_streamer(), ""

            # Prepare inputs
            try:
                inputs = self.processor(
                    text=prompt,
                    images=self.last_image,
                    return_tensors="pt",
                ).to(self.model.device, self.model.dtype)
            except Exception as e:
                 logger.error(f"Error processing inputs for Gemma: {e}", exc_info=True)
                 async def error_streamer(): yield "Error preparing model input."
                 return error_streamer(), "Error preparing model input."


        logger.info(f"Starting Gemma generation for prompt: '{text[:50]}...'")
        start_time = time.time()

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        # Run generation in a separate thread
        # Ensure thread safety if models/processors have internal state issues,
        # but HF models are generally okay if inputs are distinct.
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()

        self.generation_count += 1
        # logger.info(f"Gemma generation #{self.generation_count} thread started.")

        # Return streamer and the original user text used for this turn's prompt
        return streamer, text

    def update_history(self, user_text, assistant_response):
        """Update message history safely."""
        # Basic validation
        if not isinstance(user_text, str) or not isinstance(assistant_response, str):
             logger.warning("Invalid type for history update. Skipping.")
             return

        if not user_text or not assistant_response:
             logger.warning("Empty user text or assistant response. Skipping history update.")
             return

        # Add user message (text only)
        self.message_history.append({"role": "user", "content": user_text})
        # Add assistant response
        self.message_history.append({"role": "assistant", "content": assistant_response})

        # Trim history
        while len(self.message_history) > MAX_HISTORY_MESSAGES * 2:
            self.message_history.pop(0) # Remove oldest message (FIFO)

        logger.info(f"History updated. Current length: {len(self.message_history)} messages.")


class KokoroTTSProcessor:
    """Handles text-to-speech conversion using Kokoro."""
    _instance = None

    @classmethod
    def get_instance(cls):
        """Gets the singleton instance of the KokoroTTSProcessor."""
        if cls._instance is None:
            logger.info("Creating KokoroTTSProcessor singleton instance.")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initializes the Kokoro TTS pipeline. Should only be called once."""
        if hasattr(self.__class__, '_instance_initialized') and self.__class__._instance_initialized:
            return # Already initialized

        logger.info("Initializing Kokoro TTS processor...")
        try:
            self.pipeline = KPipeline(lang_code='a') # Assuming 'a' is correct
            self.default_voice = 'af_sarah' # Assuming 'af_sarah' is valid for 'a'
            logger.info(f"Kokoro TTS processor initialized with voice '{self.default_voice}'.")
            self.synthesis_count = 0
            self.__class__._instance_initialized = True
        except NameError:
             logger.critical("FATAL: Kokoro TTS library (KPipeline) not found or imported.", exc_info=True)
             self.pipeline = None # Explicitly set pipeline to None
             self.__class__._instance_initialized = False
             # Don't raise here, allow server to potentially run without TTS
        except FileNotFoundError: # Example: If language model files are missing
             logger.critical("FATAL: Kokoro TTS model files not found.", exc_info=True)
             self.pipeline = None
             self.__class__._instance_initialized = False
        except Exception as e:
            logger.critical(f"FATAL: Failed to initialize Kokoro TTS: {e}", exc_info=True)
            self.pipeline = None
            self.__class__._instance_initialized = False

    async def synthesize_speech_stream(self, text_streamer):
        """Synthesize speech chunk by chunk as text arrives from a streamer."""
        # Check initialization status *and* if pipeline loaded successfully
        if not getattr(self.__class__, '_instance_initialized', False) or not self.pipeline:
            logger.error("Kokoro TTS pipeline not available or not initialized.")
            # Yield nothing to indicate failure to the caller
            return # Effectively yields nothing

        if not text_streamer:
             logger.warning("No text streamer provided for TTS.")
             return

        logger.info("Starting streaming TTS synthesis...")
        sentence_buffer = ""
        # More robust sentence splitting (handles more punctuation, keeps delimiter for context)
        sentence_split_pattern = re.compile(r'([.!?。！？]+["\']?\s*)')

        try:
            async for text_chunk in text_streamer:
                if text_chunk is None: break # Handle explicit stream termination signal
                if not isinstance(text_chunk, str): continue # Skip non-strings
                if not text_chunk: continue # Skip empty chunks

                sentence_buffer += text_chunk
                # logger.debug(f"TTS Buffer: '{sentence_buffer}'")

                # Process buffer for sentences
                parts = sentence_split_pattern.split(sentence_buffer)
                # Example: "Hello there. How are you?" -> ["Hello there", ". ", "How are you?"]
                # Need to combine delimiter back for processing

                processed_upto_index = 0
                # Iterate in pairs (text, delimiter)
                for i in range(0, len(parts) - 1, 2):
                     sentence = (parts[i] + parts[i+1]).strip() # Combine text + delimiter
                     if sentence:
                        # logger.debug(f"Synthesizing sentence: '{sentence}'")
                        start_time = time.time()
                        try:
                            # Use run_in_executor for the blocking Kokoro call
                            audio_generator = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda s=sentence: self.pipeline( # Use lambda default arg capture
                                    s,
                                    voice=self.default_voice,
                                    speed=1.0
                                )
                            )
                            sentence_audio_parts = []
                            for _, _, audio_part in audio_generator:
                                if audio_part is not None and audio_part.size > 0:
                                     sentence_audio_parts.append(audio_part)

                            if sentence_audio_parts:
                                combined_audio = np.concatenate(sentence_audio_parts)
                                duration = time.time() - start_time
                                self.synthesis_count += 1
                                # logger.info(f"TTS Synthesis #{self.synthesis_count} (sentence) took {duration:.2f}s, yielding {len(combined_audio)} samples.")
                                yield combined_audio # Yield the synthesized audio chunk
                            else:
                                logger.warning(f"No audio generated for sentence: '{sentence}'")

                        except Exception as e:
                            logger.error(f"Error during sentence TTS synthesis for '{sentence}': {e}", exc_info=True)
                            # Continue to next sentence/chunk

                        # Update how much of the buffer we've processed
                        processed_upto_index += len(parts[i]) + len(parts[i+1])

                     else: # If combining resulted in empty, still advance index
                          processed_upto_index += len(parts[i]) + len(parts[i+1])


                # Keep the remaining unprocessed part in the buffer
                sentence_buffer = sentence_buffer[processed_upto_index:]

            # After the loop, synthesize any remaining text in the buffer
            final_sentence = sentence_buffer.strip()
            if final_sentence:
                logger.debug(f"Synthesizing remaining text: '{final_sentence}'")
                start_time = time.time()
                try:
                    audio_generator = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda s=final_sentence: self.pipeline(s, voice=self.default_voice, speed=1.0)
                    )
                    sentence_audio_parts = []
                    for _, _, audio_part in audio_generator:
                       if audio_part is not None and audio_part.size > 0:
                           sentence_audio_parts.append(audio_part)

                    if sentence_audio_parts:
                        combined_audio = np.concatenate(sentence_audio_parts)
                        duration = time.time() - start_time
                        self.synthesis_count += 1
                        # logger.info(f"TTS Synthesis #{self.synthesis_count} (remainder) took {duration:.2f}s, yielding {len(combined_audio)} samples.")
                        yield combined_audio
                    else:
                         logger.warning(f"No audio generated for remaining text: '{final_sentence}'")

                except Exception as e:
                    logger.error(f"Error during final TTS synthesis: {e}", exc_info=True)

            logger.info("Streaming TTS synthesis finished.")

        except asyncio.CancelledError:
            logger.info("TTS synthesis stream cancelled.")
            raise # Re-raise cancellation
        except Exception as e:
            logger.error(f"Error consuming text stream for TTS: {e}", exc_info=True)
        finally:
             pass # Cleanup if needed


# --- WebSocket Handler Components (consume_text_streamer, consume_audio_stream, process_interaction, handle_client) ---
# ... (Keep these exactly as they were in the previous corrected version) ...
async def consume_text_streamer(streamer):
    """Consumes an async text streamer, yielding chunks and collecting the full text."""
    full_text = ""
    try:
        async for chunk in streamer:
            if chunk is None: break # End of stream signal
            # logger.debug(f"Text chunk received: '{chunk}'")
            if isinstance(chunk, str): # Ensure it's a string
                full_text += chunk
                yield chunk # Yield the chunk for immediate use (e.g., by TTS)
            else:
                 logger.warning(f"Received non-string chunk from text streamer: {type(chunk)}")
            # Check for cancellation frequently
            await asyncio.sleep(0.001) # Yield control briefly
    except asyncio.CancelledError:
        logger.info("Text streamer consumption cancelled.")
        # Don't yield anything further, but return what was collected so far
    except Exception as e:
        logger.error(f"Error consuming text streamer: {e}", exc_info=True)
    finally:
        # Return the fully collected text when done or cancelled/errored
        # logger.debug(f"Finished consuming text streamer. Full text length: {len(full_text)}")
        # The return value isn't directly used if yielding, but useful if called differently
         pass # We yielded chunks, so final return isn't the primary output


async def consume_audio_stream(websocket, audio_streamer, detector):
    """Consumes an async audio stream (numpy arrays), encodes, and sends via WebSocket."""
    chunks_sent = 0
    try:
        async for audio_chunk in audio_streamer:
            # Check for interrupt request BEFORE processing/sending audio
            if await detector.is_interrupt_requested():
                logger.warning("Interrupt requested during audio streaming. Stopping TTS playback.")
                # Don't clear the flag here, let the main processing loop handle it after cancellation
                raise asyncio.CancelledError("TTS interrupted by new speech")

            if audio_chunk is not None and isinstance(audio_chunk, np.ndarray) and audio_chunk.size > 0:
                try:
                    # Ensure correct dtype and range before conversion
                    if audio_chunk.dtype != np.float32: audio_chunk = audio_chunk.astype(np.float32)
                    # Clamp values just in case TTS outputs something slightly out of range
                    np.clip(audio_chunk, -1.0, 1.0, out=audio_chunk)
                    audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()

                    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                    await websocket.send(json.dumps({"audio": base64_audio}))
                    chunks_sent += 1
                    # logger.debug(f"Sent audio chunk {chunks_sent}: {len(audio_bytes)} bytes")
                except websockets.exceptions.ConnectionClosed:
                     logger.warning("WebSocket closed while trying to send audio.")
                     raise # Re-raise to stop the process
                except Exception as e:
                     logger.error(f"Error encoding/sending audio chunk: {e}", exc_info=True)
                     # Continue trying to send subsequent chunks? Or break? Let's break.
                     raise asyncio.CancelledError("Error during audio send") # Treat as cancellation
            elif audio_chunk is not None:
                 logger.warning(f"Received non-numpy or empty audio chunk from TTS: {type(audio_chunk)}")

            # Small sleep to yield control, crucial for responsiveness
            await asyncio.sleep(0.01)

        logger.info(f"Finished sending {chunks_sent} TTS audio chunks.")
    except asyncio.CancelledError as e:
         logger.info(f"Audio streaming task cancelled. Reason: {e}")
         raise # Re-raise to signal cancellation upwards
    except websockets.exceptions.ConnectionClosed:
        logger.warning("WebSocket closed during audio streaming (detected in loop).")
        raise # Re-raise to signal closure
    except Exception as e:
        logger.error(f"Unexpected error during audio streaming: {e}", exc_info=True)
        raise asyncio.CancelledError(f"Error in audio streaming: {e}") # Treat unexpected errors as cancellation

async def process_interaction(websocket, transcription, detector, gemma_processor, tts_processor):
    """Handles the full interaction cycle: Generation -> TTS -> Sending Audio"""
    logger.info(f"Starting processing for transcription: '{transcription[:50]}...'")
    gemma_streamer = None
    audio_streamer = None
    gemma_consumer_task = None
    audio_sending_task = None
    full_generated_text = ""
    user_prompt_for_history = "" # Store the prompt used

    try:
        # 1. Start Gemma Generation (returns streamer and the user prompt used)
        gemma_streamer, user_prompt_for_history = await gemma_processor.generate_streaming(transcription)

        # We need to consume the gemma_streamer in two ways:
        # - Feed it chunk-by-chunk into TTS
        # - Collect the full text for history update

        # Create a queue to act as a buffer/splitter
        text_queue = asyncio.Queue(maxsize=100) # Limit queue size to prevent excessive memory use

        # Task to consume Gemma and put chunks into the queue
        async def gemma_queue_filler():
            """Consumes Gemma streamer and puts chunks into the text_queue."""
            temp_full_text = ""
            try:
                async for chunk in gemma_streamer:
                    try:
                        await asyncio.wait_for(text_queue.put(chunk), timeout=5.0) # Timeout for queue put
                        temp_full_text += chunk
                    except asyncio.TimeoutError:
                         logger.error("Timeout putting text chunk into queue. TTS might be stuck.")
                         # Signal error downstream? Maybe put None to stop TTS?
                         await text_queue.put(None)
                         raise
                    except asyncio.CancelledError:
                         raise # Propagate cancellation
                await asyncio.wait_for(text_queue.put(None), timeout=1.0) # Signal end of stream
                return temp_full_text
            except asyncio.CancelledError:
                logger.info("Gemma queue filler task cancelled.")
                # Try to signal end if possible, but might fail if queue is blocked/cancelled
                try: await asyncio.wait_for(text_queue.put(None), timeout=0.1)
                except Exception: pass
                raise
            except Exception as e:
                logger.error(f"Error in Gemma queue filler task: {e}", exc_info=True)
                try: await asyncio.wait_for(text_queue.put(None), timeout=0.1) # Ensure termination signal on error
                except Exception: pass
                return temp_full_text # Return whatever was collected

        gemma_consumer_task = asyncio.create_task(gemma_queue_filler(), name="GemmaQueueFiller")

        # Async generator wrapper around the queue for TTS
        async def queue_reader_for_tts():
            """Reads from text_queue and yields chunks for TTS."""
            while True:
                try:
                    chunk = await text_queue.get()
                    if chunk is None:
                        logger.debug("Received end-of-stream marker from queue.")
                        break # End of stream signalled
                    yield chunk
                    text_queue.task_done() # Mark item as processed
                except asyncio.CancelledError:
                     logger.info("Queue reader for TTS cancelled.")
                     raise
                except Exception as e:
                     logger.error(f"Error reading from text queue for TTS: {e}", exc_info=True)
                     break # Stop processing on error

        # 2. Start TTS Synthesis (consumes the queue reader stream, yields audio stream)
        tts_text_input_stream = queue_reader_for_tts()
        audio_streamer = tts_processor.synthesize_speech_stream(tts_text_input_stream)

        # 3. Start Sending Audio (consumes audio stream, sends to websocket)
        # Pass detector for interrupt checks within the audio sender
        audio_sending_task = asyncio.create_task(consume_audio_stream(websocket, audio_streamer, detector), name="AudioSender")

        # 4. Wait for tasks. Crucially, wait for audio sending OR gemma consumption to finish/fail.
        #    If audio sending is cancelled (e.g., by interrupt), we need to stop waiting and clean up.
        #    If gemma finishes, we still need audio sending to complete.
        done, pending = await asyncio.wait(
            {gemma_consumer_task, audio_sending_task},
            return_when=asyncio.FIRST_COMPLETED # Change to FIRST_COMPLETED
        )

        # --- Handle task completion/cancellation ---
        processed_normally = True
        if audio_sending_task in done:
            # Audio sending finished (or failed/cancelled). Check its state.
            try:
                audio_sending_task.result() # Raise exception if it failed
                logger.info("Audio sending task completed successfully.")
                # If audio finished, we still need to wait for Gemma text collection
                if gemma_consumer_task in pending:
                     logger.info("Waiting for Gemma text collection to complete...")
                     try:
                          await asyncio.wait_for(gemma_consumer_task, timeout=10) # Wait with timeout
                          full_generated_text = gemma_consumer_task.result()
                     except asyncio.TimeoutError:
                          logger.error("Timeout waiting for Gemma text collection after audio finished.")
                          gemma_consumer_task.cancel()
                          processed_normally = False
                     except asyncio.CancelledError:
                           logger.info("Gemma text collection was cancelled while waiting.")
                           processed_normally = False
                     except Exception as e:
                           logger.error(f"Gemma consumer task failed while waiting: {e}")
                           processed_normally = False

            except asyncio.CancelledError:
                logger.info("Audio sending task was cancelled (likely by interrupt).")
                processed_normally = False
                # Ensure Gemma task is also cancelled if audio was interrupted
                if gemma_consumer_task in pending:
                    logger.info("Cancelling pending Gemma consumer task due to audio cancellation.")
                    gemma_consumer_task.cancel()
            except Exception as e:
                 logger.error(f"Audio sending task failed: {e}")
                 processed_normally = False
                 # Ensure Gemma task is also cancelled if audio failed
                 if gemma_consumer_task in pending:
                     logger.info("Cancelling pending Gemma consumer task due to audio error.")
                     gemma_consumer_task.cancel()

        elif gemma_consumer_task in done:
            # Gemma text collection finished first. Check its state.
            try:
                full_generated_text = gemma_consumer_task.result()
                logger.info(f"Gemma generation completed. Full text length: {len(full_generated_text)}")
                # Now wait for the audio sending to complete
                if audio_sending_task in pending:
                     logger.info("Gemma finished, waiting for audio sending to complete...")
                     try:
                          # Wait indefinitely? Or with a timeout? Indefinitely is simpler.
                          await audio_sending_task
                          audio_sending_task.result() # Check for errors after waiting
                     except asyncio.CancelledError:
                           logger.info("Audio sending was cancelled while waiting after Gemma finished.")
                           processed_normally = False
                     except Exception as e:
                           logger.error(f"Audio sending task failed while waiting: {e}")
                           processed_normally = False
            except asyncio.CancelledError:
                logger.info("Gemma text collection was cancelled.")
                processed_normally = False
                 # Ensure audio task is also cancelled if Gemma was cancelled
                if audio_sending_task in pending:
                    logger.info("Cancelling pending audio sending task due to Gemma cancellation.")
                    audio_sending_task.cancel()
            except Exception as e:
                logger.error(f"Gemma consumer task failed: {e}")
                processed_normally = False
                 # Ensure audio task is also cancelled if Gemma failed
                if audio_sending_task in pending:
                    logger.info("Cancelling pending audio sending task due to Gemma error.")
                    audio_sending_task.cancel()

        # Ensure any remaining pending tasks (shouldn't be any with this logic) are cancelled
        for task in pending:
            logger.warning(f"Cancelling unexpected pending task: {task.get_name()}")
            task.cancel()

        # 5. Update History only if the process wasn't interrupted/cancelled prematurely
        #    and we actually got some generated text.
        if processed_normally and full_generated_text:
            logger.info("Updating Gemma history.")
            gemma_processor.update_history(user_prompt_for_history, full_generated_text)
        elif not processed_normally:
            logger.warning("Interaction was interrupted or failed. History not updated.")
        elif not full_generated_text:
             logger.warning("No generated text collected. History not updated.")


    except asyncio.CancelledError:
        logger.info("Interaction processing task was cancelled externally (likely by interrupt handler).")
        # Ensure sub-tasks are cleaned up if they haven't been already
        if gemma_consumer_task and not gemma_consumer_task.done(): gemma_consumer_task.cancel()
        if audio_sending_task and not audio_sending_task.done(): audio_sending_task.cancel()
    except websockets.exceptions.ConnectionClosed:
         logger.warning("WebSocket closed during interaction processing.")
         if gemma_consumer_task and not gemma_consumer_task.done(): gemma_consumer_task.cancel()
         if audio_sending_task and not audio_sending_task.done(): audio_sending_task.cancel()
         raise # Propagate closure
    except Exception as e:
        logger.error(f"Unhandled error during interaction processing: {e}", exc_info=True)
        if gemma_consumer_task and not gemma_consumer_task.done(): gemma_consumer_task.cancel()
        if audio_sending_task and not audio_sending_task.done(): audio_sending_task.cancel()
    finally:
        # Ensure state is reset regardless of how the processing ended
        logger.info("Interaction processing finished or cancelled. Resetting detector state.")
        # Crucially, clear the interrupt flag *after* processing is complete or cancelled
        await detector.clear_interrupt_request()
        # Set playing state to False - use the setter associated with current_processing_task?
        # Setting it directly might be okay here as we are at the end of the interaction task's lifecycle.
        await detector.set_tts_playing(False)
        # Clear the reference to this task in the detector AFTER it has fully completed/cancelled
        # Need to be careful not to clear it if a *new* task started immediately.
        # This might be better handled by the loop that *creates* the task.


async def handle_client(websocket):
    """Main WebSocket client handler."""
    client_id = websocket.remote_address
    logger.info(f"Client {client_id} connected.")
    detector = AudioSegmentDetector() # Each client gets its own detector state
    transcriber = None
    gemma_processor = None
    tts_processor = None

    try:
        # Get singleton instances *after* client connects
        transcriber = WhisperTranscriber.get_instance()
        gemma_processor = GemmaMultimodalProcessor.get_instance()
        tts_processor = KokoroTTSProcessor.get_instance()

        # Check if TTS is actually available after trying to get instance
        if not getattr(tts_processor, 'pipeline', None):
            logger.warning(f"TTS processor unavailable for client {client_id}. Responses will be text-only (if generation works).")
            # Optionally send a message to client about TTS status?

    except Exception as e:
         logger.error(f"FATAL: Failed to initialize/get models for client {client_id}: {e}", exc_info=True)
         # Close gracefully if models critical for function failed
         await websocket.close(code=1011, reason="Internal server error during model initialization")
         return

    async def receive_loop():
        """Handles incoming messages (audio, images)."""
        while websocket.open: # Loop while connection is open
            try:
                message = await websocket.recv()
                data = json.loads(message)

                # Check if detector exists (it should)
                if not detector: break # Should not happen

                # Handle image data first (less frequent than audio)
                image_data_b64 = None
                if "image" in data:
                     image_data_b64 = data["image"]
                elif "realtime_input" in data:
                     for chunk in data["realtime_input"]["media_chunks"]:
                         if chunk["mime_type"] == "image/jpeg":
                             image_data_b64 = chunk["data"]
                             break # Assume only one image per message

                if image_data_b64:
                    # Only process image if not actively playing TTS
                    if not await detector.tts_playing and gemma_processor:
                        image_data = base64.b64decode(image_data_b64)
                        await gemma_processor.set_image(image_data)
                    else:
                        logger.debug("Ignoring image received while TTS is playing or Gemma proc is unavailable.")


                # Handle audio data (potentially multiple chunks)
                if "realtime_input" in data:
                    for chunk in data["realtime_input"]["media_chunks"]:
                        if chunk["mime_type"] == "audio/pcm":
                            audio_data = base64.b64decode(chunk["data"])
                            # Add audio - this might trigger an interrupt request internally
                            await detector.add_audio(audio_data)

                # Handle explicit stop/interrupt from client? (Optional)
                elif "command" in data and data["command"] == "interrupt":
                    logger.info(f"Client {client_id} requested interrupt.")
                    await detector.request_interrupt()
                    await detector.cancel_current_processing() # Force cancel on client command

            except websockets.exceptions.ConnectionClosedOK:
                 logger.info(f"Receive loop: Connection closed normally for {client_id}.")
                 break
            except websockets.exceptions.ConnectionClosedError as e:
                 logger.warning(f"Receive loop: Connection closed with error for {client_id}: {e}")
                 break
            except json.JSONDecodeError:
                logger.warning(f"Received invalid JSON from {client_id}.")
            except asyncio.CancelledError:
                 logger.info(f"Receive loop for {client_id} cancelled.")
                 break # Exit loop on cancellation
            except Exception as e:
                logger.error(f"Error in receive loop for {client_id}: {e}", exc_info=True)
                # Break on unexpected errors to ensure cleanup
                break

    async def segment_processing_loop():
        """Processes detected speech segments."""
        while websocket.open: # Loop while connection is open
            try:
                 # Check for interrupt FIRST - if requested, cancel ongoing task
                if await detector.is_interrupt_requested():
                    logger.warning(f"Interrupt detected in processing loop for {client_id}. Cancelling current task.")
                    await detector.cancel_current_processing()
                    # The finally block in process_interaction should clear the flag later
                    # Give cancellation a moment before checking queue again
                    await asyncio.sleep(0.1)
                    continue # Re-check interrupt flag/state

                # Only process new segment if not currently busy (TTS not playing)
                if not await detector.tts_playing:
                    segment = await detector.get_next_segment()
                    if segment:
                        # Ensure processors are available
                        if not transcriber or not gemma_processor or not tts_processor:
                             logger.error("Attempted to process segment, but models are not ready.")
                             # Maybe clear the queue or wait? For now, skip.
                             continue

                        # Start transcription (non-blocking conceptually)
                        transcription_text = await transcriber.transcribe(segment)
                        transcription_text = transcription_text.strip() if transcription_text else ""

                        # --- Filter Transcription ---
                        is_valid_transcription = False
                        if transcription_text:
                            if any(c.isalnum() for c in transcription_text):
                                words = [w for w in transcription_text.split() if any(c.isalnum() for c in w)]
                                if len(words) >= 1: # Allow single words if not just filler
                                    filler_patterns = [
                                        r'^(um+|uh+|ah+|oh+|hm+|mhm+|hmm+)$',
                                        r'^(okay|yes|no|yeah|nah|hi|hello|hey)$',
                                        r'^bye+$',
                                        r'^(thank you|thanks)$',
                                        r'^(the|a|an|is|of|to|in|it|i|me|my|you|your|he|she|they|them|their|we|us|our)$' # Very common short words
                                    ]
                                    # Check if it's ONLY a filler/common short word
                                    is_only_filler = len(words) == 1 and any(re.fullmatch(pattern, transcription_text.lower()) for pattern in filler_patterns)

                                    if not is_only_filler:
                                        is_valid_transcription = True
                                    else:
                                        logger.info(f"Skipping filler/common short transcription: '{transcription_text}'")
                                else:
                                     # This case should be caught by isalnum check mostly, but safety belt
                                    logger.info(f"Skipping empty/non-word transcription: '{transcription_text}'")
                            else:
                                logger.info(f"Skipping non-alphanumeric transcription: '{transcription_text}'")
                        # else: # No need for else, empty transcription already handled


                        if is_valid_transcription:
                            # --- Start Interaction ---
                             logger.info(f"Valid transcription detected for {client_id}. Starting interaction task.")
                            # Create the main processing task for this interaction
                            # Pass all necessary objects
                            interaction_task = asyncio.create_task(
                                process_interaction(websocket, transcription_text, detector, gemma_processor, tts_processor),
                                name=f"InteractionTask-{client_id}-{time.time()}"
                            )
                            # Register this task with the detector
                            await detector.set_current_processing_task(interaction_task)
                            # Add a callback to clear the task from detector upon completion/cancellation
                            interaction_task.add_done_callback(
                                lambda t: asyncio.create_task(detector.processing_task_lock.acquire())
                                .add_done_callback(lambda _: asyncio.create_task(
                                        _clear_task_reference(detector, t)).add_done_callback(lambda __: detector.processing_task_lock.release()))
                                )

                            # Loop will continue, next iteration will see tts_playing is True
                        else:
                            # If transcription invalid, ensure we are not marked as playing
                             # (should already be false, but doesn't hurt to ensure)
                             if await detector.tts_playing:
                                  logger.warning("Invalid transcription, but TTS was marked as playing. Resetting.")
                                  await detector.set_tts_playing(False)

                # Short sleep ONLY if no work was done (no segment processed AND not playing)
                # If playing TTS, we want the loop to check for interrupts quickly.
                if not await detector.tts_playing and segment is None:
                     await asyncio.sleep(0.05)
                else:
                     await asyncio.sleep(0.01) # Shorter sleep if busy or just processed

            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Segment processing loop: Connection closed for {client_id}.")
                break # Exit loop
            except asyncio.CancelledError:
                 logger.info(f"Segment processing loop for {client_id} cancelled.")
                 break # Exit loop
            except Exception as e:
                logger.error(f"Error in segment processing loop for {client_id}: {e}", exc_info=True)
                # Attempt to reset state on error? Cancelling current processing might be safest.
                try:
                    await detector.cancel_current_processing()
                except Exception as cancel_e:
                     logger.error(f"Error trying to cancel processing after loop error: {cancel_e}")
                await asyncio.sleep(1) # Pause before retrying

    # Helper function for task done callback to clear reference safely
    async def _clear_task_reference(detector, task):
        # async with detector.processing_task_lock: # Acquire lock in caller now
        if detector.current_processing_task is task:
            logger.debug(f"Clearing completed/cancelled task reference: {task.get_name()}")
            detector.current_processing_task = None
            # State (tts_playing, interrupt) should be handled within process_interaction's finally block

    # --- Start client tasks ---
    receiver_task = asyncio.create_task(receive_loop(), name=f"Receiver-{client_id}")
    processor_task = asyncio.create_task(segment_processing_loop(), name=f"Processor-{client_id}")
    all_client_tasks = {receiver_task, processor_task}

    try:
        # Wait for either task to complete (normally or abnormally)
        done, pending = await asyncio.wait(
            all_client_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Log which task finished first and why (if possible)
        for task in done:
            try:
                 result = task.result() # Check for exceptions
                 logger.info(f"Task {task.get_name()} completed normally for {client_id}.")
            except asyncio.CancelledError:
                 logger.info(f"Task {task.get_name()} was cancelled for {client_id}.")
            except websockets.exceptions.ConnectionClosed:
                 logger.info(f"Task {task.get_name()} detected connection closed for {client_id}.")
            except Exception as e:
                 logger.error(f"Task {task.get_name()} failed for {client_id}: {e}", exc_info=True)


    except Exception as e:
         # This catches errors in the asyncio.wait itself, unlikely but possible
         logger.error(f"Error in main client handler wait for {client_id}: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        logger.info(f"Client {client_id} disconnecting or handler exiting. Cleaning up tasks...")

        # Cancel any pending tasks (the one that didn't finish first)
        for task in pending: # 'pending' is from the asyncio.wait result
            if not task.done():
                logger.info(f"Cancelling pending task: {task.get_name()}")
                task.cancel()

         # Also cancel the main interaction task if it's still running
        if detector: # Check if detector was initialized
             await detector.cancel_current_processing()

        # Gather all tasks (including potentially the interaction task) to wait for cancellation
        tasks_to_gather = all_client_tasks
        if detector and detector.current_processing_task:
             tasks_to_gather.add(detector.current_processing_task)

        # Wait briefly for all tasks to finish cancelling
        if tasks_to_gather:
             logger.debug(f"Waiting for {len(tasks_to_gather)} tasks to finalize cancellation...")
             await asyncio.gather(*tasks_to_gather, return_exceptions=True)
             logger.debug("Task cancellation gathering complete.")

        # Close WebSocket connection if not already closed
        if websocket.open:
            logger.info(f"Closing WebSocket connection for {client_id}.")
            await websocket.close(code=1000, reason="Server shutting down client handler")

        logger.info(f"Client {client_id} cleanup complete.")


async def main():
    """Initializes models and starts the WebSocket server."""
    logger.info("Server starting up...")
    # Initialize models eagerly at startup by calling get_instance
    logger.info("Initializing models...")
    try:
        # These calls will instantiate the singletons if they don't exist
        WhisperTranscriber.get_instance()
        GemmaMultimodalProcessor.get_instance()
        KokoroTTSProcessor.get_instance()
        logger.info("All models initialized successfully (or initialization attempted).")
        # Add checks here if initialization methods return status or raise specific exceptions
        if not getattr(KokoroTTSProcessor.get_instance(), 'pipeline', None):
             logger.warning("Kokoro TTS pipeline failed to initialize during startup.")

    except Exception as e:
        # Catch exceptions raised during the __init__ of the singletons
        logger.critical(f"FATAL: Could not initialize models on startup: {e}", exc_info=True)
        sys.exit(1) # Exit if critical models fail

    logger.info(f"Starting WebSocket server on 0.0.0.0:{WEBSOCKET_PORT}")
    try:
        # Set higher limits for queue size and buffer size if needed, but be mindful of memory
        async with websockets.serve(
            handle_client,
            "0.0.0.0",
            WEBSOCKET_PORT,
            ping_interval=PING_INTERVAL,
            ping_timeout=PING_TIMEOUT,
            close_timeout=10,
            # max_size=2**22,  # Increase max message size (e.g., 4MB) if sending large images/audio
            # max_queue=128    # Increase server's internal queue size
        ):
            logger.info("WebSocket server running.")
            await asyncio.Future()  # Run forever
    except OSError as e:
        logger.critical(f"FATAL: Could not start server on port {WEBSOCKET_PORT}. Port likely in use or insufficient permissions: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"FATAL: WebSocket server failed to start or run: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt).")
    except Exception as e:
        logger.critical(f"Unhandled exception in main asyncio run: {e}", exc_info=True)
    finally:
        logger.info("Server shutdown process complete.")
