import asyncio
import json
import websockets
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, Gemma3ForConditionalGeneration, TextIteratorStreamer
from threading import Thread
import numpy as np
import logging
import sys
import io
from PIL import Image
import time
import os
from datetime import datetime
import re

# Import Kokoro TTS library
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    # Define a dummy KPipeline if kokoro is not installed
    class KPipeline:
        def __init__(self, *args, **kwargs):
            logging.warning("Kokoro TTS library not found. TTS functionality will be disabled.")
        def __call__(self, *args, **kwargs):
            logging.warning("Kokoro TTS called, but library is unavailable.")
            # Yield nothing to simulate an empty generator
            if False: # This makes it a generator function
                 yield

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Constants ---
AUDIO_SAMPLE_RATE = 16000
AUDIO_ENERGY_THRESHOLD = 0.015 # Adjust based on mic sensitivity
AUDIO_SILENCE_DURATION = 0.7 # Seconds of silence to end segment
AUDIO_MIN_SPEECH_DURATION = 0.5 # Seconds minimum speech length
AUDIO_MAX_SPEECH_DURATION = 10.0 # Seconds maximum speech length before forced cut
WEBSOCKET_PORT = 9073
WEBSOCKET_PING_INTERVAL = 20
WEBSOCKET_PING_TIMEOUT = 60
MAX_CONVERSATION_HISTORY = 6 # Keep last 3 user/assistant pairs
GEMMA_MAX_NEW_TOKENS = 150 # Max tokens for Gemma response
TTS_CHUNK_SPLIT_PATTERN = r'[.!?。！？,，;；: ]+' # Split text for TTS chunking more aggressively

# --- Singleton Resource Managers ---

class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        # Basic thread safety for instance creation
        if cls not in cls._instances:
            with cls.__lock__: # Use a class-level lock
                 if cls not in cls._instances:
                      cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def __init__(cls, name, bases, dct):
         super().__init__(name, bases, dct)
         cls.__lock__ = asyncio.Lock() # Changed to asyncio.Lock assuming use in async context primarily


class WhisperTranscriber(metaclass=SingletonMeta):
    """Handles speech transcription using Whisper large-v3 model."""
    def __init__(self):
        logger.info("Initializing Whisper Transcriber...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
        model_id = "openai/whisper-large-v3" # Using v3 directly

        try:
            # Load model to appropriate device
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            # Only move to device if not using device_map='auto' which handles placement
            if hasattr(self.model, 'hf_device_map'): # Check if device_map was used internally
                 logger.info(f"Whisper model device map: {self.model.hf_device_map}")
            else:
                 self.model.to(self.device)

            self.processor = AutoProcessor.from_pretrained(model_id)

            # Determine pipeline device - crucial for performance
            pipeline_device = self.device
            # If model was loaded with device_map, pipeline needs device=None or correct index
            if hasattr(self.model, 'hf_device_map') and 'cuda' in self.model.hf_device_map.values():
                 # Simple case: if any part is on cuda, target cuda:0 for pipeline
                 # More complex maps might require adjusting pipeline_device index
                 pipeline_device = next((d for d in self.model.hf_device_map.values() if 'cuda' in d), self.device)
                 logger.info(f"Using pipeline device based on model map: {pipeline_device}")

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=pipeline_device, # Use determined device
                chunk_length_s=30, # Process audio in chunks
                batch_size=8       # Batch inference if possible (helps on GPU)
            )
            logger.info(f"Whisper model ({model_id}) ready on pipeline device: {pipeline_device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            raise

    async def transcribe(self, audio_bytes):
        """Transcribe audio bytes to text."""
        required_samples = int(AUDIO_MIN_SPEECH_DURATION * AUDIO_SAMPLE_RATE * 2) # 16-bit samples
        if not audio_bytes or len(audio_bytes) < required_samples:
            logger.debug(f"Skipping transcription, audio too short: {len(audio_bytes)} bytes < {required_samples} bytes")
            return ""
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Run in executor to avoid blocking asyncio loop
            result = await asyncio.get_event_loop().run_in_executor(
                None, # Default thread pool executor
                lambda: self.pipe(
                    {"array": audio_array, "sampling_rate": AUDIO_SAMPLE_RATE},
                     generate_kwargs={
                        "task": "transcribe",
                        "language": "english", # Force English if desired
                        "temperature": 0.0,    # Low temperature for deterministic transcription
                        # "do_sample": False, # Alternative to temperature 0
                    },
                    return_timestamps=False # Don't need timestamps here
                )
            )
            text = result.get("text", "").strip()
            logger.info(f"Transcription: '{text}'")
            return text
        except Exception as e:
            # Log detailed error, including potentially the audio length/shape if useful
            logger.error(f"Transcription error for audio of length {len(audio_bytes)} bytes: {e}", exc_info=True)
            return ""

class GemmaMultimodalProcessor(metaclass=SingletonMeta):
    """Handles multimodal generation using Gemma model."""
    def __init__(self):
        logger.info("Initializing Gemma Multimodal Processor...")
        # Use explicit device placement if needed, otherwise 'auto' is fine
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_id = "google/gemma-3-4b-it" # Specify model

        try:
            # Recommended: Use device_map="auto" for multi-GPU or large models
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto", # Let accelerate handle placement
                # load_in_8bit=True, # 8-bit quantization - Requires bitsandbytes
                quantization_config=transformers.BitsAndBytesConfig(load_in_8bit=True) if torch.cuda.is_available() else None, # Conditional quantization
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 # Use bfloat16 if available
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            # Log the actual device map used by accelerate
            if hasattr(self.model, 'hf_device_map'):
                 logger.info(f"Gemma model loaded with device map: {self.model.hf_device_map}")
            else:
                 logger.info(f"Gemma model loaded (no device map used, likely on: {self.model.device})")

        except ImportError:
             logger.error("bitsandbytes library not found. Cannot load model in 8-bit. Trying without quantization.")
             try:
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                     model_id,
                     device_map="auto",
                     torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                )
                self.processor = AutoProcessor.from_pretrained(model_id)
                if hasattr(self.model, 'hf_device_map'):
                    logger.info(f"Gemma model loaded (no 8bit) with device map: {self.model.hf_device_map}")
                else:
                    logger.info(f"Gemma model loaded (no 8bit, no device map used, likely on: {self.model.device})")
             except Exception as e_noquant:
                 logger.error(f"Failed to load Gemma model even without quantization: {e_noquant}", exc_info=True)
                 raise

        except Exception as e:
            logger.error(f"Failed to load Gemma model: {e}", exc_info=True)
            raise

        self.last_image = None
        self.last_image_timestamp = 0
        self.image_lock = asyncio.Lock()
        self.message_history = [] # Stores {"role": "user/assistant", "content": [{"type": "text", "text": ...}]}

    async def set_image(self, image_data):
        """Cache the most recent image, resizing it."""
        async with self.image_lock:
            try:
                img_buffer = io.BytesIO(image_data)
                image = Image.open(img_buffer).convert("RGB") # Ensure RGB format

                # Optional: Resize for faster processing / lower memory
                # max_size = (1024, 1024)
                # image.thumbnail(max_size, Image.Resampling.LANCZOS)

                self.last_image = image
                self.last_image_timestamp = time.time()
                # Clear history when a new image context arrives
                self.message_history = []
                logger.info(f"New image received ({image.size[0]}x{image.size[1]}) and cached. History cleared.")
                return True
            except Exception as e:
                logger.error(f"Error processing image: {e}", exc_info=True)
                self.last_image = None # Invalidate image on error
                return False

    async def _build_prompt(self, text):
         """Builds the prompt structure for the model including history."""
         # Use async with for the lock
         async with self.image_lock:
              if not self.last_image:
                   logger.warning("Attempting to generate without an image.")
                   # Handle generation without image - maybe switch to text-only prompt?
                   return [
                        {"role": "system", "content": [{"type": "text", "text": "You are a text-only assistant."}]},
                        {"role": "user", "content": [{"type": "text", "text": "Error: No image context was provided for my question: " + text}]}
                   ]

              system_prompt = {
                   "role": "system",
                   "content": [{"type": "text", "text": """You are a helpful assistant providing spoken responses about images and engaging in natural conversation. Keep your responses concise, fluent, and conversational (1-3 sentences typically). Use natural oral language.
- If the query is about the image, describe relevant aspects.
- If the query is general, respond conversationally without forcing image descriptions.
- If unsure, ask for clarification politely.
- Maintain context from recent conversation turns."""}]
              }

              # Prepare history (limit length)
              history = self.message_history[-MAX_CONVERSATION_HISTORY:]

              current_user_message = {
                   "role": "user",
                   "content": [
                        # It's safer to pass the image data/object directly if processor handles it
                        # Using a copy might be okay, depends on processor implementation
                        {"type": "image"}, # Placeholder for processor
                        {"type": "text", "text": text}
                   ]
              }
              # Add the actual image to the message list for processing if needed
              # The processor.apply_chat_template expects images alongside text
              # We pass the image separately to the processor usually

              messages = [system_prompt] + history + [current_user_message]
              return messages # Return the message structure


    async def generate_streaming(self, text):
        """Generate response stream using the latest image and text input."""
        async with self.image_lock: # Ensure image doesn't change during generation setup
             current_image = self.last_image # Get image under lock

        if not current_image:
            logger.warning("No image available for generation.")
            async def empty_streamer():
                 yield "Sorry, I don't have an image to look at right now."
                 if False: yield # Make it an async generator
            return empty_streamer(), None # Return streamer and None for image context

        # Build prompt structure (does not need lock anymore)
        messages = await self._build_prompt(text) # Await the async build method

        try:
            # Prepare inputs using the processor
            # Pass the actual image object here along with messages
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True, # Tokenize directly
                return_dict=True,
                return_tensors="pt",
                # images=current_image # Pass image here
            ).to(self.model.device)

            # Add image pixel values manually if apply_chat_template doesn't handle it
            # (Check GemmaProcessor documentation for correct usage)
            if 'pixel_values' not in inputs and current_image:
                 image_inputs = self.processor(images=current_image, return_tensors="pt").to(self.model.device)
                 inputs['pixel_values'] = image_inputs['pixel_values']


            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = dict(
                # **inputs, # Pass processed inputs directly
                input_ids = inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
                pixel_values = inputs.get("pixel_values"), # Include pixel values
                streamer=streamer,
                max_new_tokens=GEMMA_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                use_cache=True,
            )

            # Run model.generate in a separate thread via run_in_executor
            loop = asyncio.get_event_loop()
            # Make the target function directly callable by executor
            def generation_target():
                 with torch.no_grad(): # Disable gradient calculation for inference
                     self.model.generate(**generation_kwargs)

            thread = Thread(target=lambda: loop.run_in_executor(None, generation_target), daemon=True)
            thread.start()

            logger.info("Gemma streaming generation started...")

            # Return the async iterator part of the streamer and the context used
            async def streamer_async_generator():
                 full_response = ""
                 try:
                    for token in streamer:
                       yield token
                       full_response += token
                 finally:
                     # Update history *after* full generation is complete
                     logger.info(f"Gemma full response length: {len(full_response)}")
                     # Ensure history update happens safely
                     await self.update_history(text, full_response) # await the async method
                     thread.join(timeout=5.0)
                     if thread.is_alive():
                        logger.warning("Generation thread did not finish cleanly.")

            return streamer_async_generator(), current_image # Return streamer and image used

        except Exception as e:
            logger.error(f"Gemma generation error: {e}", exc_info=True)
            async def error_streamer():
                yield "Sorry, I encountered an error while generating the response."
                if False: yield
            return error_streamer(), None


    async def update_history(self, user_text, assistant_response):
         """Update message history, trimming if necessary."""
         if not user_text or not assistant_response: # Don't add empty exchanges
              return

         # This method might be called from the streamer's finally block,
         # ensure thread safety if multiple clients could modify history concurrently
         # (Although singletons are usually accessed by one client thread at a time here)

         # Append new messages
         self.message_history.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
         # For assistant, only add text part to history to avoid image duplication
         self.message_history.append({"role": "assistant", "content": [{"type": "text", "text": assistant_response}]})


         # Trim history
         if len(self.message_history) > MAX_CONVERSATION_HISTORY:
             # Keep the last MAX_CONVERSATION_HISTORY messages
             self.message_history = self.message_history[-MAX_CONVERSATION_HISTORY:]
         logger.debug(f"History updated. Current length: {len(self.message_history)}")


class KokoroTTSProcessor(metaclass=SingletonMeta):
    """Handles text-to-speech conversion using Kokoro model."""
    def __init__(self):
        logger.info("Initializing Kokoro TTS Processor...")
        self.pipeline = None
        if KOKORO_AVAILABLE:
            try:
                # Initialize Kokoro TTS pipeline - Adjust lang_code and voice as needed
                self.pipeline = KPipeline(lang_code='en') # Explicitly English
                self.default_voice = 'en_aux_jennifer' # Example English voice, find available ones
                logger.info(f"Kokoro TTS processor initialized successfully with voice {self.default_voice}.")
            except Exception as e:
                logger.error(f"Error initializing Kokoro TTS: {e}", exc_info=True)
                self.pipeline = None
        else:
            logger.warning("Kokoro TTS library not available. TTS is disabled.")

    async def synthesize_speech_chunk(self, text_chunk):
        """Synthesize a single chunk of text to speech."""
        if not self.pipeline or not text_chunk:
            return None

        try:
            logger.info(f"Synthesizing TTS for chunk: '{text_chunk[:50]}...'")
            start_time = time.time()

            # Run TTS in executor
            audio_segments = []
            def sync_synthesize():
                 results = []
                 try:
                     # Make sure pipeline call handles potential errors internally
                     generator = self.pipeline(
                         text_chunk,
                         voice=self.default_voice,
                         speed=1.0, # Adjust speed if needed
                     )
                     for _, _, audio_data in generator:
                         if audio_data is not None and len(audio_data) > 0:
                            results.append(audio_data)

                 except Exception as e_inner:
                      logger.error(f"Error during Kokoro pipeline execution for chunk '{text_chunk[:30]}...': {e_inner}", exc_info=False) # Log less verbosely here
                      return None # Return None on inner error

                 if not results:
                      return None
                 return np.concatenate(results)

            combined_audio = await asyncio.get_event_loop().run_in_executor(None, sync_synthesize)

            end_time = time.time()
            if combined_audio is not None:
                 logger.info(f"TTS synthesis for chunk complete ({len(combined_audio)} samples) in {end_time - start_time:.2f}s")
                 return combined_audio
            else:
                 logger.warning(f"TTS synthesis for chunk '{text_chunk[:50]}...' resulted in no audio.")
                 return None

        except Exception as e:
            # Log errors occurring outside the sync_synthesize call (e.g., executor issues)
            logger.error(f"Speech synthesis task error for chunk '{text_chunk[:50]}...': {e}", exc_info=True)
            return None


class ClientHandler:
    """Manages the state and processing for a single WebSocket client."""

    def __init__(self, websocket):
        self.websocket = websocket
        self.client_address = websocket.remote_address
        self.transcriber = WhisperTranscriber()
        self.gemma_processor = GemmaMultimodalProcessor()
        self.tts_processor = KokoroTTSProcessor()

        # Audio buffering and VAD state
        self.audio_buffer = bytearray()
        self.is_speech_active = False
        self.silence_counter = 0 # In samples
        self.speech_start_time = 0
        self.vad_lock = asyncio.Lock()

        # Processing state
        self.processing_lock = asyncio.Lock() # Ensures only one message processed at a time
        self.current_processing_task = None # Holds the main task for transcription -> generation -> tts
        self.is_responding = False # Flag indicating if the system is generating/speaking

    async def _reset_vad_state(self):
        """Resets the voice activity detection state."""
        async with self.vad_lock:
            self.is_speech_active = False
            self.silence_counter = 0
            self.audio_buffer = bytearray() # Clear buffer completely after processing
            logger.debug("VAD state reset.")

    async def _cancel_current_processing(self, reason=""):
        """Cancels any ongoing transcription, generation, or TTS task."""
        async with self.processing_lock: # Ensure atomicity
            if self.current_processing_task and not self.current_processing_task.done():
                log_reason = f" ({reason})" if reason else ""
                logger.info(f"Interrupting current response generation/playback{log_reason}.")
                self.current_processing_task.cancel()
                # Send interrupt message to client immediately
                try:
                    # Check if websocket is still open before sending
                    if not self.websocket.closed:
                        await self.websocket.send(json.dumps({"interrupt": True}))
                        logger.info("Sent interrupt signal to client.")
                    else:
                         logger.warning("WebSocket closed, cannot send interrupt signal.")
                except websockets.exceptions.ConnectionClosed:
                     logger.warning("Connection closed during interrupt signal sending.")
                except Exception as e:
                     logger.error(f"Failed to send interrupt signal: {e}")

                # Reset state immediately after requesting cancellation
                self.current_processing_task = None
                self.is_responding = False
                # Give cancellation a moment to propagate if needed, but immediate reset is safer
                # await asyncio.sleep(0.05) # Optional small delay

    async def process_audio_chunk(self, audio_bytes):
        """Processes an incoming audio chunk using VAD."""
        segment_to_process = None
        start_processing = False

        async with self.vad_lock:
            self.audio_buffer.extend(audio_bytes)
            current_buffer_len_samples = len(self.audio_buffer) // 2 # Assuming 16-bit PCM

            # Analyze the *new* chunk's energy
            if len(audio_bytes) > 0:
                try:
                    # Ensure sufficient data for meaningful energy calculation
                    if len(audio_bytes) >= 320: # ~10ms at 16kHz, 16-bit
                         audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                         energy = np.sqrt(np.mean(audio_array**2))
                    else:
                         energy = 0 # Treat very small chunks as silence for energy calc
                except Exception as e:
                     logger.error(f"Error calculating energy: {e}")
                     energy = 0 # Default to silence on error

                # --- VAD Logic ---
                if self.is_speech_active:
                    if energy > AUDIO_ENERGY_THRESHOLD:
                        self.silence_counter = 0 # Reset silence counter
                    else:
                        # Accumulate silence samples based on the current chunk length
                        self.silence_counter += len(audio_bytes) // 2 # Samples

                    segment_duration_seconds = current_buffer_len_samples / AUDIO_SAMPLE_RATE

                    # Check for end of speech (silence duration met)
                    required_silence_samples = int(AUDIO_SILENCE_DURATION * AUDIO_SAMPLE_RATE)
                    if self.silence_counter >= required_silence_samples:
                        speech_end_idx_bytes = len(self.audio_buffer) - (self.silence_counter * 2) # Index in bytes
                        # Ensure minimum speech length before processing
                        if segment_duration_seconds >= AUDIO_MIN_SPEECH_DURATION:
                            segment_to_process = bytes(self.audio_buffer[:speech_end_idx_bytes])
                            logger.info(f"Speech segment DETECTED by silence ({segment_duration_seconds:.2f}s)")
                            start_processing = True # Signal to process this segment
                        else:
                            logger.info(f"Discarding short segment detected by silence ({segment_duration_seconds:.2f}s)")
                        # Reset state *after* extracting segment
                        self.is_speech_active = False
                        self.silence_counter = 0
                        self.audio_buffer = bytearray() # Clear buffer

                    # Check for maximum speech duration
                    elif segment_duration_seconds > AUDIO_MAX_SPEECH_DURATION:
                        segment_to_process = bytes(self.audio_buffer) # Process the whole buffer
                        logger.info(f"Speech segment CUT by max duration ({segment_duration_seconds:.2f}s)")
                        start_processing = True # Signal to process this segment
                        # Reset state *after* extracting segment
                        self.is_speech_active = False
                        self.silence_counter = 0
                        self.audio_buffer = bytearray() # Clear buffer

                elif energy > AUDIO_ENERGY_THRESHOLD:
                    # --- Speech Start Detected ---
                    self.is_speech_active = True
                    self.silence_counter = 0
                    self.speech_start_time = time.time()
                    logger.info(f"Speech START detected (Energy: {energy:.4f})")
                    # --- INTERRUPT Logic ---
                    if self.is_responding:
                        # Cancel outside the vad_lock to avoid deadlock if cancel needs processing_lock
                        # Use asyncio.create_task to run cancellation concurrently
                        logger.info("New speech detected while responding, initiating interruption.")
                        asyncio.create_task(self._cancel_current_processing("new speech detected"))
                    # Keep the triggering audio chunk and potentially some prior buffer
                    # For simplicity, we clear buffer on segment end/cut, so buffer starts fresh here
                    # self.audio_buffer = audio_bytes # Option: Start buffer only with triggering chunk
            else:
                # Handle case of empty audio chunk if necessary
                pass

        # Process the detected segment outside the VAD lock
        if start_processing and segment_to_process:
            # Use processing_lock to ensure only one task runs at a time
            acquired = await self.processing_lock.acquire()
            if acquired: # Should always acquire unless already locked
                 if not self.is_responding: # Double check state after acquiring lock
                     self.is_responding = True # Set flag *before* creating task
                     self.current_processing_task = asyncio.create_task(
                          self.handle_speech_segment(segment_to_process),
                          name=f"ProcessingSegment_{time.time():.0f}"
                     )
                     # Release lock once task is created and state set
                     self.processing_lock.release()
                     logger.info(f"Created processing task for segment ({len(segment_to_process)} bytes).")
                 else:
                      # This case should be rare if interrupt logic works, but handles race conditions
                      logger.warning("New segment detected, but system is already processing/cancelling. Ignoring.")
                      self.processing_lock.release() # Release lock if not starting new task
            else:
                 # Should not happen with a single lock, but indicates logic error if it does
                 logger.error("Failed to acquire processing lock.")


    async def handle_speech_segment(self, audio_segment_bytes):
        """Handles the full pipeline for a detected speech segment."""
        task_start_time = time.time()
        transcription = ""
        try:
            logger.info(f"--- Starting segment processing (Task: {asyncio.current_task().get_name()}) ---")
            transcription_start_time = time.time()
            transcription = await self.transcriber.transcribe(audio_segment_bytes)
            transcription_end_time = time.time()
            logger.info(f"Transcription took {transcription_end_time - transcription_start_time:.2f}s")

            if not transcription or not any(c.isalnum() for c in transcription):
                logger.info("Empty or non-alphanumeric transcription, skipping.")
                return # Exit processing early

            # Filter common filler sounds / short utterances
            filler_patterns = [
                r'^(um+|uh+|ah+|oh+|hm+|mhm+|hmm+|huh)$', # Added huh
                r'^(okay|yes|no|yeah|nah|bye|hello|hi)$', # Added greetings
                r'^(thanks?|thank you|please)$' # Added please
            ]
            words = re.findall(r'\b\w+\b', transcription.lower())
            is_filler = any(re.fullmatch(pattern, transcription.lower()) for pattern in filler_patterns)

            if len(words) <= 1 or is_filler:
                 logger.info(f"Skipping filler/short transcription: '{transcription}'")
                 return


            logger.info(f"Processing transcription: '{transcription}'")
            generation_start_time = time.time()

            # Start streaming generation
            gemma_streamer, image_context = await self.gemma_processor.generate_streaming(transcription)

            accumulated_text_chunk = ""
            sentence_end_pattern = re.compile(TTS_CHUNK_SPLIT_PATTERN)
            min_chunk_len = 25 # Minimum characters before attempting TTS for fluidity

            async for text_fragment in gemma_streamer:
                # Check for cancellation request frequently within the loop
                await asyncio.sleep(0) # Yield control briefly, allows cancellation check

                accumulated_text_chunk += text_fragment

                # Check if we have a potential chunk end
                found_punctuation = sentence_end_pattern.search(text_fragment) # Search in the *new* part
                # More robust check: check end of accumulated string too
                ends_with_split = sentence_end_pattern.match(accumulated_text_chunk[::-1]) is not None

                # Synthesize if: punctuation found AND long enough, OR chunk gets very long
                should_synthesize = ( (found_punctuation or ends_with_split) and len(accumulated_text_chunk) >= min_chunk_len) or \
                                    (len(accumulated_text_chunk) > 100) # Force synth on longer chunks w/o punctuation

                if should_synthesize:
                    text_to_synthesize = accumulated_text_chunk.strip()
                    accumulated_text_chunk = "" # Reset accumulator

                    if text_to_synthesize:
                        tts_start_time = time.time()
                        audio_chunk = await self.tts_processor.synthesize_speech_chunk(text_to_synthesize)
                        tts_end_time = time.time()
                        logger.debug(f"TTS chunk synthesis took {tts_end_time - tts_start_time:.2f}s")

                        if audio_chunk is not None:
                            try:
                                # Convert to 16-bit PCM bytes and Base64 encode
                                audio_bytes = (audio_chunk * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
                                base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

                                # Ensure websocket is open before sending
                                if not self.websocket.closed:
                                     await self.websocket.send(json.dumps({"audio": base64_audio}))
                                     logger.info(f"Sent audio chunk ({len(audio_bytes)} bytes)")
                                else:
                                     logger.warning("WebSocket closed before sending audio chunk.")
                                     raise asyncio.CancelledError("WebSocket closed") # Treat as cancellation

                            except websockets.exceptions.ConnectionClosed:
                                logger.warning("Connection closed while sending audio chunk.")
                                raise asyncio.CancelledError("Connection closed") # Stop processing
                            except Exception as e:
                                logger.error(f"Error sending audio chunk: {e}")
                                raise asyncio.CancelledError("Send error") # Stop on send error


            # Synthesize any remaining text after the loop finishes
            final_text_chunk = accumulated_text_chunk.strip()
            if final_text_chunk:
                 tts_start_time = time.time()
                 audio_chunk = await self.tts_processor.synthesize_speech_chunk(final_text_chunk)
                 tts_end_time = time.time()
                 logger.debug(f"TTS final chunk synthesis took {tts_end_time - tts_start_time:.2f}s")

                 if audio_chunk is not None:
                     try:
                         audio_bytes = (audio_chunk * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
                         base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                         if not self.websocket.closed:
                             await self.websocket.send(json.dumps({"audio": base64_audio}))
                             logger.info(f"Sent final audio chunk ({len(audio_bytes)} bytes)")
                         else:
                              logger.warning("WebSocket closed before sending final audio chunk.")

                     except websockets.exceptions.ConnectionClosed:
                         logger.warning("Connection closed while sending final audio chunk.")
                         # Don't necessarily raise CancelledError here, processing is technically done
                     except Exception as e:
                         logger.error(f"Error sending final audio chunk: {e}")

            generation_end_time = time.time()
            logger.info(f"Full response generation & synthesis took {generation_end_time - generation_start_time:.2f}s")

        except asyncio.CancelledError as e:
            # Log cancellation reason if provided (from custom exception or task name)
            logger.info(f"Processing task '{asyncio.current_task().get_name()}' was cancelled. Reason: {e}")
            # History update is handled within the Gemma streamer's finally block
        except websockets.exceptions.ConnectionClosed:
             logger.info(f"Connection closed during processing task '{asyncio.current_task().get_name()}'.")
        except Exception as e:
            logger.error(f"Error during speech segment processing (Task: {asyncio.current_task().get_name()}): {e}", exc_info=True)
            # Optionally send an error message to the client
            try:
                 if not self.websocket.closed:
                      await self.websocket.send(json.dumps({"error": "An internal error occurred processing your request."}))
            except:
                 pass # Ignore errors sending the error message
        finally:
            # --- Critical Cleanup ---
            # Ensure the responding flag and task reference are cleared atomically
            async with self.processing_lock:
                self.is_responding = False
                # Only clear task reference if it's the current task (prevent race conditions)
                if self.current_processing_task is asyncio.current_task():
                    self.current_processing_task = None
                else:
                     # This indicates another task might have already started, log it
                     logger.warning(f"Task {asyncio.current_task().get_name()} finished, but self.current_processing_task points to a different task ({self.current_processing_task.get_name() if self.current_processing_task else 'None'}). State might be inconsistent.")

            task_end_time = time.time()
            logger.info(f"--- Finished segment processing (Task: {asyncio.current_task().get_name()}). Duration: {task_end_time - task_start_time:.2f}s ---")


    async def handle_messages(self):
        """Main loop to receive messages from the WebSocket client."""
        logger.info(f"Starting message handling loop for {self.client_address}")
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)

                    # Handle image data first - Allow image updates anytime
                    if "image" in data:
                         logger.info("Received image data.")
                         image_data = base64.b64decode(data["image"])
                         await self.gemma_processor.set_image(image_data)
                         # Optional: Interrupt if image arrives while responding?
                         # if self.is_responding:
                         #      logger.info("New image received while responding, interrupting.")
                         #      await self._cancel_current_processing("new image received")

                    # Handle combined realtime data (preferred format)
                    elif "realtime_input" in data:
                        # logger.debug("Received realtime_input chunk.") # Too verbose
                        for chunk in data["realtime_input"]["media_chunks"]:
                             if chunk["mime_type"] == "audio/pcm":
                                 audio_data = base64.b64decode(chunk["data"])
                                 await self.process_audio_chunk(audio_data)
                             elif chunk["mime_type"] == "image/jpeg":
                                 logger.info("Received image data within realtime stream.")
                                 image_data = base64.b64decode(chunk["data"])
                                 await self.gemma_processor.set_image(image_data)
                                 # Optional: Interrupt on image within stream?
                                 # if self.is_responding:
                                 #      logger.info("New image in stream while responding, interrupting.")
                                 #      await self._cancel_current_processing("new image in stream")

                    # Handle standalone audio chunk (alternative format)
                    elif "audio_chunk" in data:
                         # logger.debug("Received audio_chunk.") # Too verbose
                         audio_data = base64.b64decode(data["audio_chunk"])
                         await self.process_audio_chunk(audio_data)

                    # Handle other message types (e.g., configuration, text commands)
                    elif "config" in data:
                         logger.info(f"Received config message: {data}")
                         # Process config if needed (e.g., adjust VAD thresholds)

                    else:
                         logger.warning(f"Received unknown message format keys: {list(data.keys())}")


                except json.JSONDecodeError:
                    logger.error(f"Received invalid JSON: {message[:100]}...")
                except KeyError as e:
                    logger.error(f"Missing key in message: {e} - Data: {data}")
                except base64.binascii.Error as e:
                     logger.error(f"Base64 decoding error: {e} - Likely corrupted data.")
                except Exception as e:
                    # Catch-all for processing errors within the loop for one message
                    logger.error(f"Error processing message for {self.client_address}: {e}", exc_info=True)

        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"Client {self.client_address} disconnected normally (ClosedOK).")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"Client {self.client_address} disconnected with error (ClosedError): {e}")
        except asyncio.CancelledError:
             logger.info(f"Message handling task for {self.client_address} was cancelled.")
             # Ensure cleanup happens even on cancellation
             raise # Re-raise to allow higher levels to handle if needed
        except Exception as e:
            # Catch-all for errors in the message *receiving* loop itself
            logger.error(f"Unexpected error in message handling loop for {self.client_address}: {e}", exc_info=True)
        finally:
            logger.info(f"Cleaning up client handler resources for {self.client_address}...")
            # Ensure any running task is cancelled on disconnect
            # Use a specific reason for clarity in logs
            await self._cancel_current_processing(reason="client disconnected")
            # Release resources if necessary (e.g., clear large buffers)
            self.audio_buffer = bytearray()
            logger.info(f"Client handler finished cleanup for {self.client_address}")


# --- WebSocket Server ---

async def websocket_handler(websocket, path):
    """Handles a new WebSocket connection."""
    # The 'path' argument is required by websockets.serve
    logger.info(f"Client connected from {websocket.remote_address} requesting path: {path}")
    client_handler = ClientHandler(websocket)
    try:
        # Keep the connection alive by running the message handler
        await client_handler.handle_messages()
    except asyncio.CancelledError:
         logger.info(f"Websocket handler task for {websocket.remote_address} cancelled.")
    finally:
         # Ensure connection is closed if not already
         if not websocket.closed:
              await websocket.close()
         logger.info(f"Websocket connection handler finished for {websocket.remote_address} (path: {path})")


async def main():
    """Initializes models and starts the WebSocket server."""
    # Initialize singletons eagerly to load models on startup
    logger.info("Pre-loading models...")
    try:
        # Added import for bitsandbytes check
        import bitsandbytes
        # Initialize resources
        WhisperTranscriber()
        GemmaMultimodalProcessor()
        KokoroTTSProcessor()
        logger.info("Models initialized successfully.")
    except ImportError:
         logger.warning("bitsandbytes not installed. 8-bit quantization for Gemma will be disabled.")
         # Continue initialization without it if Gemma handles the fallback
         try:
             WhisperTranscriber()
             GemmaMultimodalProcessor() # Will try loading without 8bit inside
             KokoroTTSProcessor()
             logger.info("Models initialized (Gemma without 8-bit quantization).")
         except Exception as e_fallback:
             logger.critical(f"Failed to initialize models even without 8-bit quantization: {e_fallback}. Server cannot start.", exc_info=True)
             return
    except Exception as e:
        logger.critical(f"Failed to initialize models: {e}. Server cannot start.", exc_info=True)
        return # Exit if models fail to load

    logger.info(f"Starting WebSocket server on 0.0.0.0:{WEBSOCKET_PORT}")

    # Configure server with ping/pong for keepalive
    # Use loop= parameter for older asyncio versions if needed
    # loop = asyncio.get_event_loop() # Not usually needed in modern Python/websockets
    server = websockets.serve(
        websocket_handler, # Correct handler with path argument
        "0.0.0.0",
        WEBSOCKET_PORT,
        ping_interval=WEBSOCKET_PING_INTERVAL,
        ping_timeout=WEBSOCKET_PING_TIMEOUT,
        close_timeout=10,
        # Increase max message size if needed (e.g., for large images)
        max_size= 2**24 # 16MB example
    )

    try:
        async with server:
            logger.info(f"WebSocket server running. Press Ctrl+C to stop.")
            await asyncio.Future()  # Run forever until interrupted
    except asyncio.CancelledError:
         logger.info("Server main task cancelled.")
    except Exception as e:
        logger.error(f"WebSocket server encountered an unhandled error: {e}", exc_info=True)
    finally:
         logger.info("WebSocket server shutting down.")
         # Server context manager handles closing connections gracefully

# --- Entry Point ---

if __name__ == "__main__":
    # Add necessary imports for quantization check at the top if needed
    import transformers # Ensure transformers is imported for BitsAndBytesConfig
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt).")
    except Exception as e:
         logger.critical(f"Application failed to run: {e}", exc_info=True)
