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
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class WhisperTranscriber(metaclass=SingletonMeta):
    """Handles speech transcription using Whisper large-v3 model."""
    def __init__(self):
        logger.info("Initializing Whisper Transcriber...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
        model_id = "openai/whisper-large-v3" # Using v3 directly

        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
                chunk_length_s=30, # Process audio in chunks
                batch_size=8       # Batch inference if possible (helps on GPU)
            )
            logger.info(f"Whisper model ({model_id}) ready on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            raise

    async def transcribe(self, audio_bytes):
        """Transcribe audio bytes to text."""
        if not audio_bytes or len(audio_bytes) < AUDIO_MIN_SPEECH_DURATION * AUDIO_SAMPLE_RATE * 2: # Check minimum length
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
            logger.error(f"Transcription error: {e}", exc_info=True)
            return ""

class GemmaMultimodalProcessor(metaclass=SingletonMeta):
    """Handles multimodal generation using Gemma model."""
    def __init__(self):
        logger.info("Initializing Gemma Multimodal Processor...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_id = "google/gemma-3-4b-it" # Specify model

        try:
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                load_in_8bit=True, # 8-bit quantization
                torch_dtype=torch.bfloat16 # Use bfloat16 with 8-bit
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            logger.info(f"Gemma model ({model_id}) ready on {self.device}")
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
                image = Image.open(img_buffer)
                # Optional: Resize for faster processing / lower memory
                # max_size = (1024, 1024)
                # image.thumbnail(max_size, Image.Resampling.LANCZOS)
                self.last_image = image
                self.last_image_timestamp = time.time()
                # Clear history when a new image context arrives
                self.message_history = []
                logger.info(f"New image received and cached. History cleared.")
                return True
            except Exception as e:
                logger.error(f"Error processing image: {e}", exc_info=True)
                self.last_image = None # Invalidate image on error
                return False

    def _build_prompt(self, text):
        """Builds the prompt structure for the model including history."""
        asyncio.run(self.image_lock.acquire()) # Ensure image isn't changed during prompt build
        try:
            if not self.last_image:
                logger.warning("Attempting to generate without an image.")
                # Handle generation without image - maybe switch to text-only prompt?
                # For now, return a minimal structure or raise error
                return [{"role": "user", "content": [{"type": "text", "text": "Error: No image context provided."}]}]

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
                    {"type": "image", "image": self.last_image.copy()}, # Send a copy
                    {"type": "text", "text": text}
                ]
            }

            messages = [system_prompt] + history + [current_user_message]
            return messages
        finally:
             self.image_lock.release()


    async def generate_streaming(self, text):
        """Generate response stream using the latest image and text input."""
        if not self.last_image:
            logger.warning("No image available for generation.")
            async def empty_streamer():
                 yield "Sorry, I don't have an image to look at right now."
                 # Needed to make it an async generator
                 if False: yield
            return empty_streamer() # Return an empty async generator

        messages = self._build_prompt(text)

        try:
            # Prepare inputs (no need to run in executor, processor is fast)
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True, # Important for instruction-tuned models
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)

            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True, # Don't yield the input prompt
                skip_special_tokens=True
            )

            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=GEMMA_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                use_cache=True, # Important for generation speed
            )

            # Run model.generate in a separate thread via run_in_executor
            # This prevents blocking the main asyncio event loop
            loop = asyncio.get_event_loop()
            thread = Thread(target=lambda: loop.run_in_executor(None, self.model.generate, **generation_kwargs), daemon=True)
            thread.start()

            logger.info("Gemma streaming generation started...")
            # Return the async iterator part of the streamer
            # Need an async generator wrapper
            async def streamer_async_generator():
                 full_response = ""
                 try:
                    for token in streamer:
                       yield token
                       full_response += token
                 finally:
                     # Update history *after* full generation is complete
                     logger.info(f"Gemma full response length: {len(full_response)}")
                     # Needs access to original 'text' (user input)
                     self.update_history(text, full_response)
                     # Wait for the generation thread to finish if it hasn't
                     # This might block slightly, but ensures model resources are freed
                     thread.join(timeout=5.0) # Add a timeout
                     if thread.is_alive():
                        logger.warning("Generation thread did not finish cleanly.")

            return streamer_async_generator()

        except Exception as e:
            logger.error(f"Gemma generation error: {e}", exc_info=True)
            async def error_streamer():
                yield "Sorry, I encountered an error while generating the response."
                if False: yield
            return error_streamer()


    def update_history(self, user_text, assistant_response):
         """Update message history, trimming if necessary."""
         if not user_text or not assistant_response: # Don't add empty exchanges
              return

         self.message_history.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
         self.message_history.append({"role": "assistant", "content": [{"type": "text", "text": assistant_response}]})

         # Trim history
         if len(self.message_history) > MAX_CONVERSATION_HISTORY:
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
                # 'a' might be automatic? Specify 'en' or 'zh' if needed.
                self.pipeline = KPipeline(lang_code='a') # Or 'en' if only English needed
                self.default_voice = 'af_sarah' # Example English voice, change if needed
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
            # The pipeline itself yields segments based on internal splitting or the generator nature
            # We want to run the *entire* call for this chunk in the executor
            def sync_synthesize():
                 results = []
                 generator = self.pipeline(
                     text_chunk,
                     voice=self.default_voice,
                     speed=1.0, # Adjust speed if needed
                     # No explicit split_pattern here, let Kokoro handle chunking internally if it does
                 )
                 for _, _, audio_data in generator:
                     if audio_data is not None and len(audio_data) > 0:
                        results.append(audio_data)
                 if not results:
                      return None
                 return np.concatenate(results)

            combined_audio = await asyncio.get_event_loop().run_in_executor(None, sync_synthesize)

            end_time = time.time()
            if combined_audio is not None:
                 logger.info(f"TTS synthesis for chunk complete ({len(combined_audio)} samples) in {end_time - start_time:.2f}s")
                 return combined_audio
            else:
                 logger.warning(f"TTS synthesis for chunk resulted in no audio.")
                 return None

        except Exception as e:
            logger.error(f"Speech synthesis error for chunk '{text_chunk[:50]}...': {e}", exc_info=True)
            return None


class ClientHandler:
    """Manages the state and processing for a single WebSocket client."""

    def __init__(self, websocket):
        self.websocket = websocket
        self.transcriber = WhisperTranscriber()
        self.gemma_processor = GemmaMultimodalProcessor()
        self.tts_processor = KokoroTTSProcessor()

        # Audio buffering and VAD state
        self.audio_buffer = bytearray()
        self.is_speech_active = False
        self.silence_counter = 0
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
            # Keep some buffer overlap? Maybe not necessary if segments are processed quickly.
            # self.audio_buffer = self.audio_buffer[-int(AUDIO_SAMPLE_RATE*0.1*2):] # Keep last 0.1s
            self.audio_buffer = bytearray() # Clear buffer completely after processing

    async def _cancel_current_processing(self):
        """Cancels any ongoing transcription, generation, or TTS task."""
        async with self.processing_lock: # Ensure atomicity
            if self.current_processing_task and not self.current_processing_task.done():
                logger.info("Interrupting current response generation/playback.")
                self.current_processing_task.cancel()
                # Send interrupt message to client immediately
                try:
                    await self.websocket.send(json.dumps({"interrupt": True}))
                    logger.info("Sent interrupt signal to client.")
                except websockets.exceptions.ConnectionClosed:
                     pass # Ignore if connection closed during interrupt
                except Exception as e:
                     logger.error(f"Failed to send interrupt signal: {e}")

                # Give cancellation a moment to propagate
                await asyncio.sleep(0.1) # Small delay to allow task cancellation
                self.current_processing_task = None
                self.is_responding = False # Ensure flag is reset

    async def process_audio_chunk(self, audio_bytes):
        """Processes an incoming audio chunk using VAD."""
        async with self.vad_lock:
            self.audio_buffer.extend(audio_bytes)
            segment_to_process = None

            # Simple energy-based VAD (can be replaced with a more sophisticated VAD)
            # Analyze the *new* chunk's energy primarily
            if len(audio_bytes) > 0:
                 audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                 energy = np.sqrt(np.mean(audio_array**2)) if len(audio_array) > 0 else 0

                 if self.is_speech_active:
                     if energy > AUDIO_ENERGY_THRESHOLD:
                         self.silence_counter = 0 # Reset silence counter
                     else:
                         self.silence_counter += len(audio_array)

                     segment_duration_samples = len(self.audio_buffer)
                     segment_duration_seconds = segment_duration_samples / 2.0 / AUDIO_SAMPLE_RATE

                     # Check for end of speech (silence duration met)
                     if self.silence_counter >= AUDIO_SILENCE_DURATION * AUDIO_SAMPLE_RATE:
                         speech_end_idx = len(self.audio_buffer) - self.silence_counter
                         # Ensure minimum length before processing
                         if segment_duration_seconds >= AUDIO_MIN_SPEECH_DURATION:
                             segment_to_process = bytes(self.audio_buffer[:speech_end_idx])
                             logger.info(f"Speech segment detected by silence ({segment_duration_seconds:.2f}s)")
                         else:
                              logger.info(f"Discarding short segment detected by silence ({segment_duration_seconds:.2f}s)")
                         await self._reset_vad_state()

                     # Check for maximum speech duration
                     elif segment_duration_seconds > AUDIO_MAX_SPEECH_DURATION:
                         segment_to_process = bytes(self.audio_buffer)
                         logger.info(f"Speech segment cut by max duration ({segment_duration_seconds:.2f}s)")
                         await self._reset_vad_state() # Reset VAD state after max duration cut

                 elif energy > AUDIO_ENERGY_THRESHOLD:
                     # Speech start detected
                     self.is_speech_active = True
                     self.silence_counter = 0
                     self.speech_start_time = time.time()
                     # If system is currently speaking, interrupt it
                     if self.is_responding:
                         await self._cancel_current_processing()
                     logger.info(f"Speech start detected (Energy: {energy:.4f})")
                     # Adjust buffer start? Might lose context. Keep full buffer for now.
                     # self.audio_buffer = self.audio_buffer[-len(audio_bytes):] # Start buffer from this chunk

            # Process the detected segment outside the VAD lock
            if segment_to_process:
                # Don't start processing if already busy (e.g., interrupted but task takes time to cancel)
                if not self.is_responding:
                     # Create the main processing task
                     async with self.processing_lock: # Ensure task creation and flag setting is atomic
                          self.is_responding = True
                          self.current_processing_task = asyncio.create_task(
                               self.handle_speech_segment(segment_to_process),
                               name=f"ProcessingSegment_{time.time()}"
                          )
                else:
                     logger.warning("New segment detected, but system is already processing/cancelling. Ignoring.")


    async def handle_speech_segment(self, audio_segment_bytes):
        """Handles the full pipeline for a detected speech segment."""
        try:
            transcription_start_time = time.time()
            transcription = await self.transcriber.transcribe(audio_segment_bytes)
            transcription_end_time = time.time()
            logger.info(f"Transcription took {transcription_end_time - transcription_start_time:.2f}s")

            if not transcription or not any(c.isalnum() for c in transcription):
                logger.info("Empty or non-alphanumeric transcription, skipping.")
                return # Exit processing early

            # Filter common filler sounds / short utterances
            filler_patterns = [
                r'^(um+|uh+|ah+|oh+|hm+|mhm+|hmm+)$',
                r'^(okay|yes|no|yeah|nah|bye)$', # Added bye
                r'^(thanks|thank you)$' # Optional: filter thank you?
            ]
            # Check word count after potential punctuation removal for robustness
            words = re.findall(r'\b\w+\b', transcription.lower())
            if len(words) <= 1:
                 if not any(re.fullmatch(pattern, transcription.lower()) for pattern in filler_patterns):
                     logger.info(f"Skipping single-word transcription (not filler): '{transcription}'")
                 else:
                     logger.info(f"Skipping filler/short transcription: '{transcription}'")
                 return


            logger.info(f"Processing transcription: '{transcription}'")
            generation_start_time = time.time()

            # Start streaming generation
            gemma_streamer = await self.gemma_processor.generate_streaming(transcription)

            accumulated_text_chunk = ""
            sentence_end_pattern = re.compile(TTS_CHUNK_SPLIT_PATTERN)
            min_chunk_len = 20 # Minimum characters before attempting TTS

            async for text_fragment in gemma_streamer:
                # Check for cancellation request at each fragment
                # asyncio.CancelledError will be raised here if cancelled
                await asyncio.sleep(0) # Yield control briefly

                accumulated_text_chunk += text_fragment

                # Check if we have a potential chunk end (punctuation or length)
                # Use search to find punctuation anywhere in the *new* fragment or end of accumulated
                found_punctuation = sentence_end_pattern.search(text_fragment)
                should_synthesize = (found_punctuation and len(accumulated_text_chunk) >= min_chunk_len) or \
                                    (len(accumulated_text_chunk) > 80) # Force synth on longer chunks w/o punctuation

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

                                await self.websocket.send(json.dumps({"audio": base64_audio}))
                                logger.info(f"Sent audio chunk ({len(audio_bytes)} bytes)")
                            except websockets.exceptions.ConnectionClosed:
                                logger.warning("Connection closed while sending audio chunk.")
                                raise asyncio.CancelledError # Stop processing if connection closed
                            except Exception as e:
                                logger.error(f"Error sending audio chunk: {e}")
                                raise asyncio.CancelledError # Stop on send error


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
                         await self.websocket.send(json.dumps({"audio": base64_audio}))
                         logger.info(f"Sent final audio chunk ({len(audio_bytes)} bytes)")
                     except websockets.exceptions.ConnectionClosed:
                         logger.warning("Connection closed while sending final audio chunk.")
                     except Exception as e:
                         logger.error(f"Error sending final audio chunk: {e}")

            generation_end_time = time.time()
            logger.info(f"Full response generation & synthesis took {generation_end_time - generation_start_time:.2f}s")

        except asyncio.CancelledError:
            logger.info("Processing task was cancelled.")
            # History update is handled within the Gemma streamer's finally block if needed
        except websockets.exceptions.ConnectionClosed:
             logger.info("Connection closed during processing task.")
        except Exception as e:
            logger.error(f"Error during speech segment processing: {e}", exc_info=True)
            # Optionally send an error message to the client
            try:
                 await self.websocket.send(json.dumps({"error": "An internal error occurred."}))
            except:
                 pass # Ignore errors sending the error message
        finally:
            # Ensure the responding flag and task reference are cleared
            async with self.processing_lock:
                self.is_responding = False
                # Check if the current task is indeed this one before clearing
                # (Avoid race condition if a new task started somehow)
                # This check might be overly cautious if processing_lock is held correctly
                # if self.current_processing_task is asyncio.current_task():
                self.current_processing_task = None
            logger.info("Finished processing segment.")


    async def handle_messages(self):
        """Main loop to receive messages from the WebSocket client."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)

                    # Handle image data first - Allow image updates anytime
                    if "image" in data:
                         image_data = base64.b64decode(data["image"])
                         await self.gemma_processor.set_image(image_data)
                         # If an image arrives while responding, should we interrupt?
                         # Current logic: New speech interrupts, new image just updates context for *next* turn.
                         # Could add: await self._cancel_current_processing() if self.is_responding else None

                    # Handle audio data chunks (assuming custom format)
                    # Adapt this if your client sends audio differently
                    elif "audio_chunk" in data: # Example: client sends {'audio_chunk': 'base64pcm'}
                         audio_data = base64.b64decode(data["audio_chunk"])
                         await self.process_audio_chunk(audio_data)

                    # Handle combined realtime data (like original code)
                    elif "realtime_input" in data:
                        for chunk in data["realtime_input"]["media_chunks"]:
                             if chunk["mime_type"] == "audio/pcm":
                                 audio_data = base64.b64decode(chunk["data"])
                                 await self.process_audio_chunk(audio_data)
                             elif chunk["mime_type"] == "image/jpeg":
                                 image_data = base64.b64decode(chunk["data"])
                                 await self.gemma_processor.set_image(image_data)
                                 # Optional: Interrupt on image within stream?
                                 # if self.is_responding: await self._cancel_current_processing()

                    # Handle other message types (e.g., configuration, text commands)
                    elif "config" in data:
                         logger.info(f"Received config message: {data}")
                         # Process config if needed

                except json.JSONDecodeError:
                    logger.error(f"Received invalid JSON: {message[:100]}...")
                except KeyError as e:
                    logger.error(f"Missing key in message: {e} - Data: {data}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)

        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"Client {self.websocket.remote_address} disconnected normally.")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"Client {self.websocket.remote_address} disconnected with error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in message handling loop: {e}", exc_info=True)
        finally:
            logger.info("Cleaning up client handler resources...")
            # Ensure any running task is cancelled on disconnect
            await self._cancel_current_processing()
            # Release resources if necessary (e.g., clear large buffers)
            self.audio_buffer = bytearray()
            logger.info(f"Client handler finished cleanup for {self.websocket.remote_address}")


async def websocket_handler(websocket, path):
    """Handles a new WebSocket connection."""
    logger.info(f"Client connected from {websocket.remote_address}")
    client_handler = ClientHandler(websocket)
    # Keep the connection alive by running the message handler
    await client_handler.handle_messages()
    logger.info(f"Connection handler finished for {websocket.remote_address}")


async def main():
    """Initializes models and starts the WebSocket server."""
    # Initialize singletons eagerly to load models on startup
    logger.info("Pre-loading models...")
    try:
        WhisperTranscriber()
        GemmaMultimodalProcessor()
        KokoroTTSProcessor()
        logger.info("Models initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize models: {e}. Server cannot start.", exc_info=True)
        return # Exit if models fail to load

    logger.info(f"Starting WebSocket server on 0.0.0.0:{WEBSOCKET_PORT}")

    # Configure server with ping/pong for keepalive
    server = websockets.serve(
        websocket_handler,
        "0.0.0.0",
        WEBSOCKET_PORT,
        ping_interval=WEBSOCKET_PING_INTERVAL,
        ping_timeout=WEBSOCKET_PING_TIMEOUT,
        close_timeout=10,
        # Increase max message size if needed (e.g., for large images)
        # max_size= 2**24 # 16MB example
    )

    try:
        async with server:
            logger.info(f"WebSocket server running.")
            await asyncio.Future()  # Run forever until interrupted
    except asyncio.CancelledError:
         logger.info("Server main task cancelled.")
    except Exception as e:
        logger.error(f"WebSocket server error: {e}", exc_info=True)
    finally:
         logger.info("WebSocket server shutting down.")
         # Server context manager handles closing connections

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt).")
