import asyncio
import json
import websockets
import base64
import torch
import numpy as np
import logging
import sys
import time
from accelerate import Accelerator
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoModelForCausalLM, # Use for Gemma text models
    AutoProcessor,         # Use for Whisper
    AutoTokenizer,         # Use for Gemma text models
    pipeline,
    TextIteratorStreamer,
    BitsAndBytesConfig # Keep import if you want to easily add quantization later
)

# Optional TTS - Attempt to import Kokoro
try:
    # Ensure you have the correct import for your Kokoro installation
    from kokoro import KPipeline # Replace if your import path is different
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    # Define a dummy KPipeline if not available to prevent errors later
    class KPipeline:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs):
             print("Warning: KokoroTTS called, but library is not available.")
             # Yield empty audio or handle differently as needed
             yield None, None, np.array([], dtype=np.float32)

# --- Configuration Constants ---
WHISPER_MODEL_ID = "openai/whisper-large-v3"
# *** Using Gemma 7B Instruct (Text-Only) as requested ***
GEMMA_MODEL_ID = "google/gemma-7b-it"
WEBSOCKET_PORT = 9073
AUDIO_SAMPLE_RATE = 16000
DEFAULT_TTS_VOICE = 'en_amy' # Example Kokoro voice if using TTS (adjust as needed)
TORCH_DTYPE = torch.bfloat16 # Recommended for modern GPUs (Ampere+)
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
# Automatically select best available attention implementation
# Gemma supports FA2 and SDPA. Whisper might require specific checks or None.
# Let's try using it for both where applicable.
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

            # Clear the segment queue as well if tasks are cancelled
            while not self.segment_queue.empty():
                try:
                    self.segment_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            await self.set_tts_playing(False) # Ensure TTS is marked as stopped
            # if cancelled_something:
            #     logger.info("Cancellation complete.")

    async def set_current_tasks(self, processing_task=None):
        """Sets the current active processing task."""
        async with self.task_lock:
            self.current_processing_task = processing_task
            # logger.debug(f"Set current task: {processing_task is not None}")

    async def add_audio(self, audio_bytes):
        """Adds audio data and detects speech segments based on VAD logic."""
        async with self.tts_lock:
            if self.tts_playing:
                # logger.debug("TTS playing, discarding incoming audio chunk.")
                return None # Don't process audio if TTS is active

        async with self.lock:
            self.audio_buffer.extend(audio_bytes)
            chunk_samples = len(audio_bytes) // 2 # 2 bytes per int16 sample
            if chunk_samples == 0: return None

            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            # Avoid division by zero if audio_array is all zeros
            if np.any(audio_array):
                energy = np.sqrt(np.mean(audio_array**2))
            else:
                energy = 0.0


            detected_segment = None # Store detected segment here

            if not self.is_speech_active:
                if energy > self.energy_threshold:
                    self.is_speech_active = True
                    self.speech_start_idx = len(self.audio_buffer) - len(audio_bytes)
                    self.silence_counter = 0
                    logger.info(f"Speech start detected (energy: {energy:.6f})")
            else: # Speech is active
                if energy > self.energy_threshold:
                    self.silence_counter = 0
                else:
                    self.silence_counter += chunk_samples
                    # logger.debug(f"Silence counter: {self.silence_counter}/{self.silence_samples}")

                    # Silence threshold met?
                    if self.silence_counter >= self.silence_samples:
                        speech_end_idx = len(self.audio_buffer) - self.silence_counter * 2
                        segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])
                        segment_duration_samples = len(segment_bytes) // 2

                        # Reset state *after* extracting segment
                        self.is_speech_active = False
                        self.audio_buffer = self.audio_buffer[speech_end_idx:] # Keep remaining buffer
                        self.speech_start_idx = 0
                        self.silence_counter = 0

                        if segment_duration_samples >= self.min_speech_samples:
                            logger.info(f"Speech segment detected (Silence): {segment_duration_samples / self.sample_rate:.2f}s")
                            detected_segment = segment_bytes
                        else:
                            logger.info(f"Discarding short segment (Silence): {segment_duration_samples / self.sample_rate:.2f}s")

                # Max duration check (only if speech still active)
                # Ensure speech_start_idx is valid before calculating duration
                current_buffer_len = len(self.audio_buffer)
                if self.is_speech_active and self.speech_start_idx < current_buffer_len:
                    current_speech_duration_samples = (current_buffer_len - self.speech_start_idx) // 2
                    if current_speech_duration_samples > self.max_speech_samples:
                        speech_end_idx = self.speech_start_idx + self.max_speech_samples * 2
                        segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])

                        # Reset state differently for max duration cut
                        self.audio_buffer = self.audio_buffer[speech_end_idx:]
                        self.speech_start_idx = 0 # Next segment starts from beginning of new buffer
                        self.silence_counter = 0 # Reset silence as we forced a cut

                        logger.info(f"Max duration speech segment extracted: {len(segment_bytes) / 2 / self.sample_rate:.2f}s")
                        detected_segment = segment_bytes
                elif self.is_speech_active and self.speech_start_idx >= current_buffer_len:
                    # This case indicates an issue, potentially buffer was cleared unexpectedly. Reset state.
                    logger.warning("Inconsistent state in VAD (speech_start_idx >= buffer length). Resetting.")
                    self.is_speech_active = False
                    self.speech_start_idx = 0
                    self.silence_counter = 0


            # If a segment was detected (either by silence or max duration)
            if detected_segment:
                self.segments_detected += 1
                await self.cancel_current_tasks() # Cancel any ongoing processing
                # Ensure queue is empty before putting (should be due to cancel)
                while not self.segment_queue.empty():
                    try:
                        self.segment_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                await self.segment_queue.put(detected_segment)
                return detected_segment

            return None # No segment completed in this chunk

    async def get_next_segment(self):
        """Retrieves the next detected speech segment from the queue."""
        try:
            return await asyncio.wait_for(self.segment_queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            return None
# ---

# --- Whisper Transcriber Class ---
class WhisperTranscriber:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Initializing WhisperTranscriber...")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if WhisperTranscriber._instance is not None:
            raise Exception("WhisperTranscriber is a singleton!")

        self.accelerator = Accelerator()
        # Use device_map for potential multi-GPU distribution of Whisper
        logger.info(f"Attempting to load Whisper model {WHISPER_MODEL_ID} using device_map='auto' with {TORCH_DTYPE}")

        try:
            # Note: attn_implementation might need verification for Whisper + device_map
            # Start without it, add if stable and beneficial
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                WHISPER_MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                device_map="auto", # Use accelerate for multi-GPU placement
                # attn_implementation=ATTN_IMPLEMENTATION # Maybe add later if needed/stable
            )
            self.processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)

            # Pipeline setup needs care with device_map
            # Often better to handle inference directly or ensure pipeline respects map
            # Let's try direct model inference without pipeline for more control
            self.transcription_count = 0
            logger.info(f"Whisper model {WHISPER_MODEL_ID} loaded successfully using device_map.")
            # Log actual device placement if possible (device_map hides specifics easily)
            try:
                logger.info(f"Whisper device map: {self.model.hf_device_map}")
            except AttributeError:
                 logger.info("Could not retrieve detailed Whisper device map.")


        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            raise

    # Updated transcribe method to use model directly, suitable for device_map
    async def transcribe(self, audio_bytes, sample_rate=AUDIO_SAMPLE_RATE):
        if not audio_bytes or len(audio_bytes) < 1000: # Basic check
             logger.warning(f"Audio too short for transcription ({len(audio_bytes)} bytes)")
             return ""
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Prepare inputs using the processor - This moves data to appropriate device based on model map
            inputs = self.processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
            # Manually move inputs if processor doesn't handle device_map well, though it should
            # inputs = inputs.to(self.model.device) # Usually not needed with device_map in model

            # Generate transcription tokens - device_placement ensures parts run on correct GPUs
            with torch.no_grad(): # Ensure no gradients are calculated
                predicted_ids = await asyncio.get_event_loop().run_in_executor(
                    None, # Default executor
                    lambda: self.model.generate(
                        inputs["input_features"].to(TORCH_DTYPE), # Ensure dtype match
                        # Add generation config arguments if needed (language, task etc.)
                        language="english",
                        task="transcribe",
                        max_new_tokens=128 # Adjust as needed
                    )
                )

            # Decode the tokens using the processor
            # Ensure skip_special_tokens is True to remove BOS, EOS etc.
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            text = transcription.strip()

            self.transcription_count += 1
            logger.info(f"Transcription #{self.transcription_count}: '{text}'")
            return text

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return ""
# ---

# --- Gemma Text Processor Class ---
class GemmaTextProcessor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Initializing GemmaTextProcessor...")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if GemmaTextProcessor._instance is not None:
            raise Exception("GemmaTextProcessor is a singleton!")

        self.accelerator = Accelerator() # Accelerator manages device mapping
        logger.info(f"Attempting to load Gemma model {GEMMA_MODEL_ID} using device_map='auto' with {TORCH_DTYPE}")

        try:
            # Use AutoTokenizer for text models
            self.tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID)
            self.model = AutoModelForCausalLM.from_pretrained(
                GEMMA_MODEL_ID,
                device_map="auto", # Key for multi-GPU & fitting large models
                torch_dtype=TORCH_DTYPE,
                attn_implementation=ATTN_IMPLEMENTATION # Use FA2 or SDPA
            )

            self.lock = asyncio.Lock() # Lock for accessing shared history state
            self.message_history = [] # Store conversation history
            self.generation_count = 0
            logger.info(f"Gemma model {GEMMA_MODEL_ID} loaded successfully using device_map.")
            # Log actual device placement if possible
            try:
                 logger.info(f"Gemma device map: {self.model.hf_device_map}")
            except AttributeError:
                 logger.info("Could not retrieve detailed Gemma device map.")

        except Exception as e:
             logger.error(f"Failed to load Gemma model: {e}", exc_info=True)
             raise

    # Builds the chat prompt using the tokenizer's template
    def _build_prompt(self, text):
        # Construct the messages list for apply_chat_template
        # System prompt (optional but often helpful)
        # messages = [{"role": "system", "content": "You are a helpful voice assistant."}]
        messages = [] # Start clean or add system prompt above

        # Add history - keep it simple: last exchange only
        # For longer context, append user/assistant turns carefully
        messages.extend(self.message_history)

        # Add current user turn
        messages.append({"role": "user", "content": text})

        # apply_chat_template handles the <start_of_turn>, <end_of_turn> tokens
        # tokenize=False returns the formatted string prompt
        # add_generation_prompt=True adds the prompt for the model to continue ('<start_of_turn>model\n')
        try:
            # Tokenize=True returns input_ids, attention_mask etc. directly
             prompt_data = self.tokenizer.apply_chat_template(
                 messages,
                 tokenize=True,
                 add_generation_prompt=True,
                 return_tensors="pt"
             )
             return prompt_data # This is now a dict of tensors
        except Exception as e:
             logger.error(f"Error applying chat template: {e}")
             # Fallback: create a simple prompt manually (might lack special tokens)
             fallback_prompt = f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
             return self.tokenizer(fallback_prompt, return_tensors="pt")


    # Updates history (simple version: keep only last Q&A)
    def _update_history(self, user_text, assistant_response):
        # Keep only the last user message and assistant response
        self.message_history = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_response}
        ]
        # Optional: Limit history length (e.g., keep last N turns)


    async def generate_streaming(self, text):
        """Generates text response stream using Gemma."""
        try:
            async with self.lock: # Ensure history is read consistently
                 inputs = self._build_prompt(text).to(self.model.device) # Build prompt and move to device

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,      # Don't include the input prompt in the output stream
                skip_special_tokens=True # Remove special tokens like <bos>, <eos>
            )

            # Generation settings
            generation_kwargs = dict(
                **inputs, # Pass input_ids, attention_mask
                streamer=streamer,
                max_new_tokens=256, # Adjust response length as needed
                do_sample=False,    # Use greedy decoding for deterministic output
                # temperature=0.7,  # For sampling
                # top_p=0.9,        # For sampling
                use_cache=True,     # Important for speed
                pad_token_id=self.tokenizer.eos_token_id # Avoid warning
             )

            # Run blocking model.generate in a separate thread via executor
            loop = asyncio.get_running_loop()
            generation_future = loop.run_in_executor(
                 None, # Default executor
                 lambda: self.model.generate(**generation_kwargs)
            )
            # Note: We don't await generation_future here directly.
            # The streamer will yield results as the thread runs.

            # --- Stream initial part for faster TTS response ---
            initial_text = ""
            timeout_seconds = 5.0 # Max wait for the *first* chunk
            start_time = time.time()
            first_chunk_received = False
            try:
                 async for chunk in streamer: # Use async for iteration
                      initial_text += chunk
                      first_chunk_received = True
                      # Heuristic: break after first sentence-like structure or enough chars
                      if time.time() - start_time > timeout_seconds and not initial_text:
                           logger.warning("Timeout waiting for initial text chunk from Gemma.")
                           break # Stop waiting if timeout hit before *any* text
                      if "." in chunk or "!" in chunk or "?" in chunk or "\n" in chunk or len(initial_text) > 30:
                           break # Break after potential sentence end or enough content
                 else:
                      # Streamer finished before heuristic break (short response)
                      if first_chunk_received:
                          logger.info("Gemma stream finished before initial break condition.")
                      else:
                          logger.warning("Gemma stream finished without yielding any text.")

            except asyncio.CancelledError:
                 logger.info("Gemma initial streaming cancelled.")
                 # Ensure the background generation task is also cancelled/handled if needed
                 if not generation_future.done(): generation_future.cancel()
                 raise # Re-raise cancellation
            except Exception as e:
                 logger.error(f"Error during initial Gemma streaming: {e}", exc_info=True)
                 if not generation_future.done(): generation_future.cancel()
                 # Fallback: return error message, streamer is now likely broken
                 return None, "Sorry, I encountered an error during generation."

            self.generation_count += 1
            logger.info(f"Gemma Initial Text #{self.generation_count}: '{initial_text.strip()}'")

            # Return the streamer (for consuming remaining text) and the initial text
            # The background generation task continues implicitly via the streamer
            return streamer, initial_text.strip()

        except Exception as e:
            logger.error(f"Gemma streaming setup error: {e}", exc_info=True)
            return None, "Sorry, I couldn’t prepare the response generator."
# ---

# --- Kokoro TTS Processor Class ---
class KokoroTTSProcessor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if not KOKORO_AVAILABLE:
             logger.warning("Attempted to get KokoroTTS instance, but library not available.")
             return None # Return None if Kokoro is not installed
        if cls._instance is None:
            logger.info("Initializing KokoroTTSProcessor...")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if not KOKORO_AVAILABLE:
            self.pipeline = None
            return # Don't initialize if lib not found

        if KokoroTTSProcessor._instance is not None:
            raise Exception("KokoroTTSProcessor is a singleton!")

        try:
            # lang_code='a' for auto-detect? Check Kokoro docs. 'en' might be safer.
            self.pipeline = KPipeline(lang_code='en') # Assuming English focus
            self.default_voice = DEFAULT_TTS_VOICE
            self.synthesis_count = 0
            logger.info(f"Kokoro TTS loaded with default voice {self.default_voice}")
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS: {e}", exc_info=True)
            self.pipeline = None # Ensure pipeline is None on error

    async def synthesize_speech(self, text):
        if not self.pipeline:
             logger.error("KokoroTTS pipeline not available for synthesis.")
             return None
        if not text or not text.strip():
            logger.warning("TTS received empty text, skipping synthesis.")
            return None

        try:
            # Use run_in_executor as Kokoro might be blocking
            def sync_synthesize():
                audio_segments = []
                # Generator needs to be fully consumed in the executor thread
                try:
                     # Adjust split pattern if needed. Basic punctuation split:
                     generator = self.pipeline(text, voice=self.default_voice, speed=1.0, split_pattern=r'[.!?]+')
                     for _, _, audio in generator:
                         if audio is not None and audio.size > 0:
                              audio_segments.append(audio)
                except Exception as synth_err:
                     logger.error(f"Error during Kokoro synthesis call: {synth_err}")
                     return None # Return None on internal error

                if not audio_segments: return None
                return np.concatenate(audio_segments)

            # Run the synchronous synthesis function in the default executor
            combined_audio = await asyncio.get_event_loop().run_in_executor(None, sync_synthesize)

            if combined_audio is not None and combined_audio.size > 0:
                self.synthesis_count += 1
                # Assuming 16kHz output from Kokoro, adjust if different
                logger.info(f"TTS Synthesis #{self.synthesis_count}: '{text[:50]}...' -> {combined_audio.shape[0] / AUDIO_SAMPLE_RATE:.2f}s audio")
                return combined_audio
            else:
                logger.warning(f"TTS synthesis resulted in empty audio for text: '{text[:50]}...'")
                return None
        except Exception as e:
            # Catch errors during the executor call itself
            logger.error(f"Kokoro TTS synthesis task error: {e}", exc_info=True)
            return None
# ---

# --- WebSocket Handler ---
async def handle_client(websocket): # path argument is provided by websockets
    client_address = websocket.remote_address
    logger.info(f"Client connected: {client_address}")

    # Create instances specific to this connection for VAD state
    detector = AudioSegmentDetector()
    # Get shared model instances (initialized at startup)
    transcriber = WhisperTranscriber.get_instance()
    gemma_processor = GemmaTextProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance() # Might be None

    # Task for processing detected speech segments (transcription -> generation -> TTS)
    current_speech_processing_task = None

    async def process_speech_segment(segment_bytes):
        """Handles the full pipeline for a detected speech segment."""
        nonlocal current_speech_processing_task
        start_time = time.time()
        try:
            logger.info(f"Processing speech segment of {len(segment_bytes)/2/AUDIO_SAMPLE_RATE:.2f}s")

            # 1. Transcribe
            transcription = await transcriber.transcribe(segment_bytes)
            if not transcription or not any(c.isalnum() for c in transcription):
                logger.info(f"Skipping empty or non-alphanumeric transcription: '{transcription}'")
                await detector.set_tts_playing(False) # Ensure TTS state is reset
                return # Stop processing this segment

            logger.info(f"User said: '{transcription}' ({(time.time() - start_time):.2f}s)")
            transcription_time = time.time()

            # Mark TTS as potentially starting (Gemma might fail)
            await detector.set_tts_playing(True)

            # 2. Generate Response (Streaming)
            gemma_streamer, initial_text = await gemma_processor.generate_streaming(transcription)
            generation_start_time = time.time()
            logger.info(f"Gemma initial generation took: {(generation_start_time - transcription_time):.2f}s")


            # Handle potential Gemma errors
            if gemma_streamer is None:
                logger.error("Gemma generation failed, sending error TTS if possible.")
                if tts_processor:
                    error_audio = await tts_processor.synthesize_speech(initial_text) # Contains error msg
                    if error_audio is not None:
                        audio_bytes = (error_audio * 32767).astype(np.int16).tobytes()
                        await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
                await detector.set_tts_playing(False) # Reset TTS state on error
                return # Stop processing this segment

            # 3. Synthesize and send initial part of the response quickly
            initial_audio = None
            if initial_text and tts_processor:
                tts_start_time = time.time()
                initial_audio = await tts_processor.synthesize_speech(initial_text)
                tts_end_time = time.time()
                if initial_audio is not None:
                    audio_bytes = (initial_audio * 32767).astype(np.int16).tobytes()
                    logger.info(f"Sending initial TTS audio ({len(audio_bytes)/2/AUDIO_SAMPLE_RATE:.2f}s) (TTS took {(tts_end_time - tts_start_time):.2f}s)")
                    await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
                else:
                     logger.warning("Initial TTS synthesis failed.")

            # 4. Process the rest of the Gemma stream and synthesize remaining text
            remaining_text = ""
            full_response = initial_text # Start building full response for history
            try:
                async for chunk in gemma_streamer:
                     remaining_text += chunk
                     full_response += chunk
                logger.info(f"Finished consuming Gemma stream. Total length: {len(full_response)}")
            except asyncio.CancelledError:
                 logger.info("Gemma streaming iteration cancelled.")
                 remaining_text = "" # Don't synthesize if cancelled
            except Exception as e:
                logger.error(f"Error iterating Gemma streamer: {e}")
                remaining_text = "" # Don't synthesize on error

            # Synthesize and send the rest of the audio
            if remaining_text.strip() and tts_processor:
                tts_remaining_start = time.time()
                remaining_audio = await tts_processor.synthesize_speech(remaining_text.strip())
                tts_remaining_end = time.time()
                if remaining_audio is not None:
                    audio_bytes = (remaining_audio * 32767).astype(np.int16).tobytes()
                    logger.info(f"Sending remaining TTS audio ({len(audio_bytes)/2/AUDIO_SAMPLE_RATE:.2f}s) (TTS took {(tts_remaining_end - tts_remaining_start):.2f}s)")
                    await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
                else:
                     logger.warning("Remaining TTS synthesis failed.")

            # 5. Update Gemma's conversation history (only if generation was successful)
            if gemma_streamer is not None: # Check if generation started successfully
                 async with gemma_processor.lock: # Need lock to update history safely
                     gemma_processor._update_history(transcription, full_response.strip())
                 logger.debug("Gemma history updated.")

        except asyncio.CancelledError:
            logger.info(f"Speech processing task cancelled.")
            # Ensure TTS state is reset if cancelled mid-way
            await detector.set_tts_playing(False)
            raise # Re-raise cancellation

        except Exception as e:
            logger.error(f"Error in process_speech_segment: {e}", exc_info=True)
            if tts_processor: # Attempt to send generic error TTS
                try:
                    error_audio = await tts_processor.synthesize_speech("Sorry, an error occurred.")
                    if error_audio is not None:
                        audio_bytes = (error_audio * 32767).astype(np.int16).tobytes()
                        await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
                except Exception as send_e:
                    logger.error(f"Failed to send error TTS: {send_e}")
        finally:
            # Ensure TTS playing is set to false when processing finishes or errors out
            # unless it was explicitly cancelled (handled in cancel block)
            if not asyncio.current_task().cancelled():
                 await detector.set_tts_playing(False)
            # Clear the reference to this task in the handler's scope
            current_speech_processing_task = None
            end_time = time.time()
            logger.info(f"Total segment processing time: {(end_time - start_time):.2f}s")


    # Coroutine to continuously check for and process speech segments
    async def speech_segment_consumer():
        nonlocal current_speech_processing_task
        while True:
            segment = await detector.get_next_segment() # Waits briefly for a segment
            if segment:
                # Cancel any existing processing task *before* starting a new one
                if current_speech_processing_task and not current_speech_processing_task.done():
                     logger.info("New segment detected, cancelling previous processing task.")
                     current_speech_processing_task.cancel()
                     try:
                         await current_speech_processing_task # Wait for cancellation
                     except asyncio.CancelledError:
                         logger.info("Previous task cancellation confirmed.")
                     except Exception as e:
                         logger.error(f"Error waiting for previous task cancellation: {e}")

                # Start processing the new segment
                logger.info("Creating new speech processing task.")
                current_speech_processing_task = asyncio.create_task(process_speech_segment(segment))
                # Link the task in the detector for potential external cancellation if needed
                await detector.set_current_tasks(processing_task=current_speech_processing_task)

            # Prevent tight loop if queue is empty
            await asyncio.sleep(0.01)


    # Coroutine to handle incoming WebSocket messages (audio ONLY)
    async def receive_data():
        async for message in websocket:
            try:
                data = json.loads(message)

                # Handle audio data ONLY
                if "realtime_input" in data and "media_chunks" in data["realtime_input"]:
                     for chunk in data["realtime_input"]["media_chunks"]:
                         if chunk.get("mime_type") == "audio/pcm":
                             audio_data = base64.b64decode(chunk["data"])
                             await detector.add_audio(audio_data) # Feeds the VAD
                         # else: ignore other mime types (like image/jpeg)

                # Ignore standalone 'image' keys or other formats
                # if "image" in data:
                #     logger.debug("Ignoring received image data (text-only mode).")

            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message from {client_address}. Ignoring.")
            except websockets.exceptions.ConnectionClosed:
                 logger.info("Connection closed during receive.")
                 break # Exit loop on connection close
            except Exception as e:
                logger.error(f"Error processing received message: {e}", exc_info=True)


    # Keepalive task
    async def send_keepalive():
        while True:
            try:
                await websocket.ping()
                await asyncio.sleep(20) # Send ping every 20 seconds
            except asyncio.CancelledError:
                logger.info("Keepalive task cancelled.")
                break
            except websockets.exceptions.ConnectionClosed:
                logger.info("Keepalive task: Connection closed.")
                break # Exit loop if connection is closed
            except Exception as e:
                 logger.error(f"Keepalive task error: {e}")
                 await asyncio.sleep(20) # Wait before retrying ping


    # Main management loop for the connection
    receive_task = None
    consumer_task = None
    keepalive_task = None
    try:
        logger.info(f"Starting tasks for client {client_address}")
        consumer_task = asyncio.create_task(speech_segment_consumer(), name=f"Consumer_{client_address}")
        receive_task = asyncio.create_task(receive_data(), name=f"Receiver_{client_address}")
        keepalive_task = asyncio.create_task(send_keepalive(), name=f"Keepalive_{client_address}")

        # Wait for any of the essential tasks to finish (e.g., connection closed in receiver/keepalive)
        done, pending = await asyncio.wait(
            [receive_task, keepalive_task, consumer_task], # Consumer finishing isn't necessarily an end condition
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            try:
                 result = task.result() # Check for exceptions in completed tasks
                 logger.info(f"Task {task.get_name()} finished.")
            except websockets.exceptions.ConnectionClosed:
                 logger.info(f"Task {task.get_name()} finished due to connection close.")
            except asyncio.CancelledError:
                 logger.info(f"Task {task.get_name()} was cancelled.")
            except Exception as e:
                 logger.error(f"Task {task.get_name()} finished with error: {e}", exc_info=True)


    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Client {client_address} disconnected cleanly: {e.code} {e.reason}")
    except Exception as e:
        logger.error(f"WebSocket handler error for {client_address}: {e}", exc_info=True)
    finally:
        logger.info(f"Cleaning up connection for {client_address}")
        # Cancel all related tasks explicitly
        all_tasks = [receive_task, consumer_task, keepalive_task, current_speech_processing_task]
        for task in all_tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task # Allow cancellation to propagate
                except asyncio.CancelledError:
                    pass # Expected
                except Exception as e:
                    logger.error(f"Error during final task cancellation ({task.get_name()}): {e}")
        logger.info(f"Finished cleanup for {client_address}")
# ---

# --- Main Application Setup ---
async def main():
    # Initialize singletons eagerly at startup
    logger.info("Pre-initializing models...")
    try:
        WhisperTranscriber.get_instance()
        GemmaTextProcessor.get_instance()
        KokoroTTSProcessor.get_instance() # Will be None if Kokoro not installed
        logger.info("Models initialized.")
    except Exception as e:
        logger.error(f"Fatal error during model initialization: {e}", exc_info=True)
        sys.exit(1) # Exit if models can't load

    logger.info(f"Starting WebSocket server on 0.0.0.0:{WEBSOCKET_PORT}")
    # Set higher limits for message size if needed (e.g., for large audio chunks)
    # Increase ping timeout for potentially slower model responses
    async with websockets.serve(handle_client, "0.0.0.0", WEBSOCKET_PORT,
                                 ping_interval=20, ping_timeout=120, close_timeout=10,
                                 # max_size=2*1024*1024 # Example: 2MB limit if needed
                                 ):
        await asyncio.Future() # Run forever until interrupted

if __name__ == "__main__":
    print("-" * 30)
    print("Starting Real-time ASR + LLM (Text-Only) + TTS Server")
    print(f" - Whisper Model: {WHISPER_MODEL_ID}")
    print(f" - Gemma Model:   {GEMMA_MODEL_ID}")
    print(f" - TTS Enabled:   {KOKORO_AVAILABLE}")
    print(f" - Flash Attn:    {FLASH_ATTN_AVAILABLE} ({ATTN_IMPLEMENTATION})")
    print(f" - DType:         {TORCH_DTYPE}")
    print(f" - WebSocket Port:{WEBSOCKET_PORT}")
    print("-" * 30)
    print("Ensure required libraries are installed:")
    print("  pip install transformers torch accelerate websockets numpy")
    print("Optional (highly recommended for speed):")
    print("  pip install flash-attn --no-build-isolation")
    print("Optional (for VRAM saving - requires code changes to enable):")
    print("  pip install bitsandbytes")
    print("Optional (for TTS):")
    print("  pip install kokoro") # Replace with actual Kokoro install command
    print("-" * 30)

    # Consider adding argument parsing here for model IDs, port, etc.

    asyncio.run(main())
# --- End of Script ---
