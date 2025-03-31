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

# --- Singleton Decorator ---
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

class AudioSegmentDetector:
    """Detects speech segments based on audio energy levels and manages interaction flow"""

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
            if self.current_processing_task and not self.current_processing_task.done():
                logger.warning("Cancelling current processing task.")
                self.current_processing_task.cancel()
                try:
                    # Give cancellation a moment to propagate
                    await asyncio.wait_for(self.current_processing_task, timeout=0.5)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    logger.info("Current processing task cancelled or timed out.")
                except Exception as e:
                    logger.error(f"Error during task cancellation: {e}")
                self.current_processing_task = None
            # Ensure state is reset after cancellation attempt
            await self.set_tts_playing(False)
            await self.clear_interrupt_request() # Clear interrupt flag after cancellation

    async def set_current_processing_task(self, task):
        """Set the current main processing task."""
        async with self.processing_task_lock:
            # Cancel previous task if any exists (should have been cancelled by request_interrupt already)
            if self.current_processing_task and not self.current_processing_task.done():
                 logger.warning("Overwriting an existing processing task. Cancelling previous one.")
                 await self.cancel_current_processing() # Ensure cleanup
            self.current_processing_task = task
            if task:
                 await self.set_tts_playing(True) # Mark as busy when a task starts

    async def add_audio(self, audio_bytes):
        """Add audio data and detect speech segments."""
        async with self.buffer_lock:
            self.audio_buffer.extend(audio_bytes)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_array) == 0:
                return

            energy = np.sqrt(np.mean(audio_array**2))

            segment_to_process = None

            if not self.is_speech_active and energy > self.energy_threshold:
                # --- Speech Start ---
                self.is_speech_active = True
                self.speech_start_idx = max(0, len(self.audio_buffer) - len(audio_bytes))
                self.silence_counter = 0
                logger.info(f"Speech start detected (energy: {energy:.6f})")
                # If TTS is currently playing, request an interrupt
                if await self.tts_playing:
                    logger.info("New speech detected while TTS playing. Requesting interrupt.")
                    await self.request_interrupt() # Signal the main processing loop to cancel

            elif self.is_speech_active:
                if energy > self.energy_threshold:
                    self.silence_counter = 0 # Reset silence counter
                else:
                    self.silence_counter += len(audio_array)

                    # --- Speech End by Silence ---
                    if self.silence_counter >= self.silence_samples:
                        speech_end_idx = len(self.audio_buffer) - self.silence_counter
                        speech_segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])

                        # Reset state for next detection
                        self.is_speech_active = False
                        self.silence_counter = 0
                        self.audio_buffer = self.audio_buffer[speech_end_idx:] # Trim buffer

                        if len(speech_segment_bytes) >= self.min_speech_samples * 2: # Check min duration
                            self.segments_detected += 1
                            logger.info(f"Speech segment [silence end]: {len(speech_segment_bytes)/2/self.sample_rate:.2f}s")
                            segment_to_process = speech_segment_bytes
                        else:
                            logger.info(f"Skipping short segment (silence end): {len(speech_segment_bytes)/2/self.sample_rate:.2f}s")


                    # --- Speech End by Max Duration ---
                    elif (len(self.audio_buffer) - self.speech_start_idx) >= self.max_speech_samples * 2:
                        speech_end_idx = self.speech_start_idx + self.max_speech_samples * 2
                        speech_segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])

                        # Reset state but keep buffer overlap for continuation
                        self.speech_start_idx = speech_end_idx
                        self.silence_counter = 0 # Reset silence as speech continues effectively

                        self.segments_detected += 1
                        logger.info(f"Speech segment [max duration]: {len(speech_segment_bytes)/2/self.sample_rate:.2f}s")
                        segment_to_process = speech_segment_bytes

            # Process detected segment outside the main energy check logic
            if segment_to_process:
                 # Check again if TTS is playing *before* putting into queue
                 # This prevents queuing up segments if an interrupt was just requested
                if await self.tts_playing:
                    logger.info("New segment detected while TTS playing. Requesting interrupt (if not already).")
                    await self.request_interrupt()
                    # Don't queue the segment yet, let the interrupt handle the current process
                else:
                    # Only queue if not currently processing/playing TTS
                    logger.info("Queuing detected speech segment.")
                    await self.segment_queue.put(segment_to_process)


    async def get_next_segment(self):
        """Get the next available speech segment non-blockingly."""
        try:
            return self.segment_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

@singleton
class WhisperTranscriber:
    """Handles speech transcription using Whisper."""
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Whisper using device: {self.device}")
        self.torch_dtype = torch.float16 if self.device == "cuda:0" else torch.float32
        model_id = "openai/whisper-large-v3" # large-v3 is good, maybe faster than turbo for some hardware? test if needed
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
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            logger.info("Whisper model ready.")
            self.transcription_count = 0
        except Exception as e:
            logger.critical(f"FATAL: Failed to load Whisper model: {e}", exc_info=True)
            sys.exit(1) # Exit if critical model fails

    async def transcribe(self, audio_bytes):
        """Transcribe audio bytes to text."""
        if len(audio_bytes) < 200: # Check length in bytes (100 samples * 2 bytes/sample)
            logger.warning("Skipping transcription for very short audio segment.")
            return ""
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            logger.info(f"Starting transcription for segment of {len(audio_array)/SAMPLE_RATE:.2f}s...")
            start_time = time.time()

            # Run blocking pipeline call in executor
            result = await asyncio.get_event_loop().run_in_executor(
                None, # Default executor
                lambda: self.pipe(
                    {"array": audio_array, "sampling_rate": SAMPLE_RATE},
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

@singleton
class GemmaMultimodalProcessor:
    """Handles multimodal generation using Gemma."""
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Gemma using device: {self.device}")
        # Using 4b-it requires less VRAM, might be more stable than 9b on some systems
        model_id = "google/gemma-3-4b-it"
        logger.info(f"Loading Gemma model: {model_id}...")
        try:
            # Use bfloat16 if available on GPU, otherwise stick to float16 or float32
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32
            logger.info(f"Gemma using dtype: {dtype}")

            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto", # Let HF handle device placement
                # load_in_8bit=True, # 8-bit can sometimes cause instability or reduce quality. Try without first.
                # load_in_4bit=True, # Consider 4-bit if 8-bit or full precision is too heavy
                torch_dtype=dtype, # Use determined dtype
                attn_implementation="flash_attention_2" if dtype != torch.float32 else None # Use Flash Attention if supported
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            logger.info("Gemma model ready.")
            self.last_image = None
            self.last_image_timestamp = 0
            self.image_lock = asyncio.Lock()
            self.message_history = []
            self.generation_count = 0
        except Exception as e:
            logger.critical(f"FATAL: Failed to load Gemma model: {e}", exc_info=True)
            sys.exit(1) # Exit if critical model fails


    async def set_image(self, image_data):
        """Cache the most recent image, resizing it."""
        async with self.image_lock:
            try:
                image = Image.open(io.BytesIO(image_data)).convert("RGB") # Ensure RGB
                # Smaller resize might speed up processing
                max_size = (1024, 1024) # Gemma might have optimal input sizes, check docs if needed
                image.thumbnail(max_size, Image.Resampling.LANCZOS)

                logger.info(f"New image set (resized to {image.size}). Clearing conversation history.")
                self.last_image = image
                self.last_image_timestamp = time.time()
                self.message_history = [] # Reset history on new image
                return True
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                self.last_image = None # Invalidate image on error
                return False

    def _build_prompt(self, text):
        """Builds the prompt string including history and system message."""
        # Simple system prompt focusing on conversational image description
        system_prompt = """You are a helpful assistant describing images and chatting naturally. Keep responses concise (2-3 sentences), conversational, and directly related to the user's question about the image if applicable. If the question isn't about the image, respond normally. Avoid unnecessary introductions or closings."""

        # Combine history and new message
        chat_template_input = [{"role": "system", "content": system_prompt}]
        # Add history (alternating user/assistant)
        chat_template_input.extend(self.message_history)
        # Add current user message (text only for template)
        chat_template_input.append({"role": "user", "content": text})

        # Use processor's chat template for proper formatting (handles roles, tokens)
        # We need the prompt text to prepend to the image
        # Important: apply_chat_template expects list of dicts for multi-turn
        prompt_text = self.processor.tokenizer.apply_chat_template(
            chat_template_input,
            tokenize=False,
            add_generation_prompt=True # Add the prompt marker for the assistant
        )
        return prompt_text


    async def generate_streaming(self, text):
        """Generate response stream using the latest image and text."""
        async with self.image_lock: # Ensure image doesn't change mid-generation
            if not self.last_image:
                logger.warning("No image available for generation.")
                # Yield a fallback message in the streamer format
                async def fallback_streamer():
                    yield "Sorry, I don't have an image to look at right now."
                return fallback_streamer(), "Sorry, I don't have an image to look at right now." # Return streamer and full text


            prompt = self._build_prompt(text) # Build text prompt using history

            # Prepare inputs FOR THE MODEL (Image + Prompt Text)
            # Important: Use the processor correctly for multimodal input
            inputs = self.processor(
                text=prompt,
                images=self.last_image,
                return_tensors="pt",
                # padding=True # Padding might be handled internally by generate
            ).to(self.model.device, self.model.dtype) # Ensure tensor dtype matches model

        logger.info(f"Starting Gemma generation for prompt: '{text}'")
        start_time = time.time()

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True, # Don't yield the input prompt
            skip_special_tokens=True
        )

        # Generation settings
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=150, # Slightly more tokens if needed
            do_sample=True,
            temperature=0.7,
            top_p=0.9, # Add top_p sampling
            # repetition_penalty=1.1 # Discourage repetition slightly
        )

        # Run generation in a separate thread as it's blocking
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()

        # Return the streamer immediately for consumption
        # We'll collect the full text later for history
        self.generation_count += 1
        logger.info(f"Gemma generation #{self.generation_count} thread started.")

        # This function now primarily returns the streamer
        # The calling function will handle consuming it and collecting text
        return streamer, text # Return streamer and the original user text for history

    def update_history(self, user_text, assistant_response):
        """Update message history safely."""
        # Add user message (text only)
        self.message_history.append({"role": "user", "content": user_text})
        # Add assistant response
        self.message_history.append({"role": "assistant", "content": assistant_response})

        # Trim history
        if len(self.message_history) > MAX_HISTORY_MESSAGES * 2: # *2 because each turn has user+assistant
            self.message_history = self.message_history[-(MAX_HISTORY_MESSAGES * 2):]
        logger.info(f"History updated. Current length: {len(self.message_history)} messages.")


@singleton
class KokoroTTSProcessor:
    """Handles text-to-speech conversion using Kokoro."""
    def __init__(self):
        logger.info("Initializing Kokoro TTS processor...")
        try:
            # Initialize Kokoro TTS pipeline (assuming 'a' is still the desired lang code)
            self.pipeline = KPipeline(lang_code='a')
            self.default_voice = 'af_sarah' # Confirm this voice exists for lang 'a'
            logger.info(f"Kokoro TTS processor initialized with voice '{self.default_voice}'.")
            self.synthesis_count = 0
        except Exception as e:
            logger.critical(f"FATAL: Failed to initialize Kokoro TTS: {e}", exc_info=True)
            # Don't exit, maybe fallback to no TTS? Or handle differently.
            # For now, allow server to run but log critical error.
            self.pipeline = None

    async def synthesize_speech_stream(self, text_streamer):
        """Synthesize speech chunk by chunk as text arrives from a streamer."""
        if not self.pipeline:
            logger.error("Kokoro TTS pipeline not available.")
            yield None # Indicate error or end
            return

        if not text_streamer:
             logger.warning("No text streamer provided for TTS.")
             yield None
             return

        logger.info("Starting streaming TTS synthesis...")
        sentence_buffer = ""
        sentence_end_pattern = re.compile(r'[.!?。！？]+') # Split on sentence endings

        try:
            async for text_chunk in text_streamer:
                if not text_chunk: continue # Skip empty chunks

                sentence_buffer += text_chunk

                # Check if we have complete sentences to synthesize
                while True:
                    match = sentence_end_pattern.search(sentence_buffer)
                    if match:
                        sentence_end_index = match.end()
                        sentence = sentence_buffer[:sentence_end_index].strip()
                        sentence_buffer = sentence_buffer[sentence_end_index:].lstrip() # Keep remainder

                        if sentence:
                            logger.debug(f"Synthesizing sentence: '{sentence}'")
                            start_time = time.time()
                            try:
                                # Use run_in_executor for the blocking Kokoro call
                                audio_generator = await asyncio.get_event_loop().run_in_executor(
                                    None,
                                    lambda: self.pipeline(
                                        sentence,
                                        voice=self.default_voice,
                                        speed=1.0 # Adjust speed if needed
                                        # No explicit split pattern needed here as we feed sentence by sentence
                                    )
                                )
                                # Combine potentially multiple small chunks from Kokoro for one sentence
                                sentence_audio_parts = []
                                for _, _, audio_part in audio_generator:
                                    sentence_audio_parts.append(audio_part)

                                if sentence_audio_parts:
                                    combined_audio = np.concatenate(sentence_audio_parts)
                                    duration = time.time() - start_time
                                    self.synthesis_count += 1
                                    logger.info(f"TTS Synthesis #{self.synthesis_count} (sentence) took {duration:.2f}s, yielding {len(combined_audio)} samples.")
                                    yield combined_audio # Yield the synthesized audio chunk
                                else:
                                    logger.warning(f"No audio generated for sentence: '{sentence}'")

                            except Exception as e:
                                logger.error(f"Error during sentence TTS synthesis for '{sentence}': {e}", exc_info=True)
                                # Continue to next sentence/chunk

                        if not match: # No more sentence ends found in current buffer
                            break
                    else: # No sentence end found in the buffer yet
                        break

            # Synthesize any remaining text in the buffer after the stream ends
            if sentence_buffer.strip():
                logger.debug(f"Synthesizing remaining text: '{sentence_buffer}'")
                start_time = time.time()
                try:
                    audio_generator = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.pipeline(
                            sentence_buffer.strip(),
                            voice=self.default_voice,
                            speed=1.0
                        )
                    )
                    sentence_audio_parts = []
                    for _, _, audio_part in audio_generator:
                       sentence_audio_parts.append(audio_part)

                    if sentence_audio_parts:
                        combined_audio = np.concatenate(sentence_audio_parts)
                        duration = time.time() - start_time
                        self.synthesis_count += 1
                        logger.info(f"TTS Synthesis #{self.synthesis_count} (remainder) took {duration:.2f}s, yielding {len(combined_audio)} samples.")
                        yield combined_audio
                    else:
                         logger.warning(f"No audio generated for remaining text: '{sentence_buffer}'")

                except Exception as e:
                    logger.error(f"Error during final TTS synthesis: {e}", exc_info=True)

            logger.info("Streaming TTS synthesis finished.")

        except asyncio.CancelledError:
            logger.info("TTS synthesis stream cancelled.")
            raise # Re-raise cancellation
        except Exception as e:
            logger.error(f"Error consuming text stream for TTS: {e}", exc_info=True)
        finally:
            # Ensure a final yield or signal? Maybe not necessary if loop completes/breaks.
             pass


# --- WebSocket Handler ---

# Wrap the text streamer consumption to handle cancellation and collect text
async def consume_text_streamer(streamer):
    full_text = ""
    async for chunk in streamer:
        # logger.debug(f"Text chunk received: '{chunk}'")
        full_text += chunk
        # Check for cancellation frequently
        await asyncio.sleep(0.001) # Yield control briefly
    return full_text

# Wrap the audio streamer consumption to send audio and handle cancellation
async def consume_audio_stream(websocket, audio_streamer, detector):
    try:
        async for audio_chunk in audio_streamer:
            if audio_chunk is not None and len(audio_chunk) > 0:
                # Check for interrupt request BEFORE sending audio
                if await detector.is_interrupt_requested():
                    logger.warning("Interrupt requested during audio streaming. Stopping TTS playback.")
                    await detector.clear_interrupt_request() # Clear the flag
                    raise asyncio.CancelledError("TTS interrupted by new speech")

                audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                await websocket.send(json.dumps({"audio": base64_audio}))
                # logger.debug(f"Sent audio chunk: {len(audio_bytes)} bytes")
            else:
                logger.debug("Received None or empty audio chunk from TTS.")
            # Small sleep to yield control and allow other tasks (like interrupt checks)
            await asyncio.sleep(0.01)
        logger.info("Finished sending all TTS audio chunks.")
    except asyncio.CancelledError:
         logger.info("Audio streaming task cancelled.")
         raise # Re-raise to signal cancellation upwards
    except websockets.exceptions.ConnectionClosed:
        logger.warning("WebSocket closed during audio streaming.")
        raise # Re-raise to signal closure
    except Exception as e:
        logger.error(f"Error during audio streaming: {e}", exc_info=True)
        raise # Re-raise other errors

async def process_interaction(websocket, transcription, detector, gemma_processor, tts_processor):
    """Handles the full interaction cycle: Generation -> TTS -> Sending Audio"""
    logger.info(f"Starting processing for transcription: '{transcription}'")
    text_streamer = None
    full_generated_text = ""
    audio_streamer = None

    try:
        # 1. Start Gemma Generation (returns streamer immediately)
        gemma_streamer, user_prompt_for_history = await gemma_processor.generate_streaming(transcription)

        # Need to consume the gemma_streamer to get text for TTS and history
        # We'll create two consumers: one for TTS, one to collect full text

        # Tee the generator: WARNING - standard tee doesn't work well with async generators easily.
        # Instead, we'll consume it once and pass the collected text if needed,
        # or ideally, the TTS process consumes the stream directly.
        # Let's try direct consumption by TTS first.

        # Use asyncio.Queue to bridge the Gemma streamer to the TTS streamer consumer
        text_queue = asyncio.Queue()

        async def gemma_consumer_task():
            """Consumes Gemma streamer and puts chunks into a queue."""
            collected_text = ""
            try:
                async for chunk in gemma_streamer:
                    await text_queue.put(chunk)
                    collected_text += chunk
                await text_queue.put(None) # Signal end of stream
                return collected_text
            except asyncio.CancelledError:
                logger.info("Gemma consumer task cancelled.")
                await text_queue.put(None) # Ensure queue gets termination signal
                raise
            except Exception as e:
                logger.error(f"Error in Gemma consumer task: {e}", exc_info=True)
                await text_queue.put(None) # Ensure queue gets termination signal on error
                return collected_text # Return what was collected

        async def queue_to_async_generator(q):
            """Turns the queue back into an async generator for TTS."""
            while True:
                item = await q.get()
                if item is None:
                    break
                yield item
                q.task_done() # Mark item as processed

        # Start the Gemma consumer
        gemma_consumer = asyncio.create_task(gemma_consumer_task())

        # Create the async generator view of the queue for TTS
        tts_text_input_stream = queue_to_async_generator(text_queue)

        # 2. Start TTS Synthesis (consumes text stream, yields audio stream)
        audio_streamer = tts_processor.synthesize_speech_stream(tts_text_input_stream)

        # 3. Start Sending Audio (consumes audio stream, sends to websocket)
        audio_sending_task = asyncio.create_task(consume_audio_stream(websocket, audio_streamer, detector))

        # 4. Wait for both Gemma text collection and Audio sending to complete
        #    We need the full text from Gemma for history.
        #    We need the audio sending to finish or be cancelled.
        done, pending = await asyncio.wait(
            [gemma_consumer, audio_sending_task],
            return_when=asyncio.ALL_COMPLETED # Wait for both
        )

        # Check results and handle potential errors/cancellations
        for task in done:
            if task == gemma_consumer:
                try:
                    full_generated_text = task.result() # Get collected text
                    logger.info(f"Gemma generation completed. Full text length: {len(full_generated_text)}")
                except asyncio.CancelledError:
                    logger.info("Gemma text collection was cancelled.")
                    # Get partially collected text if needed? Might be unreliable.
                    # full_generated_text = await get_partial_result(gemma_consumer) # Needs helper
                except Exception as e:
                    logger.error(f"Gemma consumer task failed: {e}")
            elif task == audio_sending_task:
                try:
                    task.result() # Check for exceptions
                    logger.info("Audio sending task completed successfully.")
                except asyncio.CancelledError:
                    logger.info("Audio sending task was cancelled (likely due to interrupt).")
                    # Cancellation is expected on interrupt, don't treat as error
                except Exception as e:
                     logger.error(f"Audio sending task failed: {e}")

        # Ensure pending tasks are cancelled if one failed/was cancelled (shouldn't happen with ALL_COMPLETED)
        for task in pending:
            logger.warning("Cancelling pending task after wait.")
            task.cancel()

        # 5. Update History (only if generation completed somewhat successfully)
        #    Even if TTS was cancelled, if we got text, update history.
        if full_generated_text:
            gemma_processor.update_history(user_prompt_for_history, full_generated_text)
        else:
            logger.warning("No generated text collected, history not updated.")


    except asyncio.CancelledError:
        logger.info("Interaction processing task was cancelled (likely by interrupt).")
        # Ensure any sub-tasks created here are cancelled (though wait should handle it)
        if gemma_consumer and not gemma_consumer.done(): gemma_consumer.cancel()
        if audio_sending_task and not audio_sending_task.done(): audio_sending_task.cancel()
        # History is not updated if cancelled before completion
    except websockets.exceptions.ConnectionClosed:
         logger.warning("WebSocket closed during interaction processing.")
         # Propagate closure
         raise
    except Exception as e:
        logger.error(f"Error during interaction processing: {e}", exc_info=True)
    finally:
        # Ensure state is reset regardless of how the processing ended
        logger.info("Interaction processing finished or cancelled. Resetting state.")
        await detector.set_tts_playing(False)
        await detector.clear_interrupt_request()


async def handle_client(websocket):
    """Main WebSocket client handler."""
    client_id = websocket.remote_address
    logger.info(f"Client {client_id} connected.")

    # Get singleton instances
    try:
        transcriber = WhisperTranscriber.get_instance()
        gemma_processor = GemmaMultimodalProcessor.get_instance()
        tts_processor = KokoroTTSProcessor.get_instance()
        if not tts_processor.pipeline:
            logger.warning("TTS processor unavailable. Responses will be text-only.")
            # Optionally send a message to client?
    except Exception as e:
         logger.error(f"Failed to initialize models for client {client_id}: {e}", exc_info=True)
         await websocket.close(code=1011, reason="Internal server error during model loading")
         return

    detector = AudioSegmentDetector() # Each client gets its own detector state

    async def receive_loop():
        """Handles incoming messages (audio, images)."""
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)

                # Handle audio data
                if "realtime_input" in data:
                    for chunk in data["realtime_input"]["media_chunks"]:
                        if chunk["mime_type"] == "audio/pcm":
                            audio_data = base64.b64decode(chunk["data"])
                            # Add audio - this might trigger an interrupt request internally
                            await detector.add_audio(audio_data)
                        elif chunk["mime_type"] == "image/jpeg":
                            # Only process image if not actively playing TTS (to avoid interrupting flow)
                            if not await detector.tts_playing:
                                image_data = base64.b64decode(chunk["data"])
                                await gemma_processor.set_image(image_data)
                            else:
                                logger.debug("Ignoring image received while TTS is playing.")

                # Handle standalone image
                elif "image" in data:
                     if not await detector.tts_playing:
                        image_data = base64.b64decode(data["image"])
                        await gemma_processor.set_image(image_data)
                     else:
                        logger.debug("Ignoring standalone image received while TTS is playing.")

                # Handle explicit stop/interrupt from client? (Optional)
                elif "command" in data and data["command"] == "interrupt":
                    logger.info("Client requested interrupt.")
                    await detector.request_interrupt()
                    await detector.cancel_current_processing() # Force cancel on client command

            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Receive loop: Connection closed for {client_id}.")
                break # Exit loop
            except json.JSONDecodeError:
                logger.warning("Received invalid JSON.")
            except Exception as e:
                logger.error(f"Error in receive loop: {e}", exc_info=True)
                # Decide if we should break or continue
                # break # Safer to break on unexpected errors

    async def segment_processing_loop():
        """Processes detected speech segments."""
        while True:
            try:
                 # Check for interrupt FIRST - if requested, cancel ongoing task
                if await detector.is_interrupt_requested():
                    logger.warning("Interrupt detected in processing loop. Cancelling current task.")
                    await detector.cancel_current_processing()
                    # Don't process new segment yet, let cancellation finish
                    await asyncio.sleep(0.1) # Small delay before checking queue again
                    continue # Re-check interrupt flag

                # Only process new segment if not currently busy
                if not await detector.tts_playing:
                    segment = await detector.get_next_segment()
                    if segment:
                        # Start transcription (non-blocking conceptually)
                        transcription_text = await transcriber.transcribe(segment)
                        transcription_text = transcription_text.strip()

                        # --- Filter Transcription ---
                        is_valid_transcription = False
                        if transcription_text:
                            # Check for alphanumeric characters
                            if any(c.isalnum() for c in transcription_text):
                                # Filter common fillers / short utterances
                                words = [w for w in transcription_text.split() if any(c.isalnum() for c in w)]
                                if len(words) > 1: # Require more than one word
                                    filler_patterns = [
                                        r'^(um+|uh+|ah+|oh+|hm+|mhm+|hmm+)$',
                                        r'^(okay|yes|no|yeah|nah|hi|hello|hey)$', # Added greetings
                                        r'^bye+$',
                                        r'^(thank you|thanks)$' # Added thanks
                                    ]
                                    if not any(re.fullmatch(pattern, transcription_text.lower()) for pattern in filler_patterns):
                                        is_valid_transcription = True
                                    else:
                                        logger.info(f"Skipping filler/short transcription: '{transcription_text}'")
                                else:
                                    logger.info(f"Skipping single-word transcription: '{transcription_text}'")
                            else:
                                logger.info(f"Skipping non-alphanumeric transcription: '{transcription_text}'")
                        else:
                             logger.info("Skipping empty transcription.")


                        if is_valid_transcription:
                            # --- Start Interaction ---
                            # Create the main processing task for this interaction
                            interaction_task = asyncio.create_task(
                                process_interaction(websocket, transcription_text, detector, gemma_processor, tts_processor)
                            )
                            await detector.set_current_processing_task(interaction_task)
                            # Loop will continue, next iteration will check tts_playing status
                        else:
                            # If transcription invalid, ensure we are not marked as playing
                             await detector.set_tts_playing(False)

                # Short sleep to prevent busy-waiting
                await asyncio.sleep(0.05)

            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Segment processing loop: Connection closed for {client_id}.")
                break # Exit loop
            except asyncio.CancelledError:
                 logger.info(f"Segment processing loop for {client_id} cancelled.")
                 break # Exit loop
            except Exception as e:
                logger.error(f"Error in segment processing loop: {e}", exc_info=True)
                # Attempt to reset state on error
                await detector.cancel_current_processing()
                await asyncio.sleep(1) # Pause before retrying


    # Run receiver and processor loops concurrently
    receiver_task = asyncio.create_task(receive_loop())
    processor_task = asyncio.create_task(segment_processing_loop())

    try:
        done, pending = await asyncio.wait(
            [receiver_task, processor_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            logger.info(f"Cancelling pending task for client {client_id}.")
            task.cancel()
            try:
                # Wait briefly for cancellation to be processed
                await asyncio.wait_for(task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass # Expected
            except Exception as e:
                logger.error(f"Error during task cleanup: {e}")


    except Exception as e:
         logger.error(f"Error in main client handler for {client_id}: {e}", exc_info=True)
    finally:
        # Final cleanup for the client
        logger.info(f"Client {client_id} disconnecting. Cleaning up.")
        await detector.cancel_current_processing() # Ensure any lingering task is cancelled
        # Close WebSocket connection if not already closed
        if not websocket.closed:
            await websocket.close()
        logger.info(f"Client {client_id} cleanup complete.")


async def main():
    """Initializes models and starts the WebSocket server."""
    # Initialize models eagerly at startup
    logger.info("Initializing models...")
    try:
        WhisperTranscriber.get_instance()
        GemmaMultimodalProcessor.get_instance()
        KokoroTTSProcessor.get_instance()
        logger.info("All models initialized successfully.")
    except Exception as e:
        logger.critical(f"FATAL: Could not initialize models on startup: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Starting WebSocket server on 0.0.0.0:{WEBSOCKET_PORT}")
    try:
        async with websockets.serve(
            handle_client,
            "0.0.0.0",
            WEBSOCKET_PORT,
            ping_interval=PING_INTERVAL,
            ping_timeout=PING_TIMEOUT,
            close_timeout=10
        ):
            logger.info("WebSocket server running.")
            await asyncio.Future()  # Run forever
    except OSError as e:
        logger.critical(f"FATAL: Could not start server, port {WEBSOCKET_PORT} likely in use: {e}")
    except Exception as e:
        logger.critical(f"FATAL: Server failed to start: {e}", exc_info=True)

if __name__ == "__main__":
    # Add PYTHONASYNCIODEBUG=1 env var for more detailed asyncio logs if needed
    # Example: PYTHONASYNCIODEBUG=1 python your_script.py
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception in main asyncio run: {e}", exc_info=True)
