import asyncio
import json
import websockets
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoProcessor, Gemma3ForConditionalGeneration
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
from kokoro import KPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class AudioSegmentDetector:
    """Detects speech segments based on audio energy levels"""

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
        self.lock = asyncio.Lock()
        self.segment_queue = asyncio.Queue()

        # Counters
        self.segments_detected = 0

        # TTS playback and generation control
        self.tts_playing = False
        self.tts_lock = asyncio.Lock()
        self.current_generation_task = None
        self.current_tts_task = None
        self.task_lock = asyncio.Lock()

    async def set_tts_playing(self, is_playing):
        """Set TTS playback state"""
        async with self.tts_lock:
            self.tts_playing = is_playing

    async def cancel_current_tasks(self, reason="New segment detected"):
        """Cancel any ongoing generation and TTS tasks"""
        async with self.task_lock:
            if self.current_generation_task:
                if not self.current_generation_task.done():
                    logger.info(f"Cancelling generation task: {reason}")
                    self.current_generation_task.cancel(reason)
                    try:
                        await self.current_generation_task
                    except asyncio.CancelledError:
                        logger.info("Generation task successfully cancelled.")
                    except Exception as e:
                        logger.error(f"Error awaiting generation task cancellation: {e}")
                self.current_generation_task = None

            if self.current_tts_task:
                if not self.current_tts_task.done():
                    logger.info(f"Cancelling TTS task: {reason}")
                    self.current_tts_task.cancel(reason)
                    try:
                        await self.current_tts_task
                    except asyncio.CancelledError:
                        logger.info("TTS task successfully cancelled.")
                    except Exception as e:
                        logger.error(f"Error awaiting TTS task cancellation: {e}")
                self.current_tts_task = None

            # Clear TTS playing state
            await self.set_tts_playing(False)

    async def set_current_tasks(self, generation_task=None, tts_task=None):
        """Set current generation and TTS tasks"""
        async with self.task_lock:
            self.current_generation_task = generation_task
            self.current_tts_task = tts_task

    async def add_audio(self, audio_bytes):
        """Add audio data to the buffer and check for speech segments"""
        async with self.lock:
            # Add new audio to buffer regardless of TTS state
            self.audio_buffer.extend(audio_bytes)

            # Convert recent audio to numpy for energy analysis
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Calculate audio energy (root mean square)
            if len(audio_array) > 0:
                energy = np.sqrt(np.mean(audio_array**2))

                # Speech detection logic
                if not self.is_speech_active and energy > self.energy_threshold:
                    # Speech start detected
                    self.is_speech_active = True
                    self.speech_start_idx = max(0, len(self.audio_buffer) - len(audio_bytes))
                    self.silence_counter = 0
                    logger.info(f"Speech start detected (energy: {energy:.6f})")

                elif self.is_speech_active:
                    if energy > self.energy_threshold:
                        # Continued speech
                        self.silence_counter = 0
                    else:
                        # Potential end of speech
                        self.silence_counter += len(audio_array)

                        # Check if enough silence to end speech segment
                        if self.silence_counter >= self.silence_samples:
                            speech_end_idx = len(self.audio_buffer) - self.silence_counter
                            speech_segment = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])

                            # Reset for next speech detection
                            self.is_speech_active = False
                            self.silence_counter = 0

                            # Trim buffer to keep only recent audio
                            self.audio_buffer = self.audio_buffer[speech_end_idx:]

                            # Only process if speech segment is long enough
                            if len(speech_segment) >= self.min_speech_samples * 2:  # × 2 for 16-bit
                                self.segments_detected += 1
                                logger.info(f"Speech segment detected: {len(speech_segment) / 2 / self.sample_rate:.2f}s")

                                # If TTS is playing or generation is ongoing, cancel them
                                async with self.tts_lock:
                                    if self.tts_playing:
                                        await self.cancel_current_tasks()

                                # Add to queue
                                await self.segment_queue.put(speech_segment)
                                return speech_segment

                        # Check if speech segment exceeds maximum duration
                        elif (len(self.audio_buffer) - self.speech_start_idx) > self.max_speech_samples * 2:
                            speech_segment = bytes(self.audio_buffer[self.speech_start_idx:
                                                                     self.speech_start_idx + self.max_speech_samples * 2])
                            # Update start index for next segment
                            self.speech_start_idx += self.max_speech_samples * 2
                            self.segments_detected += 1
                            logger.info(f"Max duration speech segment: {len(speech_segment) / 2 / self.sample_rate:.2f}s")

                            # If TTS is playing or generation is ongoing, cancel them
                            async with self.tts_lock:
                                if self.tts_playing:
                                    await self.cancel_current_tasks()

                            # Add to queue
                            await self.segment_queue.put(speech_segment)
                            return speech_segment

            return None

    async def get_next_segment(self):
        """Get the next available speech segment"""
        try:
            return await asyncio.wait_for(self.segment_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None


class WhisperTranscriber:
    """Handles speech transcription using Whisper large-v3 model with pipeline"""
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Use GPU for transcription
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Set torch dtype based on device
        self.torch_dtype = torch.float16 if self.device != "cpu" else torch.float32

        # Load model and processor
        model_id = "openai/whisper-large-v3-turbo"
        logger.info(f"Loading {model_id}...")

        # Load model
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.to(self.device)

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Create pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        logger.info("Whisper model ready for transcription")

        # Counter
        self.transcription_count = 0

    async def transcribe(self, audio_bytes, sample_rate=16000):
        """Transcribe audio bytes to text using the pipeline"""
        try:
            # Convert PCM bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Check for valid audio
            if len(audio_array) < 1000:  # Too short
                return ""

            # Use the pipeline to transcribe
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipe(
                    {"array": audio_array, "sampling_rate": sample_rate},
                    generate_kwargs={
                        "task": "transcribe",
                        "language": "english",
                        "temperature": 0.0
                    }
                )
            )

            # Extract the text from the result
            text = result.get("text", "").strip()

            self.transcription_count += 1
            logger.info(f"Transcription result: '{text}'")

            return text

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""


class GemmaMultimodalProcessor:
    """Handles multimodal generation using Gemma 3 model"""
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Use GPU for generation
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device for Gemma: {self.device}")

        # Load model and processor
        model_id = "google/gemma-3-4b-it"
        logger.info(f"Loading {model_id}...")

        # Load model with 8-bit quantization for memory efficiency
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            load_in_8bit=True,  # Enable 8-bit quantization
            torch_dtype=torch.bfloat16
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id)

        logger.info("Gemma model ready for multimodal generation")

        # Cache for most recent image
        self.last_image = None
        self.last_image_timestamp = 0
        self.lock = asyncio.Lock()

        # Message history management
        self.message_history = []
        self.max_history_messages = 4  # Keep last 4 exchanges (2 user, 2 assistant)

        # Counter
        self.generation_count = 0

    async def set_image(self, image_data):
        """Cache the most recent image received"""
        async with self.lock:
            try:
                # Convert image data to PIL Image
                image = Image.open(io.BytesIO(image_data))

                # Resize to 75% of original size
                new_size = (int(image.size[0] * 0.75), int(image.size[1] * 0.75))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

                # Clear message history when new image is set
                self.message_history = []
                self.last_image = image
                self.last_image_timestamp = time.time()
                return True
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                return False

    def _build_messages(self, text):
        """Build messages array with history for the model"""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": """You are a helpful assistant providing spoken
            responses about images and engaging in natural conversation. Keep your responses concise,
            fluent, and conversational. Use natural oral language that's easy to listen to.

            When responding:
            1. If the user's question or comment is clearly about the image, provide a relevant,
               focused response about what you see.
            2. If the user's input is not clearly related to the image or lacks context:
               - Don't force image descriptions into your response
               - Respond naturally as in a normal conversation
               - If needed, politely ask for clarification (e.g., "Could you please be more specific
                 about what you'd like to know about the image?")
            3. Keep responses concise:
               - Aim for 2-3 short sentences
               - Focus on the most relevant information
               - Use conversational language

            Maintain conversation context and refer to previous exchanges naturally when relevant.
            If the user's request is unclear, ask them to repeat or clarify in a friendly way."""}]
            }
        ]

        # Add conversation history
        messages.extend(self.message_history)

        # Add current user message with image
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": self.last_image},
                {"type": "text", "text": text}
            ]
        })

        return messages

    def _update_history(self, user_text, assistant_response):
        """Update message history with new exchange"""
        # Add user message
        self.message_history.append({
            "role": "user",
            "content": [{"type": "text", "text": user_text}]
        })

        # Add assistant response
        self.message_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_response}]
        })

        # Trim history to keep only recent messages
        if len(self.message_history) > self.max_history_messages:
            self.message_history = self.message_history[-self.max_history_messages:]

        logger.info(f"Updated message history with complete response ({len(assistant_response)} chars)")

    async def generate_streaming(self, text, initial_chunks=3):
        """Generate a response using the latest image and text input with streaming for initial chunks"""
        async with self.lock:
            try:
                if not self.last_image:
                    logger.warning("No image available for multimodal generation")
                    return None, f"No image context: {text}"

                # Build messages with history
                messages = self._build_messages(text)

                # Prepare inputs for the model
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.model.device)

                input_len = inputs["input_ids"].shape[-1]

                # Create a streamer for token-by-token generation
                from transformers import TextIteratorStreamer

                streamer = TextIteratorStreamer(
                    self.processor.tokenizer,
                    skip_special_tokens=True,
                    skip_prompt=True
                )

                # Start generation in a separate thread
                generation_kwargs = dict(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    use_cache=True,
                    streamer=streamer,
                )

                def run_generation():
                    try:
                        self.model.generate(**generation_kwargs)
                    except Exception as e:
                        logger.error(f"Generation thread error: {e}")

                thread = Thread(target=run_generation)
                thread.start()

                # Collect initial text until we have a complete sentence or enough content
                initial_text = ""
                min_chars = 50  # Minimum characters to collect for initial chunk
                sentence_end_pattern = re.compile(r'[.!?]')
                has_sentence_end = False

                # Collect the first sentence or minimum character count
                try:
                    for chunk in streamer:
                        # Check for cancellation
                        if asyncio.current_task().cancelled():
                            logger.info("Gemma generation cancelled during streaming.")
                            thread.join(timeout=1)  # Ensure the thread terminates if possible
                            return None, None  # Signal cancellation

                        initial_text += chunk

                        # Check if we have a sentence end
                        if sentence_end_pattern.search(chunk):
                            has_sentence_end = True
                            # If we have at least some content, break after sentence end
                            if len(initial_text) >= min_chars / 2:
                                break

                        # If we have enough content, break
                        if len(initial_text) >= min_chars and (has_sentence_end or "," in initial_text):
                            break

                        # Safety check - if we've collected a lot of text without sentence end
                        if len(initial_text) >= min_chars * 2:
                            break
                except asyncio.CancelledError:
                    logger.info("Gemma streaming cancelled.")
                    thread.join(timeout=1)
                    return None, None


                # Return initial text and the streamer for continued generation
                self.generation_count += 1
                logger.info(f"Gemma initial generation: '{initial_text}' ({len(initial_text)} chars)")
                return streamer, initial_text

            except Exception as e:
                logger.error(f"Gemma streaming generation error: {e}")
                return None, f"Error processing: {text}"

    async def generate(self, text):
        """Generate a response using the latest image and text input (non-streaming)"""
        async with self.lock:
            try:
                if not self.last_image:
                    logger.warning("No image available for multimodal generation")
                    return f"No image context: {text}"

                # Build messages with history
                messages = self._build_messages(text)

                # Prepare inputs for the model
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.model.device)

                input_len = inputs["input_ids"].shape[-1]

                # Generate response with parameters tuned for concise output
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    use_cache=True,
                )

                # Decode the generated tokens
                generated_text = self.processor.decode(
                    generation[0][input_len:],
                    skip_special_tokens=True
                )

                # Update conversation history
                self._update_history(text, generated_text)

                self.generation_count += 1
                logger.info(f"Gemma generation result ({len(generated_text)} chars)")

                return generated_text

            except Exception as e:
                logger.error(f"Gemma generation error: {e}")
                return f"Error processing: {text}"


class KokoroTTSProcessor:
    """Handles text-to-speech conversion using Kokoro model"""
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        logger.info("Initializing Kokoro TTS processor...")
        try:
            # Initialize Kokoro TTS pipeline with Chinese
            self.pipeline = KPipeline(lang_code='a')

            # Set Chinese voice to xiaobei
            self.default_voice = 'af_sarah'

            logger.info("Kokoro TTS processor initialized successfully")
            # Counter
            self.synthesis_count = 0
        except Exception as e:
            logger.error(f"Error initializing Kokoro TTS: {e}")
            self.pipeline = None

    async def synthesize_initial_speech(self, text):
        """Convert initial text to speech using Kokoro TTS with minimal splitting for speed"""
        if not text or not self.pipeline:
            return None

        try:
            logger.info(f"Synthesizing initial speech for text: '{text}'")

            # Run TTS in a thread pool to avoid blocking
            audio_segments = []

            # Use the executor to run the TTS pipeline with minimal splitting
            # For initial text, we want to process it quickly with minimal splits
            async def run_tts():
                return self.pipeline(
                    text,
                    voice=self.default_voice,
                    speed=1,
                    split_pattern=None  # No splitting for initial text to process faster
                )

            try:
                generator = await asyncio.get_event_loop().run_in_executor(
                    None,
                    run_tts
                )

                # Process all generated segments
                for gs, ps, audio in generator:
                    # Check for cancellation after processing each segment
                    if asyncio.current_task().cancelled():
                        logger.info("TTS task cancelled during segment processing.")
                        return None  # Return None to signal cancellation

                    audio_segments.append(audio)

                # Combine all audio segments
                if audio_segments:
                    combined_audio = np.concatenate(audio_segments)
                    self.synthesis_count += 1
                    logger.info(f"Initial speech synthesis complete: {len(combined_audio)} samples")
                    return combined_audio
                return None
            except asyncio.CancelledError:
                logger.info("TTS task was cancelled.")
                return None

        except Exception as e:
            logger.error(f"Initial speech synthesis error: {e}")
            return None

    async def synthesize_remaining_speech(self, text):
        """Convert remaining text to speech using Kokoro TTS with comprehensive splitting for quality"""
        if not text or not self.pipeline:
            return None

        try:
            logger.info(f"Synthesizing remaining speech for text: '{text[:50]}...' if len(text) > 50 else text")

            # Run TTS in a thread pool to avoid blocking
            audio_segments = []

            # Use the executor to run the TTS pipeline with comprehensive splitting
            # For remaining text, we want to process it with proper splits for better quality
            async def run_tts():
                return self.pipeline(
                    text,
                    voice=self.default_voice,
                    speed=1,
                    split_pattern=r'[.!?。！？,，;；:]+'  # Comprehensive splitting for remaining text
                )

            try:
                generator = await asyncio.get_event_loop().run_in_executor(
                    None,
                    run_tts
                )

                # Process all generated segments
                for gs, ps, audio in generator:
                    # Check for cancellation after processing each segment
                    if asyncio.current_task().cancelled():
                        logger.info("TTS task cancelled during segment processing.")
                        return None # Return None to signal cancellation

                    audio_segments.append(audio)

                # Combine all audio segments
                if audio_segments:
                    combined_audio = np.concatenate(audio_segments)
                    self.synthesis_count += 1
                    logger.info(f"Remaining speech synthesis complete: {len(combined_audio)} samples")
                    return combined_audio
                return None
            except asyncio.CancelledError:
                logger.info("TTS task was cancelled.")
                return None

        except Exception as e:
            logger.error(f"Remaining speech synthesis error: {e}")
            return None

    async def synthesize_speech(self, text):
        """Convert text to speech using Kokoro TTS (legacy method)"""
        if not text or not self.pipeline:
            return None

        try:
            logger.info(f"Synthesizing speech for text: '{text[:50]}...' if len(text) > 50 else text")

            # Run TTS in a thread pool to avoid blocking
            audio_segments = []

            # Use the executor to run the TTS pipeline
            # Updated split pattern to include Chinese punctuation marks
            async def run_tts():
                return self.pipeline(
                    text,
                    voice=self.default_voice,
                    speed=1,
                    split_pattern=r'[.!?。！？]+'  # Added Chinese punctuation marks
                )

            try:
                generator = await asyncio.get_event_loop().run_in_executor(
                    None,
                    run_tts
                )

                # Process all generated segments
                for gs, ps, audio in generator:
                    # Check for cancellation after processing each segment
                    if asyncio.current_task().cancelled():
                        logger.info("TTS task cancelled during segment processing.")
                        return None # Return None to signal cancellation

                    audio_segments.append(audio)

                # Combine all audio segments
                if audio_segments:
                    combined_audio = np.concatenate(audio_segments)
                    self.synthesis_count += 1
                    logger.info(f"Speech synthesis complete: {len(combined_audio)} samples")
                    return combined_audio
                return None
            except asyncio.CancelledError:
                logger.info("TTS task was cancelled.")
                return None

        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
            return None


async def handle_client(websocket):
    """Handles WebSocket client connection"""
    detector = AudioSegmentDetector()
    transcriber = WhisperTranscriber.get_instance()
    gemma_processor = GemmaMultimodalProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance()

    async def generate_response(transcription):
        try:
            streamer, initial_text = await gemma_processor.generate_streaming(transcription, initial_chunks=3)
            return streamer, initial_text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None, f"Error: {e}"

    async def synthesize_speech(text):
        try:
            if text:
                return await tts_processor.synthesize_initial_speech(text)
            return None
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return None

    async def collect_remaining_text(streamer):
        collected_text = ""
        try:
            async for chunk in streamer:
                # Check for cancellation
                if asyncio.current_task().cancelled():
                    logger.info("Remaining text collection cancelled.")
                    return None
                collected_text += chunk
        except asyncio.CancelledError:
            logger.info("Task cancelled - collection interrupted.")
            return None
        return collected_text

    async def synthesize_remaining(remaining_text):
        try:
            if remaining_text:
                return await tts_processor.synthesize_remaining_speech(remaining_text)
            return None
        except Exception as e:
            logger.error(f"Error synthesizing remaining speech: {e}")
            return None

    async def process_speech_segment(speech_segment):
        try:
            transcription = await transcriber.transcribe(speech_segment)

            if not transcription or not any(c.isalnum() for c in transcription):
                logger.info(f"Skipping invalid transcription: '{transcription}'")
                return

            words = [w for w in transcription.split() if any(c.isalnum() for c in w)]
            if len(words) <= 1:
                if transcription.lower() in ["okay", "yes", "no", "yeah", "nah", "bye"]:
                    logger.info(f"Skipping single-word filler: '{transcription}'")
                    return
                else:
                    transcription = "Sorry, Can you please repeat"

            await detector.set_tts_playing(True)
            streamer, initial_text = await generate_response(transcription)

            if not streamer and not initial_text:
                return

            initial_audio = await synthesize_speech(initial_text)
            if initial_audio is not None:
                audio_bytes = (initial_audio * 32767).astype(np.int16).tobytes()
                base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                await websocket.send(json.dumps({"audio": base64_audio}))

            remaining_text = await collect_remaining_text(streamer)

            if remaining_text is None:
                logger.info("Remaining text collection cancelled, skipping TTS.")
                return

            remaining_audio = await synthesize_remaining(remaining_text)

            if remaining_audio is not None:
                audio_bytes = (remaining_audio * 32767).astype(np.int16).tobytes()
                base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                await websocket.send(json.dumps({"audio": base64_audio}))

            gemma_processor._update_history(transcription, initial_text + remaining_text)

        except asyncio.CancelledError:
            logger.info("Task cancelled - processing interrupted.")
        except Exception as e:
            logger.error(f"Error processing segment: {e}")
        finally:
            await detector.set_tts_playing(False)

    async def detect_speech_segments():
        try:
            while True:
                # Check for cancellation before retrieving next segment
                if asyncio.current_task().cancelled():
                    logger.info("Speech detection cancelled.")
                    break
                speech_segment = await detector.get_next_segment()
                if speech_segment:
                    await detector.cancel_current_tasks()
                    await process_speech_segment(speech_segment)
                await asyncio.sleep(0.01)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed during speech detection.")
        except Exception as e:
            logger.error(f"Error in speech detection loop: {e}")
        finally:
            await detector.set_tts_playing(False)

    async def receive_audio_and_images():
        try:
            async for message in websocket:
                # Check for cancellation before processing each message
                if asyncio.current_task().cancelled():
                    logger.info("Data reception cancelled.")
                    break
                try:
                    data = json.loads(message)

                    if "realtime_input" in data:
                        for chunk in data["realtime_input"]["media_chunks"]:
                            if chunk["mime_type"] == "audio/pcm":
                                audio_data = base64.b64decode(chunk["data"])
                                await detector.add_audio(audio_data)
                            elif chunk["mime_type"] == "image/jpeg" and not detector.tts_playing:
                                image_data = base64.b64decode(chunk["data"])
                                await gemma_processor.set_image(image_data)

                    if "image" in data and not detector.tts_playing:
                        image_data = base64.b64decode(data["image"])
                        await gemma_processor.set_image(image_data)

                except Exception as e:
                    logger.error(f"Error receiving data: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed during data reception.")
        except Exception as e:
            logger.error(f"Error in data reception loop: {e}")

    async def send_keepalive():
        try:
            while True:
                # Check for cancellation before sending ping
                if asyncio.current_task().cancelled():
                    logger.info("Keepalive task cancelled.")
                    break
                await websocket.ping()
                await asyncio.sleep(20)  # Send ping every 20 seconds
        except Exception:
            logger.info("Keepalive task stopped due to connection issues.")

    try:
        await websocket.recv()
        logger.info("Client connected")

        # Create tasks and store them for potential cancellation
        receive_task = asyncio.create_task(receive_audio_and_images())
        detect_task = asyncio.create_task(detect_speech_segments())
        keepalive_task = asyncio.create_task(send_keepalive())

        await asyncio.gather(
            receive_task,
            detect_task,
            keepalive_task
        )
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"Error in handle_client: {e}")
    finally:
        logger.info("Cleaning up resources...")
        await detector.set_tts_playing(False)
        await detector.cancel_current_tasks("Client disconnected")

        # Cancel remaining tasks if they are still running
        if 'receive_task' in locals() and not receive_task.done():
            receive_task.cancel("Cleanup after client disconnect")
        if 'detect_task' in locals() and not detect_task.done():
            detect_task.cancel("Cleanup after client disconnect")
        if 'keepalive_task' in locals() and not keepalive_task.done():
            keepalive_task.cancel("Cleanup after client disconnect")


async def main():
    """Main function to start the WebSocket server"""
    try:
        # Initialize all processors ahead of time to load models
        transcriber = WhisperTranscriber.get_instance()
        gemma_processor = GemmaMultimodalProcessor.get_instance()
        tts_processor = KokoroTTSProcessor.get_instance()

        logger.info("Starting WebSocket server on 0.0.0.0:9073")
        # Add ping_interval and ping_timeout parameters
        async with websockets.serve(
            handle_client,
            "0.0.0.0",
            9073,
            ping_interval=20,    # Send ping every 20 seconds
            ping_timeout=60,     # Wait up to 60 seconds for pong response
            close_timeout=10     # Wait up to 10 seconds for close handshake
        ):
            logger.info("WebSocket server running on 0.0.0.0:9073")
            await asyncio.Future()  # Run forever
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
