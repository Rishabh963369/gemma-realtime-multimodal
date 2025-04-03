import asyncio
import json
import websockets
import base64
import torch
import re
import time
import logging
import sys
import io
from PIL import Image
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, Gemma3ForConditionalGeneration, TextIteratorStreamer
from functools import lru_cache
from kokoro import KPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class AudioSegmentDetector:
    def __init__(self, sample_rate=16000, energy_threshold=0.015, silence_duration=0.5, min_speech_duration=0.5, max_speech_duration=10):
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
        self.segment_queue = asyncio.Queue(maxsize=3)  # Increased queue size
        self.segments_detected = 0
        self.tts_playing = False
        self.tts_lock = asyncio.Lock()
        self.current_generation_task = None
        self.current_tts_task = None
        self.task_lock = asyncio.Lock()
        
        # Add VAD pre-filtering
        self.vad_buffer = []
        self.vad_threshold = 0.01
        self.vad_frame_size = int(0.03 * sample_rate)  # 30ms frames

    async def set_tts_playing(self, is_playing):
        async with self.tts_lock:
            self.tts_playing = is_playing

    async def cancel_current_tasks(self):
        async with self.task_lock:
            if self.current_generation_task and not self.current_generation_task.done():
                self.current_generation_task.cancel()
                try:
                    await self.current_generation_task
                except asyncio.CancelledError:
                    pass
            if self.current_tts_task and not self.current_tts_task.done():
                self.current_tts_task.cancel()
                try:
                    await self.current_tts_task
                except asyncio.CancelledError:
                    pass
            self.current_generation_task = None
            self.current_tts_task = None
            while not self.segment_queue.empty():
                await self.segment_queue.get()
            await self.set_tts_playing(False)

    async def set_current_tasks(self, generation_task=None, tts_task=None):
        async with self.task_lock:
            self.current_generation_task = generation_task
            self.current_tts_task = tts_task

    def _is_voice_active(self, audio_array):
        """Pre-filter using simple VAD"""
        if len(audio_array) == 0:
            return False
        energy = np.sqrt(np.mean(audio_array**2))
        return energy > self.vad_threshold

    async def add_audio(self, audio_bytes):
        async with self.lock:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Early VAD check
            if not self._is_voice_active(audio_array):
                if self.is_speech_active:
                    self.silence_counter += len(audio_array)
                return None
                
            self.audio_buffer.extend(audio_bytes)
            
            if len(audio_array) > 0:
                energy = np.sqrt(np.mean(audio_array**2))
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
                            speech_segment = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])
                            self.is_speech_active = False
                            self.silence_counter = 0
                            self.audio_buffer = self.audio_buffer[speech_end_idx:]
                            if len(speech_segment) >= self.min_speech_samples * 2:
                                self.segments_detected += 1
                                logger.info(f"Speech segment detected: {len(speech_segment) / 2 / self.sample_rate:.2f}s")
                                
                                # Don't cancel tasks immediately to allow natural flow
                                if not self.tts_playing:
                                    await self.cancel_current_tasks()
                                    
                                # Try to add to queue without blocking
                                try:
                                    self.segment_queue.put_nowait(speech_segment)
                                except asyncio.QueueFull:
                                    logger.warning("Segment queue full, dropping oldest segment")
                                    await self.segment_queue.get()
                                    await self.segment_queue.put(speech_segment)
                                return speech_segment
                        elif (len(self.audio_buffer) - self.speech_start_idx) > self.max_speech_samples * 2:
                            speech_segment = bytes(self.audio_buffer[self.speech_start_idx:self.speech_start_idx + self.max_speech_samples * 2])
                            self.speech_start_idx += self.max_speech_samples * 2
                            self.segments_detected += 1
                            logger.info(f"Max duration speech segment: {len(speech_segment) / 2 / self.sample_rate:.2f}s")
                            
                            if not self.tts_playing:
                                await self.cancel_current_tasks()
                                
                            try: 
                                self.segment_queue.put_nowait(speech_segment)
                            except asyncio.QueueFull:
                                logger.warning("Segment queue full, dropping oldest segment")
                                await self.segment_queue.get()
                                await self.segment_queue.put(speech_segment)
                            return speech_segment
        return None

    async def get_next_segment(self):
        try:
            return await asyncio.wait_for(self.segment_queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            return None

class WhisperTranscriber:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
        model_id = "openai/whisper-large-v3-turbo"
        
        # Enhanced model initialization with explicit quantization
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=self.torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        
        if self.device != "cpu":
            self.model = self.model.half()  # Explicit half-precision
            
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition", 
            model=self.model, 
            tokenizer=self.processor.tokenizer, 
            feature_extractor=self.processor.feature_extractor, 
            torch_dtype=self.torch_dtype, 
            device=self.device
        )
        
        self.transcription_count = 0
        self.language_cache = {}
        logger.info(f"Whisper model loaded on {self.device}")

    async def detect_language(self, audio_array):
        """Only detect language if we haven't seen this user/session before"""
        # This would be expanded in a multi-user system
        if 'default' not in self.language_cache:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.pipe(
                        {"raw": audio_array[:int(self.pipe.feature_extractor.sampling_rate * 3)], "sampling_rate": self.pipe.feature_extractor.sampling_rate},
                        generate_kwargs={"task": "language_identification"}
                    )
                )
                self.language_cache['default'] = result.get("language", "english")
                logger.info(f"Language detected: {self.language_cache['default']}")
            except Exception as e:
                logger.error(f"Language detection error: {e}")
                self.language_cache['default'] = "english"
        return self.language_cache.get('default', "english")

    async def transcribe(self, audio_bytes, sample_rate=16000):
        start_time = time.time()
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_array) < 500:
                logger.info("Audio too short for transcription")
                return ""
                
            # Detect language only on first few calls
            language = "english"
            if self.transcription_count < 3:
                language = await self.detect_language(audio_array)
            
            # Optimized transcription with specific params
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.pipe(
                    {"raw": audio_array, "sampling_rate": sample_rate}, 
                    generate_kwargs={
                        "task": "transcribe", 
                        "language": language,
                        "max_new_tokens": 128,  # Limit token generation
                        "return_timestamps": False  # Disable if not needed
                    }
                )
            )
            
            text = result.get("text", "").strip()
            self.transcription_count += 1
            process_time = time.time() - start_time
            logger.info(f"Transcription completed in {process_time:.2f}s: '{text}'")
            return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
        finally:
            # Optionally clear CUDA cache if many transcriptions happening
            if self.transcription_count % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

class GemmaMultimodalProcessor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_id = "google/gemma-3-4b-it"
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"  # Enabled flash attention
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.last_image = None
        self.last_image_timestamp = 0
        self.lock = asyncio.Lock()
        self.message_history = []
        self.generation_count = 0
        self.max_history_length = 4  # Keep only 2 exchanges (4 messages)
        logger.info(f"Gemma model loaded on {self.device}")

    async def set_image(self, image_data):
        async with self.lock:
            try:
                if not image_data or len(image_data) < 100:
                    logger.warning("Invalid or empty image data received")
                    return False
                    
                image = Image.open(io.BytesIO(image_data))
                
                # Standardized image processing
                target_size = (512, 512)
                resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
                
                # Clear history when new image is set
                self.message_history = []
                self.last_image = resized_image
                self.last_image_timestamp = time.time()
                logger.info(f"Image set successfully (resized to {target_size})")
                return True
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                return False

    def _build_messages(self, text):
        system_message = {
            "role": "system", 
            "content": [{"type": "text", "text": "You are a helpful assistant providing concise spoken responses about images or engaging in natural conversation. Keep responses brief and conversational."}]
        }
        messages = [system_message]
        
        # Use history with limits
        if len(self.message_history) > 0:
            # Only include most recent exchanges to prevent context growth
            messages.extend(self.message_history[-self.max_history_length:])
        
        if self.last_image:
            # Include image in current message if available
            messages.append({
                "role": "user", 
                "content": [
                    {"type": "image", "image": self.last_image}, 
                    {"type": "text", "text": text}
                ]
            })
        else:
            messages.append({
                "role": "user", 
                "content": [{"type": "text", "text": text}]
            })
        return messages

    def _update_history(self, user_text, assistant_response):
        # Add new exchange
        self.message_history.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
        self.message_history.append({"role": "assistant", "content": [{"type": "text", "text": assistant_response}]})
        
        # Limit history length
        if len(self.message_history) > self.max_history_length:
            self.message_history = self.message_history[-self.max_history_length:]

    async def generate_streaming(self, text):
        async with self.lock:
            start_time = time.time()
            try:
                messages = self._build_messages(text)
                inputs = self.processor.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    tokenize=True, 
                    return_dict=True, 
                    return_tensors="pt"
                ).to(self.model.device)
                
                # Optimized generation parameters
                streamer = TextIteratorStreamer(self.processor.tokenizer, skip_special_tokens=True, skip_prompt=True)
                generation_kwargs = dict(
                    **inputs, 
                    max_new_tokens=256, 
                    do_sample=True,
                    temperature=0.7,
                    top_k=40,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    use_cache=True, 
                    streamer=streamer
                )
                
                import threading
                threading.Thread(target=self.model.generate, kwargs=generation_kwargs).start()
                
                # Get initial text for immediate response
                initial_text = ""
                for chunk in streamer:
                    initial_text += chunk
                    if len(initial_text) > 10 or "." in chunk or "," in chunk:
                        break
                
                self.generation_count += 1
                logger.info(f"Generated initial text in {time.time() - start_time:.2f}s: '{initial_text}'")
                return streamer, initial_text
            except Exception as e:
                logger.error(f"Gemma streaming error: {e}")
                return None, f"Sorry, I couldn't process that due to an error."
            finally:
                # Clear cache periodically
                if self.generation_count % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

class KokoroTTSProcessor:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.pipeline = KPipeline(lang_code='a')
        self.default_voice = 'af_sarah'
        self.synthesis_count = 0
        self.cache = {}  # Add response caching
        self.max_cache_size = 100
        self.chunk_size_threshold = 20  # Characters per chunk
        logger.info("Kokoro TTS loaded")
        
    @lru_cache(maxsize=100)
    def _get_cached_audio(self, text_hash, voice):
        """LRU cache wrapper for TTS outputs"""
        return self.cache.get(f"{text_hash}_{voice}", None)
        
    def _add_to_cache(self, text, voice, audio):
        """Add synthesized audio to cache"""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest item when cache is full
            self.cache.pop(next(iter(self.cache)))
        
        # Use simple hash to avoid very long keys
        text_hash = str(hash(text) % 10000000)
        self.cache[f"{text_hash}_{voice}"] = audio
        
    def split_into_sentences(self, text):
        """Split text into natural sentences for parallel processing"""
        if not text:
            return []
        
        # Match sentence-ending punctuation followed by space or end of string
        sentence_pattern = r'([.!?;。！？；]+)(?:\s|$)'
        sentences = []
        
        # Split the text using regex
        parts = re.split(sentence_pattern, text)
        
        # Combine each sentence with its punctuation
        i = 0
        while i < len(parts) - 1:
            sentences.append(parts[i] + parts[i+1])
            i += 2
            
        # Add any remaining part
        if i < len(parts):
            sentences.append(parts[i])
            
        # Filter empty sentences and those that are just punctuation
        sentences = [s for s in sentences if s.strip() and not re.match(r'^[.!?;。！？；\s]+$', s)]
        
        return sentences

    async def synthesize_speech(self, text):
        if not text or not self.pipeline:
            return None
            
        start_time = time.time()
        try:
            # Check cache first using text hash
            text_hash = str(hash(text) % 10000000)
            cached_audio = self._get_cached_audio(text_hash, self.default_voice)
            if cached_audio is not None:
                logger.info(f"TTS cache hit: {len(cached_audio)} samples")
                return cached_audio
            
            # Split text into sentences for parallel processing
            sentences = self.split_into_sentences(text)
            if not sentences:
                return None
                
            # Process sentences in parallel
            audio_segments = []
            tasks = []
            
            for sentence in sentences:
                if sentence.strip():
                    tasks.append(asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda s=sentence: list(self.pipeline(s, voice=self.default_voice, speed=1.0, split_pattern=r'[.!?。！？]+'))
                    ))
            
            # Gather results
            results = await asyncio.gather(*tasks)
            for result in results:
                for _, _, audio in result:
                    audio_segments.append(audio)
                    
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                self.synthesis_count += 1
                process_time = time.time() - start_time
                logger.info(f"TTS synthesized in {process_time:.2f}s: {len(combined_audio)} samples")
                
                # Cache the result
                self._add_to_cache(text, self.default_voice, combined_audio)
                
                return combined_audio
            return None
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
            
    async def synthesize_speech_chunked(self, text, chunk_callback):
        """Synthesize speech in chunks, calling callback with each audio segment"""
        if not text:
            return
            
        sentences = self.split_into_sentences(text)
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            audio = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: next(iter([audio for _, _, audio in self.pipeline(sentence, voice=self.default_voice, speed=1.0)]), None)
            )
            
            if audio is not None:
                await chunk_callback(audio)

async def cleanup_resources():
    """Periodically clean up resources"""
    while True:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        await asyncio.sleep(300)  # Every 5 minutes

async def handle_client(websocket):
    detector = AudioSegmentDetector()
    transcriber = WhisperTranscriber.get_instance()
    gemma_processor = GemmaMultimodalProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance()
    
    # Start metrics
    start_time = time.time()
    metrics = {
        "transcriptions": 0,
        "generations": 0,
        "tts_synths": 0,
        "errors": 0
    }

    async def send_audio(audio):
        if audio is not None:
            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))

    async def process_speech_segment(speech_segment):
        segment_start = time.time()
        try:
            # Transcribe speech
            transcription = await transcriber.transcribe(speech_segment)
            if not transcription or not any(c.isalnum() for c in transcription):
                logger.info(f"Skipping empty transcription: '{transcription}'")
                return
                
            await detector.set_tts_playing(True)
            metrics["transcriptions"] += 1
            
            # Generate response
            generation_start = time.time()
            streamer, initial_text = await gemma_processor.generate_streaming(transcription)
            if not streamer or not initial_text:
                logger.error("No response generated")
                initial_audio = await tts_processor.synthesize_speech("Sorry, I couldn't generate a response.")
                await send_audio(initial_audio)
                return
                
            metrics["generations"] += 1
            logger.info(f"Generation latency: {time.time() - generation_start:.2f}s")
            
            # Send initial response audio immediately
            tts_start = time.time()
            initial_audio = await tts_processor.synthesize_speech(initial_text)
            await send_audio(initial_audio)
            metrics["tts_synths"] += 1
            
            # Process remaining response in chunks
            full_text = initial_text
            chunk_buffer = ""
            
            for chunk in streamer:
                chunk_buffer += chunk
                full_text += chunk
                
                # Process chunks at natural breaking points
                if len(chunk_buffer) >= tts_processor.chunk_size_threshold and any(p in chunk for p in ['.', ',', '!', '?', ';', ':', ' ']):
                    chunk_audio = await tts_processor.synthesize_speech(chunk_buffer)
                    await send_audio(chunk_audio)
                    metrics["tts_synths"] += 1
                    chunk_buffer = ""
            
            # Process any remaining text
            if chunk_buffer:
                chunk_audio = await tts_processor.synthesize_speech(chunk_buffer)
                await send_audio(chunk_audio)
                metrics["tts_synths"] += 1
            
            # Update conversation history
            gemma_processor._update_history(transcription, full_text)
            logger.info(f"TTS latency: {time.time() - tts_start:.2f}s")
            logger.info(f"Total segment processing time: {time.time() - segment_start:.2f}s")
            
        except asyncio.CancelledError:
            logger.info("Processing cancelled")
        except Exception as e:
            logger.error(f"Processing error: {e}")
            metrics["errors"] += 1
            error_audio = await tts_processor.synthesize_speech("Sorry, an error occurred.")
            await send_audio(error_audio)
        finally:
            await detector.set_tts_playing(False)

    async def detect_speech_segments():
        while True:
            try:
                speech_segment = await detector.get_next_segment()
                if speech_segment:
                    # Only cancel tasks if we're not actively playing TTS
                    if not detector.tts_playing:
                        await detector.cancel_current_tasks()
                    task = asyncio.create_task(process_speech_segment(speech_segment))
                    await detector.set_current_tasks(tts_task=task)
            except Exception as e:
                logger.error(f"Speech detection error: {e}")
                metrics["errors"] += 1
            await asyncio.sleep(0.01)

    async def receive_audio_and_images():
        async for message in websocket:
            try:
                data = json.loads(message)
                if "realtime_input" in data:
                    for chunk in data["realtime_input"]["media_chunks"]:
                        if chunk["mime_type"] == "audio/pcm":
                            await detector.add_audio(base64.b64decode(chunk["data"]))
                        elif chunk["mime_type"] == "image/jpeg" and not detector.tts_playing:
                            image_data = base64.b64decode(chunk["data"])
                            if image_data:
                                await gemma_processor.set_image(image_data)
                if "image" in data and not detector.tts_playing:
                    image_data = base64.b64decode(data["image"])
                    if image_data:
                        await gemma_processor.set_image(image_data)
            except Exception as e:
                logger.error(f"Error receiving data: {e}")
                metrics["errors"] += 1

    async def send_keepalive():
        while True:
            await websocket.ping()
            
            # Log performance metrics periodically
            runtime = time.time() - start_time
            if runtime > 60:  # Log every minute
                logger.info(f"Performance metrics - "
                           f"Uptime: {runtime:.1f}s, "
                           f"Transcriptions: {metrics['transcriptions']}, "
                           f"Generations: {metrics['generations']}, "
                           f"TTS: {metrics['tts_synths']}, "
                           f"Errors: {metrics['errors']}")
                
            await asyncio.sleep(20)

    try:
        await websocket.recv()
        logger.info("Client connected")
        await asyncio.gather(
            receive_audio_and_images(), 
            detect_speech_segments(), 
            send_keepalive()
        )
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    finally:
        await detector.cancel_current_tasks()

async def main():
    # Pre-initialize models
    WhisperTranscriber.get_instance()
    GemmaMultimodalProcessor.get_instance()
    KokoroTTSProcessor.get_instance()
    
    # Start resource cleanup task
    asyncio.create_task(cleanup_resources())
    
    logger.info("Starting WebSocket server on 0.0.0.0:9073")
    async with websockets.serve(
        handle_client, 
        "0.0.0.0", 
        9073, 
        ping_interval=20, 
        ping_timeout=60, 
        close_timeout=10,
        max_size=10 * 1024 * 1024  # 10MB max message size
    ):
        logger.info("Server started successfully")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
