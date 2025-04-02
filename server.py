import asyncio
import json
import websockets
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import logging
import sys
import io
from PIL import Image
import time
from concurrent.futures import ThreadPoolExecutor
import re

# Assuming kokoro is an external library - replace with actual import
from kokoro import KPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class AudioSegmentDetector:
    def __init__(self, sample_rate=16000, energy_threshold=0.015, silence_duration=0.3, min_speech_duration=0.3, max_speech_duration=8):
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
        self.segment_queue = asyncio.Queue(maxsize=5)
        self.active_tasks = set()
        self.tts_playing = False
        self.last_processed_time = time.time()

    async def cancel_all_tasks(self):
        async with self.lock:
            for task in self.active_tasks.copy():
                if not task.done():
                    task.cancel()
            self.active_tasks.clear()
            self.audio_buffer.clear()
            self.is_speech_active = False
            self.tts_playing = False
            while not self.segment_queue.empty():
                await self.segment_queue.get()

    async def add_audio(self, audio_bytes):
        async with self.lock:
            current_time = time.time()
            if current_time - self.last_processed_time > 0.1:
                self.last_processed_time = current_time
                self.audio_buffer.extend(audio_bytes)
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                if len(audio_array) > 0:
                    energy = np.mean(audio_array**2)
                    if energy > self.energy_threshold:
                        if not self.is_speech_active:
                            await self.cancel_all_tasks()
                            self.is_speech_active = True
                            self.speech_start_idx = len(self.audio_buffer) - len(audio_bytes)
                        self.silence_counter = 0
                    elif self.is_speech_active:
                        self.silence_counter += len(audio_array)
                        if self.silence_counter >= self.silence_samples:
                            speech_end_idx = len(self.audio_buffer) - self.silence_samples
                            if speech_end_idx > self.speech_start_idx:
                                speech_segment = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])
                                if len(speech_segment) >= self.min_speech_samples * 2:
                                    try:
                                        await self.segment_queue.put_nowait(speech_segment)
                                    except asyncio.QueueFull:
                                        pass
                                self.audio_buffer = self.audio_buffer[speech_end_idx:]
                            self.is_speech_active = False
                            self.silence_counter = 0

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
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline("automatic-speech-recognition", model=self.model, tokenizer=self.processor.tokenizer, feature_extractor=self.processor.feature_extractor, torch_dtype=self.torch_dtype, device=self.device)
        logger.info("Whisper model loaded")

    async def transcribe(self, audio_bytes, sample_rate=16000):
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_array) < 500:
                return ""
            result = await asyncio.get_event_loop().run_in_executor(None, lambda: self.pipe({"raw": audio_array, "sampling_rate": sample_rate}, generate_kwargs={"task": "transcribe", "language": "english"}))
            text = result.get("text", "").strip()
            logger.info(f"Transcription: '{text}'")
            return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

class GemmaMultimodalProcessor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Note: Gemma3ForConditionalGeneration was in original code but not imported
        # Using a placeholder - replace with actual model
        from transformers import AutoModelForCausalLM
        model_id = "google/gemma-2b-it"  # Using smaller model for speed
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.last_image = None
        self.last_image_timestamp = 0
        self.lock = asyncio.Lock()
        self.message_history = []

    async def set_image(self, image_data):
        async with self.lock:
            try:
                if not image_data or len(image_data) < 100:
                    return False
                image = Image.open(io.BytesIO(image_data))
                resized_image = image.resize((int(image.size[0] * 0.75), int(image.size[1] * 0.75)), Image.Resampling.LANCZOS)
                self.message_history = []
                self.last_image = resized_image
                self.last_image_timestamp = time.time()
                return True
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                return False

    def _build_messages(self, text):
        messages = [{"role": "system", "content": "You are a helpful assistant providing concise spoken responses."}]
        messages.extend(self.message_history)
        if self.last_image:
            messages.append({"role": "user", "content": f"[Image provided] {text}"})
        else:
            messages.append({"role": "user", "content": text})
        return messages

    async def generate_streaming(self, text):
        async with self.lock:
            try:
                messages = self._build_messages(text)
                inputs = self.processor(messages, return_tensors="pt").to(self.model.device)
                from transformers import TextIteratorStreamer
                streamer = TextIteratorStreamer(self.processor.tokenizer, skip_special_tokens=True)
                generation_kwargs = dict(**inputs, max_new_tokens=64, do_sample=True, temperature=0.7, streamer=streamer)
                import threading
                threading.Thread(target=self.model.generate, kwargs=generation_kwargs).start()
                initial_text = ""
                for chunk in streamer:
                    initial_text += chunk
                    if len(initial_text) > 20 or "." in chunk or "," in chunk:
                        break
                return streamer, initial_text
            except Exception as e:
                logger.error(f"Gemma streaming error: {e}")
                return None, f"Sorry, I couldn’t process that."

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
        self.audio_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=2)

    async def synthesize_speech(self, text):
        if not text:
            return None
            
        cache_key = hash(text)
        if cache_key in self.audio_cache:
            return self.audio_cache[cache_key]

        try:
            sentences = re.split(r'[.!?]+', text)
            audio_tasks = []
            
            for sentence in sentences:
                if sentence.strip():
                    audio_tasks.append(
                        asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            lambda s=sentence: self.pipeline(s.strip(), voice=self.default_voice, speed=1.1)
                        )
                    )
            
            audio_segments = []
            for task in asyncio.as_completed(audio_tasks):
                _, _, audio = await task
                audio_segments.append(audio)
                
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                self.audio_cache[cache_key] = combined_audio
                return combined_audio
            return None
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

async def handle_client(websocket):
    detector = AudioSegmentDetector()
    transcriber = WhisperTranscriber.get_instance()
    gemma_processor = GemmaMultimodalProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance()

    async def process_speech_segment(speech_segment):
        try:
            transcription = await transcriber.transcribe(speech_segment)
            if not transcription or detector.is_speech_active:
                return
                
            streamer, initial_text = await gemma_processor.generate_streaming(transcription)
            if not streamer or detector.is_speech_active:
                return
                
            initial_audio = await tts_processor.synthesize_speech(initial_text)
            if initial_audio is not None and not detector.is_speech_active:
                audio_bytes = (initial_audio * 32767).astype(np.int16).tobytes()
                await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
                
            task = asyncio.create_task(detector.cancel_all_tasks())
            detector.active_tasks.add(task)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Processing error: {e}")

    async def detect_speech_segments():
        while True:
            try:
                speech_segment = await detector.segment_queue.get()
                if not detector.is_speech_active:
                    task = asyncio.create_task(process_speech_segment(speech_segment))
                    detector.active_tasks.add(task)
            except Exception as e:
                logger.error(f"Segment processing error: {e}")
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
                            await gemma_processor.set_image(base64.b64decode(chunk["data"]))
                if "image" in data and not detector.tts_playing:
                    image_data = base64.b64decode(data["image"])
                    if image_data:
                        await gemma_processor.set_image(image_data)
            except Exception as e:
                logger.error(f"Error receiving data: {e}")

    async def send_keepalive():
        while True:
            await websocket.ping()
            await asyncio.sleep(20)

    try:
        await websocket.recv()
        logger.info("Client connected")
        await asyncio.gather(receive_audio_and_images(), detect_speech_segments(), send_keepalive())
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    finally:
        await detector.cancel_all_tasks()

async def main():
    WhisperTranscriber.get_instance()
    GemmaMultimodalProcessor.get_instance()
    KokoroTTSProcessor.get_instance()
    logger.info("Starting WebSocket server on 0.0.0.0:9073")
    async with websockets.serve(handle_client, "0.0.0.0", 9073, ping_interval=20, ping_timeout=60, close_timeout=10):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
