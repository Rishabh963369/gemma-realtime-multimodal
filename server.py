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
from concurrent.futures import ThreadPoolExecutor
from kokoro import KPipeline

# Configure logging (reduced verbosity)
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
        self.segment_queue = asyncio.Queue(maxsize=5)  # Reduced size for faster clearing
        self.segments_detected = 0
        self.tts_playing = False
        self.active_tasks = set()  # Track tasks for faster cancellation

    async def cancel_current_tasks(self):
        async with self.lock:
            for task in self.active_tasks.copy():
                if not task.done():
                    task.cancel()
            self.active_tasks.clear()
            self.audio_buffer.clear()  # Immediate buffer clear
            self.is_speech_active = False
            self.tts_playing = False
            while not self.segment_queue.empty():
                await self.segment_queue.get()

    async def add_audio(self, audio_bytes):
        async with self.lock:
            self.audio_buffer.extend(audio_bytes)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_array) > 0:
                energy = np.sqrt(np.mean(audio_array**2))
                if not self.is_speech_active and energy > self.energy_threshold:
                    await self.cancel_current_tasks()  # Immediate cancellation on new speech
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
                                await self.segment_queue.put(speech_segment)
                                return speech_segment
                        elif (len(self.audio_buffer) - self.speech_start_idx) > self.max_speech_samples * 2:
                            speech_segment = bytes(self.audio_buffer[self.speech_start_idx:self.speech_start_idx + self.max_speech_samples * 2])
                            self.speech_start_idx += self.max_speech_samples * 2
                            self.segments_detected += 1
                            logger.info(f"Max duration speech segment: {len(speech_segment) / 2 / self.sample_rate:.2f}s")
                            await self.segment_queue.put(speech_segment)
                            return speech_segment
        return None

    async def get_next_segment(self):
        try:
            return await asyncio.wait_for(self.segment_queue.get(), timeout=0.02)  # Reduced timeout
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
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline("automatic-speech-recognition", model=self.model, tokenizer=self.processor.tokenizer, feature_extractor=self.processor.feature_extractor, torch_dtype=self.torch_dtype, device=self.device)
        logger.info("Whisper model loaded")

    async def transcribe(self, audio_bytes, sample_rate=16000):
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_array) < 500:
                return ""
            result = await asyncio.get_event_loop().run_in_executor(None, lambda: self.pipe({"input_features": audio_array, "sampling_rate": sample_rate}, generate_kwargs={"task": "transcribe", "language": "english"}))
            return result.get("text", "").strip()
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
        model_id = "google/gemma-3-4b-it"
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # Uncomment if Flash Attention is supported
            # attn_implementation="flash_attention_2"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.last_image = None
        self.lock = asyncio.Lock()
        self.message_history = []
        self.executor = ThreadPoolExecutor(max_workers=1)  # For pre-tokenization
        logger.info("Gemma model loaded")

    async def set_image(self, image_data):
        if not image_data or len(image_data) < 100:
            return False
        try:
            image = Image.open(io.BytesIO(image_data))
            resized_image = image.resize((int(image.size[0] * 0.5), int(image.size[1] * 0.5)), Image.Resampling.LANCZOS)
            async with self.lock:
                self.message_history = []
                self.last_image = resized_image
            return True
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return False

    def _build_messages(self, text):
        messages = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant providing concise spoken responses about images or engaging in natural conversation."}]}]
        messages.extend(self.message_history)
        if self.last_image:
            messages.append({"role": "user", "content": [{"type": "image", "image": self.last_image}, {"type": "text", "text": text}]})
        else:
            messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
        return messages

    def _update_history(self, user_text, assistant_response):
        self.message_history = [{"role": "user", "content": [{"type": "text", "text": user_text}]}, {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]}]

    async def generate_streaming(self, text):
        async with self.lock:
            try:
                messages = self._build_messages(text)
                inputs = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device)
                )
                from transformers import TextIteratorStreamer
                streamer = TextIteratorStreamer(self.processor.tokenizer, skip_special_tokens=True, skip_prompt=True)
                generation_kwargs = dict(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,  # Greedy decoding for speed
                    use_cache=True,
                    streamer=streamer
                )
                with torch.amp.autocast(self.device):  # Mixed precision
                    import threading
                    threading.Thread(target=self.model.generate, kwargs=generation_kwargs).start()
                initial_text = ""
                for chunk in streamer:
                    initial_text += chunk
                    if len(initial_text) > 10 or "." in chunk or "," in chunk:  # Smaller chunk
                        break
                return streamer, initial_text
            except Exception as e:
                logger.error(f"Gemma streaming error: {e}")
                return None, "Sorry, I couldn’t process that."

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
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.audio_cache = {}

    async def synthesize_speech(self, text):
        if not text:
            return None
        cache_key = hash(text)
        if cache_key in self.audio_cache:
            return self.audio_cache[cache_key]
        try:
            audio = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.pipeline(text, voice=self.default_voice, speed=1.5)  # Increased speed
            )
            self.audio_cache[cache_key] = audio
            return audio
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
            if not transcription or not any(c.isalnum() for c in transcription):
                return
            detector.tts_playing = True
            streamer, initial_text = await gemma_processor.generate_streaming(transcription)
            if not streamer or not initial_text:
                audio = await tts_processor.synthesize_speech("Sorry, I couldn’t respond.")
                if audio is not None:
                    await websocket.send(json.dumps({"audio": base64.b64encode((audio * 32767).astype(np.int16).tobytes()).decode('utf-8')}))
                return
            audio = await tts_processor.synthesize_speech(initial_text)
            if audio is not None:
                await websocket.send(json.dumps({"audio": base64.b64encode((audio * 32767).astype(np.int16).tobytes()).decode('utf-8')}))
            remaining_text = ""
            for chunk in streamer:
                remaining_text += chunk
            if remaining_text:
                audio = await tts_processor.synthesize_speech(remaining_text)
                if audio is not None:
                    await websocket.send(json.dumps({"audio": base64.b64encode((audio * 32767).astype(np.int16).tobytes()).decode('utf-8')}))
            gemma_processor._update_history(transcription, initial_text + remaining_text)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Processing error: {e}")
        finally:
            detector.tts_playing = False

    async def detect_speech_segments():
        while True:
            speech_segment = await detector.get_next_segment()
            if speech_segment:
                task = asyncio.create_task(process_speech_segment(speech_segment))
                detector.active_tasks.add(task)
            await asyncio.sleep(0.005)  # Faster polling

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
        await detector.cancel_current_tasks()

async def main():
    WhisperTranscriber.get_instance()
    GemmaMultimodalProcessor.get_instance()
    KokoroTTSProcessor.get_instance()
    logger.info("Starting WebSocket server on 0.0.0.0:9073")
    async with websockets.serve(handle_client, "0.0.0.0", 9073, ping_interval=20, ping_timeout=60, close_timeout=10):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
