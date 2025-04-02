import asyncio
import json
import websockets
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, TextIteratorStreamer
import numpy as np
import logging
import sys
import io
from PIL import Image
import time
from kokoro import KPipeline  # Assuming this is your TTS library

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
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
        self.segment_queue = asyncio.Queue(maxsize=1)  # Reduced to 1 to prioritize latest input
        self.segments_detected = 0
        self.tts_playing = False
        self.current_tasks = set()  # Track all active tasks
        self.task_lock = asyncio.Lock()

    async def set_tts_playing(self, is_playing):
        async with self.task_lock:
            self.tts_playing = is_playing

    async def cancel_all_tasks(self):
        async with self.task_lock:
            for task in self.current_tasks.copy():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            self.current_tasks.clear()
            while not self.segment_queue.empty():
                await self.segment_queue.get()  # Clear queue to ensure only latest segment is processed
            await self.set_tts_playing(False)

    async def add_task(self, task):
        async with self.task_lock:
            self.current_tasks.add(task)

    async def remove_task(self, task):
        async with self.task_lock:
            self.current_tasks.discard(task)

    async def add_audio(self, audio_bytes):
        async with self.lock:
            self.audio_buffer.extend(audio_bytes)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
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
                                await self.cancel_all_tasks()  # Cancel everything on new segment
                                await self.segment_queue.put(speech_segment)
                                return speech_segment
                        elif (len(self.audio_buffer) - self.speech_start_idx) > self.max_speech_samples * 2:
                            speech_segment = bytes(self.audio_buffer[self.speech_start_idx:self.speech_start_idx + self.max_speech_samples * 2])
                            self.speech_start_idx += self.max_speech_samples * 2
                            self.segments_detected += 1
                            logger.info(f"Max duration speech segment: {len(speech_segment) / 2 / self.sample_rate:.2f}s")
                            await self.cancel_all_tasks()  # Cancel everything on new segment
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
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline("automatic-speech-recognition", model=self.model, tokenizer=self.processor.tokenizer, feature_extractor=self.processor.feature_extractor, torch_dtype=self.torch_dtype, device=self.device)

    async def transcribe(self, audio_bytes, sample_rate=16000):
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio_array) < 500:
            return ""
        result = await asyncio.get_event_loop().run_in_executor(None, lambda: self.pipe({"raw": audio_array, "sampling_rate": sample_rate}, generate_kwargs={"task": "transcribe", "language": "english"}))
        return result.get("text", "").strip()

class GemmaMultimodalProcessor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # Uncomment if Flash Attention is supported
            # attn_implementation="flash_attention_2"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.last_image = None
        self.message_history = []
        self.lock = asyncio.Lock()

    async def set_image(self, image_data):
        async with self.lock:
            self.last_image = Image.open(io.BytesIO(image_data)).resize((224, 224), Image.Resampling.LANCZOS)
            self.message_history = []

    async def generate_streaming(self, text):
        async with self.lock:
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]  # Simplified
            inputs = self.processor.apply_chat_template(messages, return_tensors="pt").to(self.device)
            streamer = TextIteratorStreamer(self.processor.tokenizer, skip_special_tokens=True)
            threading.Thread(target=self.model.generate, kwargs=dict(**inputs, max_new_tokens=512, streamer=streamer)).start()
            initial_text = ""
            for chunk in streamer:
                initial_text += chunk
                if len(initial_text) > 10 or "." in chunk:
                    break
            return streamer, initial_text

    def _update_history(self, user_text, assistant_response):
        self.message_history = [{"role": "user", "content": [{"type": "text", "text": user_text}]}, {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]}]

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

    async def synthesize_speech(self, text):
        audio = await asyncio.get_event_loop().run_in_executor(None, lambda: self.pipeline(text, voice=self.default_voice, speed=1))
        return np.concatenate([seg[2] for seg in audio]) if audio else None

async def handle_client(websocket):
    detector = AudioSegmentDetector()
    transcriber = WhisperTranscriber.get_instance()
    gemma_processor = GemmaMultimodalProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance()

    async def process_speech_segment(speech_segment):
        try:
            transcription = await transcriber.transcribe(speech_segment)
            if not transcription:
                return
            await detector.set_tts_playing(True)
            streamer, initial_text = await gemma_processor.generate_streaming(transcription)
            if not initial_text:
                return
            initial_audio = await tts_processor.synthesize_speech(initial_text)
            if initial_audio is not None:
                await websocket.send(json.dumps({"audio": base64.b64encode((initial_audio * 32767).astype(np.int16).tobytes()).decode('utf-8')}))
            remaining_text = "".join(chunk for chunk in streamer)
            remaining_audio = await tts_processor.synthesize_speech(remaining_text)
            if remaining_audio is not None:
                await websocket.send(json.dumps({"audio": base64.b64encode((remaining_audio * 32767).astype(np.int16).tobytes()).decode('utf-8')}))
            gemma_processor._update_history(transcription, initial_text + remaining_text)
        finally:
            await detector.set_tts_playing(False)
            await detector.remove_task(asyncio.current_task())

    async def detect_speech_segments():
        while True:
            speech_segment = await detector.get_next_segment()
            if speech_segment:
                task = asyncio.create_task(process_speech_segment(speech_segment))
                await detector.add_task(task)
            await asyncio.sleep(0.01)

    async def receive_audio_and_images():
        async for message in websocket:
            data = json.loads(message)
            if "realtime_input" in data:
                for chunk in data["realtime_input"]["media_chunks"]:
                    if chunk["mime_type"] == "audio/pcm":
                        await detector.add_audio(base64.b64decode(chunk["data"]))
                    elif chunk["mime_type"] == "image/jpeg" and not detector.tts_playing:
                        await gemma_processor.set_image(base64.b64decode(chunk["data"]))

    await asyncio.gather(receive_audio_and_images(), detect_speech_segments())

async def main():
    async with websockets.serve(handle_client, "0.0.0.0", 9073):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
