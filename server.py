import asyncio
import json
import websockets
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, GemmaForCausalLM
import numpy as np
import logging
import sys
import io
from PIL import Image
import time
from datetime import datetime
import re

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
        self.segment_queue = asyncio.Queue()
        self.tts_playing = False
        self.current_tasks = set()

    async def add_audio(self, audio_bytes):
        self.audio_buffer.extend(audio_bytes)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        if len(audio_array) == 0:
            return None
        
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
                        await self.segment_queue.put(speech_segment)
                        return speech_segment
                elif (len(self.audio_buffer) - self.speech_start_idx) > self.max_speech_samples * 2:
                    speech_segment = bytes(self.audio_buffer[self.speech_start_idx:self.speech_start_idx + self.max_speech_samples * 2])
                    self.speech_start_idx += self.max_speech_samples * 2
                    await self.segment_queue.put(speech_segment)
                    return speech_segment
        return None

    async def get_next_segment(self):
        try:
            return await asyncio.wait_for(self.segment_queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            return None

    async def cancel_tasks(self):
        for task in self.current_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self.current_tasks.clear()
        self.tts_playing = False

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
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=self.torch_dtype).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline("automatic-speech-recognition", model=self.model, tokenizer=self.processor.tokenizer,
                             feature_extractor=self.processor.feature_extractor, torch_dtype=self.torch_dtype, device=self.device)
        logger.info("Whisper model ready")

    async def transcribe(self, audio_bytes, sample_rate=16000):
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio_array) < 1000:
            return ""
        result = await asyncio.get_event_loop().run_in_executor(None, lambda: self.pipe(
            {"array": audio_array, "sampling_rate": sample_rate}, generate_kwargs={"task": "transcribe", "language": "english"}
        ))
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
        model_id = "google/gemma-2b-it"  # Using lighter model for faster response
        self.model = GemmaForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.last_image = None
        self.message_history = []

    async def set_image(self, image_data):
        try:
            image = Image.open(io.BytesIO(image_data)).resize((224, 224), Image.Resampling.LANCZOS)
            self.last_image = image
            self.message_history = []
            return True
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return False

    def _build_messages(self, text):
        messages = [{"role": "system", "content": "You are a helpful assistant providing concise spoken responses."}]
        messages.extend(self.message_history)
        content = [{"type": "text", "text": text}]
        if self.last_image:
            content.insert(0, {"type": "image", "image": self.last_image})
        messages.append({"role": "user", "content": content})
        return messages

    async def generate(self, text):
        if not self.last_image:
            return f"No image context: {text}"
        messages = self._build_messages(text)
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        generation = self.model.generate(**inputs, max_new_tokens=64, temperature=0.7)
        response = self.processor.decode(generation[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        self.message_history.extend([{"role": "user", "content": [{"type": "text", "text": text}]},
                                    {"role": "assistant", "content": [{"type": "text", "text": response}]}])
        if len(self.message_history) > 4:
            self.message_history = self.message_history[-4:]
        return response

class TTSProcessor:  # Simplified TTS for this example (replace with Kokoro if needed)
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def synthesize_speech(self, text):
        # Placeholder: Replace with actual TTS implementation
        await asyncio.sleep(0.1)  # Simulate synthesis
        return np.zeros(16000, dtype=np.float32)  # Dummy audio

async def handle_client(websocket):
    await websocket.recv()
    logger.info("Client connected")
    
    detector = AudioSegmentDetector()
    transcriber = WhisperTranscriber.get_instance()
    gemma_processor = GemmaMultimodalProcessor.get_instance()
    tts_processor = TTSProcessor.get_instance()

    async def send_keepalive():
        while True:
            await websocket.ping()
            await asyncio.sleep(10)

    async def process_speech():
        while True:
            segment = await detector.get_next_segment()
            if segment:
                transcription = await transcriber.transcribe(segment)
                if not transcription or not any(c.isalnum() for c in transcription) or len(transcription.split()) <= 1:
                    continue
                
                await websocket.send(json.dumps({"interrupt": True}))
                await detector.cancel_tasks()
                
                response = await gemma_processor.generate(transcription)
                audio = await tts_processor.synthesize_speech(response)
                audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
            await asyncio.sleep(0.01)

    async def receive_data():
        async for message in websocket:
            data = json.loads(message)
            if "realtime_input" in data:
                for chunk in data["realtime_input"]["media_chunks"]:
                    if chunk["mime_type"] == "audio/pcm":
                        await detector.add_audio(base64.b64decode(chunk["data"]))
                    elif chunk["mime_type"] == "image/jpeg" and not detector.tts_playing:
                        await gemma_processor.set_image(base64.b64decode(chunk["data"]))
            elif "image" in data and not detector.tts_playing:
                await gemma_processor.set_image(base64.b64decode(data["image"]))

    await asyncio.gather(receive_data(), process_speech(), send_keepalive(), return_exceptions=True)

async def main():
    logger.info("Starting WebSocket server on 0.0.0.0:9073")
    async with websockets.serve(handle_client, "0.0.0.0", 9073, ping_interval=20, ping_timeout=60, close_timeout=10):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
