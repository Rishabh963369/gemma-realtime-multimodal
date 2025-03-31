import asyncio
import json
import websockets
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class AudioSegmentDetector:
    def __init__(self, sample_rate=16000, energy_threshold=0.02, silence_duration=0.5,
                 min_speech_duration=0.5, max_speech_duration=10):
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
        self.segment_queue = asyncio.Queue(maxsize=2)  # Limited queue size
        
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
                                try:
                                    await self.segment_queue.put_nowait(speech_segment)
                                    return speech_segment
                                except asyncio.QueueFull:
                                    self.segment_queue.get_nowait()  # Remove oldest
                                    await self.segment_queue.put(speech_segment)
                                    return speech_segment
                            
                        elif (len(self.audio_buffer) - self.speech_start_idx) > self.max_speech_samples * 2:
                            speech_segment = bytes(self.audio_buffer[self.speech_start_idx:
                                                            self.speech_start_idx + self.max_speech_samples * 2])
                            self.speech_start_idx += self.max_speech_samples * 2
                            try:
                                await self.segment_queue.put_nowait(speech_segment)
                                return speech_segment
                            except asyncio.QueueFull:
                                self.segment_queue.get_nowait()
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
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            batch_size=1  # Process one at a time for faster response
        )
        
    async def transcribe(self, audio_bytes, sample_rate=16000):
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_array) < 500:
                return ""
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipe(
                    {"array": audio_array, "sampling_rate": sample_rate},
                    generate_kwargs={"task": "transcribe", "language": "english", "temperature": 0.0}
                )
            )
            return result.get("text", "").strip()
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

class TextProcessor:
    def __init__(self):
        self.message_history = []
        self.max_history_messages = 4
        self.lock = asyncio.Lock()
    
    async def generate(self, text):
        async with self.lock:
            try:
                # Simple response generation (replace with your preferred model)
                response = f"As Grok 3 from xAI, I can tell you that: {text}"
                self._update_history(text, response)
                return response
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return f"Error processing: {text}"
    
    def _update_history(self, user_text, assistant_response):
        self.message_history.append({"role": "user", "content": user_text})
        self.message_history.append({"role": "assistant", "content": assistant_response})
        if len(self.message_history) > self.max_history_messages:
            self.message_history = self.message_history[-self.max_history_messages:]

class TTSSimulator:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.sample_rate = 16000
    
    async def synthesize_speech(self, text):
        try:
            # Simulate TTS with silence (for testing without actual TTS)
            duration = min(len(text) * 0.1, 5.0)  # Rough estimate: 0.1s per character, max 5s
            samples = int(self.sample_rate * duration)
            return np.zeros(samples, dtype=np.float32)
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

async def handle_client(websocket):
    try:
        await websocket.recv()
        logger.info("Client connected")
        
        detector = AudioSegmentDetector()
        transcriber = WhisperTranscriber.get_instance()
        text_processor = TextProcessor()
        tts_processor = TTSSimulator.get_instance()
        
        current_task = None
        lock = asyncio.Lock()
        
        async def send_keepalive():
            while True:
                try:
                    await websocket.ping()
                    await asyncio.sleep(10)
                except Exception:
                    break
        
        async def process_speech():
            nonlocal current_task
            while True:
                try:
                    speech_segment = await detector.get_next_segment()
                    if speech_segment:
                        async with lock:
                            if current_task and not current_task.done():
                                current_task.cancel()
                                try:
                                    await current_task
                                except asyncio.CancelledError:
                                    pass
                            
                            transcription = await transcriber.transcribe(speech_segment)
                            if transcription and len(transcription.split()) > 1:
                                await websocket.send(json.dumps({"interrupt": True}))
                                
                                response = await text_processor.generate(transcription)
                                audio = await tts_processor.synthesize_speech(response)
                                
                                if audio is not None:
                                    audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                                    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                                    await websocket.send(json.dumps({"audio": base64_audio}))
                    await asyncio.sleep(0.01)
                except websockets.exceptions.ConnectionClosed:
                    break
                except Exception as e:
                    logger.error(f"Speech processing error: {e}")
                    await asyncio.sleep(0.1)
        
        async def receive_audio():
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if "realtime_input" in data:
                        for chunk in data["realtime_input"]["media_chunks"]:
                            if chunk["mime_type"] == "audio/pcm":
                                audio_data = base64.b64decode(chunk["data"])
                                await detector.add_audio(audio_data)
                except Exception as e:
                    logger.error(f"Receive error: {e}")
        
        await asyncio.gather(
            receive_audio(),
            process_speech(),
            send_keepalive(),
            return_exceptions=True
        )
        
    except websockets.exceptions.ConnectionClosed:
        logger.info("Connection closed")
    except Exception as e:
        logger.error(f"Session error: {e}")

async def main():
    try:
        logger.info("Starting WebSocket server on 0.0.0.0:9073")
        async with websockets.serve(
            handle_client,
            "0.0.0.0",
            9073,
            ping_interval=20,
            ping_timeout=60,
            close_timeout=10
        ):
            await asyncio.Future()
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
