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
from kokoro import KPipeline  # Assuming this is your TTS library

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# CUDA optimizations for NVIDIA GPU with CUDA 12
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')  # Optimize for bfloat16

class AudioSegmentDetector:
    def __init__(self, sample_rate=16000, energy_threshold=0.01, silence_duration=0.2, min_speech_duration=0.2, max_speech_duration=3):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_samples = int(silence_duration * sample_rate)
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.max_speech_samples = int(max_speech_duration * sample_rate)
        self.audio_buffer = bytearray()
        self.is_speech_active = False
        self.silence_counter = 0
        self.speech_start_idx = 0
        self.segment_queue = asyncio.Queue(maxsize=4)  # Larger queue for parallelism

    async def add_audio(self, audio_bytes):
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
            return await asyncio.wait_for(self.segment_queue.get(), timeout=0.01)  # Ultra-low timeout
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
        self.device = "cuda:0"
        self.torch_dtype = torch.bfloat16  # Full bfloat16 for CUDA 12
        model_id = "distil-whisper/distil-large-v3"  # Faster Whisper variant
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
            model_kwargs={"use_flash_attention_2": True}  # CUDA 12 optimization
        )
        logger.info("Whisper model loaded (Distil variant)")

    async def transcribe(self, audio_bytes, sample_rate=16000):
        audio_array = torch.from_numpy(np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0).to(self.device)
        if len(audio_array) < 500:
            return ""
        result = await asyncio.get_event_loop().run_in_executor(None, lambda: self.pipe(
            {"raw": audio_array, "sampling_rate": sample_rate},
            generate_kwargs={"task": "transcribe", "language": "english", "max_new_tokens": 64}
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
        self.device = "cuda:0"
        model_id = "google/gemma-3-4b-it"  # Larger Gemma3 model to maximize VRAM
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"  # CUDA 12 optimization
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.last_image = None
        self.message_history = []

    async def set_image(self, image_data):
        if not image_data or len(image_data) < 100:
            return False
        image = Image.open(io.BytesIO(image_data)).resize((224, 224), Image.Resampling.LANCZOS)
        self.last_image = image
        self.message_history = []
        return True

    def _build_messages(self, text):
        messages = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant providing concise spoken responses."}]}]
        messages.extend(self.message_history)
        if self.last_image:
            messages.append({"role": "user", "content": [{"type": "image", "image": self.last_image}, {"type": "text", "text": text}]})
        else:
            messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
        return messages

    async def generate_streaming(self, text):
        messages = self._build_messages(text)
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_special_tokens=True, skip_prompt=True)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=64,  # Reduced for speed
            do_sample=False,
            use_cache=True,
            streamer=streamer
        )
        import threading
        threading.Thread(target=self.model.generate, kwargs=generation_kwargs).start()
        initial_text = ""
        for chunk in streamer:
            initial_text += chunk
            if len(initial_text) > 8 or "." in chunk or "," in chunk:
                break
        self.message_history = [{"role": "user", "content": [{"type": "text", "text": text}]}, {"role": "assistant", "content": [{"type": "text", "text": initial_text}]}]
        return streamer, initial_text

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
        if not text:
            return None
        audio = await asyncio.get_event_loop().run_in_executor(None, lambda: self.pipeline(
            text, voice=self.default_voice, speed=1.2, split_pattern=r'[.!?。！？]+'
        ))
        # Ensure the output is a NumPy array
        if audio:
            return np.concatenate([seg[2] for seg in audio])
        return None

async def handle_client(websocket):
    detector = AudioSegmentDetector()
    transcriber = WhisperTranscriber.get_instance()
    gemma_processor = GemmaMultimodalProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance()

    async def process_speech_segment(speech_segment):
        try:
            start_time = time.time()
            transcription = await transcriber.transcribe(speech_segment)
            if not transcription:
                return
            streamer, initial_text = await gemma_processor.generate_streaming(transcription)
            if not initial_text:
                return
            initial_audio = await tts_processor.synthesize_speech(initial_text)
            if initial_audio is not None:
                # Convert to NumPy array if it’s a Tensor, then to int16
                if isinstance(initial_audio, torch.Tensor):
                    initial_audio = initial_audio.cpu().numpy()
                audio_bytes = (initial_audio * 32767).astype(np.int16).tobytes()
                await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
            remaining_text = "".join(chunk for chunk in streamer)
            remaining_audio = await tts_processor.synthesize_speech(remaining_text)
            if remaining_audio is not None:
                # Convert to NumPy array if it’s a Tensor, then to int16
                if isinstance(remaining_audio, torch.Tensor):
                    remaining_audio = remaining_audio.cpu().numpy()
                audio_bytes = (remaining_audio * 32767).astype(np.int16).tobytes()
                await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
            logger.info(f"Processing took {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Processing error: {e}")

    async def detect_speech_segments():
        while True:
            speech_segment = await detector.get_next_segment()
            if speech_segment:
                asyncio.create_task(process_speech_segment(speech_segment))
            await asyncio.sleep(0.002)  # Minimal sleep

    async def receive_audio_and_images():
        try:
            async for message in websocket:
                data = json.loads(message)
                if "realtime_input" in data:
                    for chunk in data["realtime_input"]["media_chunks"]:
                        if chunk["mime_type"] == "audio/pcm":
                            await detector.add_audio(base64.b64decode(chunk["data"]))
                        elif chunk["mime_type"] == "image/jpeg":
                            await gemma_processor.set_image(base64.b64decode(chunk["data"]))
        except websockets.exceptions.ConnectionClosedError as e:
            logger.info(f"WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"Error in receive_audio_and_images: {e}")

    try:
        logger.info("Connection open")
        await asyncio.gather(receive_audio_and_images(), detect_speech_segments())
    except websockets.exceptions.ConnectionClosedError:
        logger.info("Connection closed by client")
    except Exception as e:
        logger.error(f"Handle client error: {e}")

async def main():
    logger.info("Server listening on 0.0.0.0:9073")
    WhisperTranscriber.get_instance()
    GemmaMultimodalProcessor.get_instance()
    KokoroTTSProcessor.get_instance()
    async with websockets.serve(handle_client, "0.0.0.0", 9073, ping_interval=20, ping_timeout=60, close_timeout=10):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
