import asyncio
import json
import websockets
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForVision2Seq, TextIteratorStreamer
import numpy as np
import logging
import sys
import io
from PIL import Image
import time
from kokoro import KPipeline  # Assuming this is your custom TTS library

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class AudioSegmentDetector:
    def __init__(self, sample_rate=16000, energy_threshold=0.02, silence_duration=0.3, min_speech_duration=0.3, max_speech_duration=5):
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
        self.segments_detected = 0
        self.tts_playing = False
        self.tts_lock = asyncio.Lock()
        self.current_generation_task = None
        self.current_tts_task = None
        self.task_lock = asyncio.Lock()

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
                                await self.cancel_current_tasks()
                                await self.segment_queue.put(speech_segment)
                                return speech_segment
                        elif (len(self.audio_buffer) - self.speech_start_idx) > self.max_speech_samples * 2:
                            speech_segment = bytes(self.audio_buffer[self.speech_start_idx:self.speech_start_idx + self.max_speech_samples * 2])
                            self.speech_start_idx += self.max_speech_samples * 2
                            self.segments_detected += 1
                            logger.info(f"Max duration speech segment: {len(speech_segment) / 2 / self.sample_rate:.2f}s")
                            await self.cancel_current_tasks()
                            await self.segment_queue.put(speech_segment)
                            return speech_segment
        return None

    async def get_next_segment(self):
        try:
            return await asyncio.wait_for(self.segment_queue.get(), timeout=0.02)
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
        model_id = "distil-whisper/distil-large-v2"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            model_kwargs={"use_flash_attention_2": True}
        )
        self.transcription_count = 0
        logger.info("Distil-Whisper model loaded")

    async def transcribe(self, audio_bytes, sample_rate=16000):
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_array) < 300:
                logger.info("Audio too short for transcription")
                return ""
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipe({"raw": audio_array, "sampling_rate": sample_rate}, generate_kwargs={"task": "transcribe", "language": "english"})
            )
            text = result.get("text", "").strip()
            self.transcription_count += 1
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
        model_id = "llava-hf/llava-1.5-13b-hf"  # Updated to a valid model ID
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.last_image = None
        self.last_image_timestamp = 0
        self.lock = asyncio.Lock()
        self.message_history = []
        self.generation_count = 0
        logger.info("LLaVA-1.5-13B model loaded with 8-bit quantization")

    async def set_image(self, image_data):
        async with self.lock:
            try:
                if not image_data or len(image_data) < 100:
                    logger.warning("Invalid or empty image data received")
                    return False
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                resized_image = image.resize((336, 336), Image.Resampling.LANCZOS)
                self.message_history = []
                self.last_image = resized_image
                self.last_image_timestamp = time.time()
                logger.info("Image set successfully")
                return True
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                return False

    def _build_inputs(self, text):
        if self.last_image:
            prompt = f"You are a helpful assistant. Describe the image or answer based on it.\nUser: {text}"
            inputs = self.processor(text=prompt, images=self.last_image, return_tensors="pt").to(self.model.device)
        else:
            prompt = f"You are a helpful assistant providing concise spoken responses.\nUser: {text}"
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
        return inputs

    def _update_history(self, user_text, assistant_response):
        self.message_history = [{"role": "user", "content": user_text}, {"role": "assistant", "content": assistant_response}]

    async def generate_streaming(self, text):
        async with self.lock:
            try:
                inputs = self._build_inputs(text)
                streamer = TextIteratorStreamer(self.processor.tokenizer, skip_special_tokens=True, skip_prompt=True)
                generation_kwargs = dict(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=True,
                    temperature=0.6,
                    use_cache=True,
                    streamer=streamer
                )
                import threading
                threading.Thread(target=self.model.generate, kwargs=generation_kwargs).start()
                initial_text = ""
                for chunk in streamer:
                    initial_text += chunk
                    if len(initial_text) > 10 or "." in chunk or "," in chunk:
                        break
                self.generation_count += 1
                logger.info(f"Generated initial text: '{initial_text}'")
                return streamer, initial_text
            except Exception as e:
                logger.error(f"LLaVA streaming error: {e}")
                return None, "Sorry, I couldn’t process that due to an error."

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
        logger.info("Kokoro TTS loaded")

    async def synthesize_speech(self, text):
        if not text or not self.pipeline:
            return None
        try:
            audio_segments = []
            generator = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipeline(text, voice=self.default_voice, speed=1.2, split_pattern=r'[.,!?。！？]+')
            )
            for _, _, audio in generator:
                audio_segments.append(audio)
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                self.synthesis_count += 1
                logger.info(f"TTS synthesized: {len(combined_audio)} samples")
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
            if not transcription or not any(c.isalnum() for c in transcription):
                logger.info(f"Skipping empty transcription: '{transcription}'")
                return
            await detector.set_tts_playing(True)
            streamer, initial_text = await gemma_processor.generate_streaming(transcription)
            if not streamer or not initial_text:
                logger.error("No response generated")
                initial_audio = await tts_processor.synthesize_speech("Sorry, I couldn’t generate a response.")
                if initial_audio is not None:
                    audio_bytes = (initial_audio * 32767).astype(np.int16).tobytes()
                    await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
                return
            initial_audio = await tts_processor.synthesize_speech(initial_text)
            if initial_audio is not None:
                audio_bytes = (initial_audio * 32767).astype(np.int16).tobytes()
                await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
            else:
                logger.error("Initial audio synthesis failed")
            remaining_text = ""
            for chunk in streamer:
                remaining_text += chunk
            remaining_audio = await tts_processor.synthesize_speech(remaining_text)
            if remaining_audio is not None:
                audio_bytes = (remaining_audio * 32767).astype(np.int16).tobytes()
                await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
            else:
                logger.error("Remaining audio synthesis failed")
            gemma_processor._update_history(transcription, initial_text + remaining_text)
        except asyncio.CancelledError:
            logger.info("Processing cancelled")
        except Exception as e:
            logger.error(f"Processing error: {e}")
            error_audio = await tts_processor.synthesize_speech("Sorry, an error occurred.")
            if error_audio is not None:
                audio_bytes = (error_audio * 32767).astype(np.int16).tobytes()
                await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
        finally:
            await detector.set_tts_playing(False)

    async def detect_speech_segments():
        while True:
            speech_segment = await detector.get_next_segment()
            if speech_segment:
                await detector.cancel_current_tasks()
                task = asyncio.create_task(process_speech_segment(speech_segment))
                await detector.set_current_tasks(tts_task=task)
            await asyncio.sleep(0.005)

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

    async def send_keepalive():
        while True:
            await websocket.ping()
            await asyncio.sleep(15)

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
    async with websockets.serve(handle_client, "0.0.0.0", 9073, ping_interval=15, ping_timeout=45, close_timeout=5):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
