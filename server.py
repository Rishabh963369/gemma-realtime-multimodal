import asyncio
import json
import websockets
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, TextIteratorStreamer
from transformers import Gemma3ForConditionalGeneration  # Adjust if needed
import numpy as np
import logging
import sys
import io
from PIL import Image
import time
from concurrent.futures import ThreadPoolExecutor
from kokoro import KPipeline  # Placeholder; replace with actual import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Number of model instances to maximize GPU usage
NUM_INSTANCES = 1  # Adjust based on VRAM (e.g., 2-3 for 22 GB)

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
        self.segment_queue = asyncio.Queue(maxsize=10)  # Increased queue size
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
            return await asyncio.wait_for(self.segment_queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            return None

class WhisperTranscriber:
    def __init__(self, instance_id):
        self.device = f"cuda:{instance_id % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float32  # Full precision for more VRAM usage
        model_id = "openai/whisper-large-v3-turbo"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=False, use_safetensors=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline("automatic-speech-recognition", model=self.model, tokenizer=self.processor.tokenizer, feature_extractor=self.processor.feature_extractor, torch_dtype=self.torch_dtype, device=self.device)
        self.transcription_count = 0
        logger.info(f"Whisper model loaded on {self.device}")

    async def transcribe(self, audio_bytes_list, sample_rate=16000):
        try:
            audio_arrays = [np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0 for audio_bytes in audio_bytes_list]
            if not audio_arrays or all(len(arr) < 500 for arr in audio_arrays):
                logger.info("Audio too short for transcription")
                return ["" for _ in audio_bytes_list]
            with ThreadPoolExecutor(max_workers=4) as executor:
                result = await asyncio.get_event_loop().run_in_executor(executor, lambda: self.pipe(
                    [{"raw": audio, "sampling_rate": sample_rate} for audio in audio_arrays],
                    generate_kwargs={"task": "transcribe", "language": "english"}
                ))
            texts = [r.get("text", "").strip() for r in result]
            self.transcription_count += len(texts)
            logger.info(f"Transcription batch size {len(texts)}: {texts}")
            return texts
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ["" for _ in audio_bytes_list]

class GemmaMultimodalProcessor:
    def __init__(self, instance_id):
        self.device = f"cuda:{instance_id % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
        model_id = "google/gemma-3-4b-it"  # Adjust if model name differs
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float32,  # Full precision
            attn_implementation="flash_attention_2"  # Enable Flash Attention
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.last_image = None
        self.last_image_timestamp = 0
        self.lock = asyncio.Lock()
        self.message_history = []
        self.generation_count = 0
        logger.info(f"Gemma model loaded on {self.device}")

    async def set_image(self, image_data):
        async with self.lock:
            try:
                if not image_data or len(image_data) < 100:
                    logger.warning("Invalid or empty image data received")
                    return False
                image = Image.open(io.BytesIO(image_data))
                resized_image = image.resize((int(image.size[0] * 0.75), int(image.size[1] * 0.75)), Image.Resampling.LANCZOS)
                self.message_history = []
                self.last_image = resized_image
                self.last_image_timestamp = time.time()
                logger.info("Image set successfully")
                return True
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                return False

    def _build_messages(self, texts):
        messages = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant providing concise spoken responses about images or engaging in natural conversation."}]}]
        messages.extend(self.message_history)
        if self.last_image:
            messages.extend([{"role": "user", "content": [{"type": "image", "image": self.last_image}, {"type": "text", "text": text}]} for text in texts])
        else:
            messages.extend([{"role": "user", "content": [{"type": "text", "text": text}]} for text in texts])
        return messages

    def _update_history(self, user_texts, assistant_responses):
        self.message_history = [
            item
            for pair in zip(
                [{"role": "user", "content": [{"type": "text", "text": user_text}]} for user_text in user_texts],
                [{"role": "assistant", "content": [{"type": "text", "text": assistant_response}]} for assistant_response in assistant_responses]
            )
            for item in pair
        ]

    async def generate_streaming(self, texts):
        async with self.lock:
            try:
                messages = self._build_messages(texts)
                inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device)
                streamer = TextIteratorStreamer(self.processor.tokenizer, skip_special_tokens=True, skip_prompt=True)
                generation_kwargs = dict(**inputs, max_new_tokens=256, do_sample=False, use_cache=True, streamer=streamer)
                import threading
                threading.Thread(target=self.model.generate, kwargs=generation_kwargs).start()
                initial_texts = ["" for _ in texts]
                for chunk in streamer:
                    for i in range(len(initial_texts)):
                        initial_texts[i] += chunk
                        if len(initial_texts[i]) > 10 or "." in chunk or "," in chunk:
                            break
                self.generation_count += len(texts)
                logger.info(f"Generated initial texts: {initial_texts}")
                return streamer, initial_texts
            except Exception as e:
                logger.error(f"Gemma streaming error: {e}")
                return None, [f"Sorry, I couldn’t process that due to an error." for _ in texts]

class KokoroTTSProcessor:
    def __init__(self, instance_id):
        self.pipeline = KPipeline(lang_code='a')
        self.default_voice = 'af_sarah'
        self.synthesis_count = 0
        logger.info(f"Kokoro TTS loaded (instance {instance_id})")

    async def synthesize_speech(self, texts):
        if not texts or not self.pipeline:
            return [None for _ in texts]
        try:
            audio_segments_list = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                generator = await asyncio.get_event_loop().run_in_executor(executor, lambda: [self.pipeline(text, voice=self.default_voice, speed=1, split_pattern=r'[.!?。！？]+') for text in texts])
                for gen in generator:
                    audio_segments = []
                    for _, _, audio in gen:
                        audio_segments.append(audio)
                    if audio_segments:
                        combined_audio = np.concatenate(audio_segments)
                        audio_segments_list.append(combined_audio)
                    else:
                        audio_segments_list.append(None)
            self.synthesis_count += len(texts)
            logger.info(f"TTS synthesized batch: {len(audio_segments_list)} items")
            return audio_segments_list
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return [None for _ in texts]

async def handle_client(websocket):
    detector = AudioSegmentDetector()
    transcribers = [WhisperTranscriber(i) for i in range(NUM_INSTANCES)]
    gemma_processors = [GemmaMultimodalProcessor(i) for i in range(NUM_INSTANCES)]
    tts_processors = [KokoroTTSProcessor(i) for i in range(NUM_INSTANCES)]

    async def process_speech_segment(speech_segments):
        try:
            instance_idx = detector.segments_detected % NUM_INSTANCES
            transcriber = transcribers[instance_idx]
            gemma_processor = gemma_processors[instance_idx]
            tts_processor = tts_processors[instance_idx]

            transcriptions = await transcriber.transcribe(speech_segments)
            valid_transcriptions = [t for t in transcriptions if t and any(c.isalnum() for c in t)]
            if not valid_transcriptions:
                logger.info(f"Skipping empty transcriptions: {transcriptions}")
                return

            await detector.set_tts_playing(True)
            streamer, initial_texts = await gemma_processor.generate_streaming(valid_transcriptions)
            if not streamer or not initial_texts:
                logger.error("No response generated")
                initial_audios = await tts_processor.synthesize_speech(["Sorry, I couldn’t generate a response."])
                for audio in initial_audios:
                    if audio is not None:
                        audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                        await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
                return

            initial_audios = await tts_processor.synthesize_speech(initial_texts)
            for audio in initial_audios:
                if audio is not None:
                    audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                    await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
                else:
                    logger.error("Initial audio synthesis failed")

            remaining_texts = ["" for _ in valid_transcriptions]
            for chunk in streamer:
                for i in range(len(remaining_texts)):
                    remaining_texts[i] += chunk
            remaining_audios = await tts_processor.synthesize_speech(remaining_texts)
            for audio in remaining_audios:
                if audio is not None:
                    audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                    await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
                else:
                    logger.error("Remaining audio synthesis failed")

            gemma_processor._update_history(valid_transcriptions, [it + rt for it, rt in zip(initial_texts, remaining_texts)])
        except asyncio.CancelledError:
            logger.info("Processing cancelled")
        except Exception as e:
            logger.error(f"Processing error: {e}")
            error_audios = await tts_processor.synthesize_speech(["Sorry, an error occurred."])
            for audio in error_audios:
                if audio is not None:
                    audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                    await websocket.send(json.dumps({"audio": base64.b64encode(audio_bytes).decode('utf-8')}))
        finally:
            await detector.set_tts_playing(False)

    async def detect_speech_segments():
        batch = []
        while True:
            speech_segment = await detector.get_next_segment()
            if speech_segment:
                batch.append(speech_segment)
                if len(batch) >= NUM_INSTANCES:  # Process in batches
                    await detector.cancel_current_tasks()
                    task = asyncio.create_task(process_speech_segment(batch))
                    await detector.set_current_tasks(tts_task=task)
                    batch = []
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
                                for gemma_processor in gemma_processors:
                                    await gemma_processor.set_image(image_data)
                if "image" in data and not detector.tts_playing:
                    image_data = base64.b64decode(data["image"])
                    if image_data:
                        for gemma_processor in gemma_processors:
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
        await detector.cancel_current_tasks()

async def main():
    logger.info("Starting WebSocket server on 0.0.0.0:9073")
    async with websockets.serve(handle_client, "0.0.0.0", 9073, ping_interval=20, ping_timeout=60, close_timeout=10):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
