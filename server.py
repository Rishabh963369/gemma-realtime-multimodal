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
import os
from datetime import datetime
import google.generativeai as genai 
from kokoro import KPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = "AIzaSyC0jAjJsgxGUBOvEw8h_L5HlzRVkaS9T-0"  # Replace with your actual key
genai.configure(api_key=GOOGLE_API_KEY)

class AudioSegmentDetector:
    """Detects speech segments based on audio energy levels"""
    
    def __init__(self, 
                 sample_rate=16000,
                 energy_threshold=0.015,
                 silence_duration=0.8,
                 min_speech_duration=0.5,  # Reduced for faster response
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
        
        # Add TTS playback lock
        self.tts_playing = False
        self.tts_lock = asyncio.Lock()
    
    async def set_tts_playing(self, is_playing):
        """Set TTS playback state"""
        async with self.tts_lock:
            self.tts_playing = is_playing
    
    async def add_audio(self, audio_bytes):
        """Add audio data to the buffer and check for speech segments"""
        async with self.lock:
            async with self.tts_lock:
                if self.tts_playing:
                    return None
                    
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
                                logger.info(f"Speech segment detected: {len(speech_segment)/2/self.sample_rate:.2f}s")
                                await self.segment_queue.put(speech_segment)
                                return speech_segment
                            
                        elif (len(self.audio_buffer) - self.speech_start_idx) > self.max_speech_samples * 2:
                            speech_segment = bytes(self.audio_buffer[self.speech_start_idx:
                                                             self.speech_start_idx + self.max_speech_samples * 2])
                            self.speech_start_idx += self.max_speech_samples * 2
                            self.segments_detected += 1
                            logger.info(f"Max duration speech segment: {len(speech_segment)/2/self.sample_rate:.2f}s")
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
    """Handles speech transcription using Distil-Whisper for faster inference"""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
        model_id = "distil-whisper/distil-small.en"
        logger.info(f"Loading {model_id}...")
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            model_kwargs={"use_flash_attention_2": True},
        )
        
        logger.info("Distil-Whisper model ready for transcription")
        self.transcription_count = 0
    
    async def transcribe(self, audio_bytes, sample_rate=16000):
        """Transcribe audio bytes to text using optimized pipeline"""
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_array) < 500:
                return ""
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipe(
                    {"array": audio_array, "sampling_rate": sample_rate},
                    generate_kwargs={
                        "task": "transcribe",
                        "language": "english",
                        "temperature": 0.0,
                        "max_new_tokens": 128
                    },
                    return_timestamps=False
                )
            )
            
            text = result.get("text", "").strip()
            self.transcription_count += 1
            logger.info(f"Transcription result: '{text}'")
            return text
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

class GeminiMultimodalProcessor:
    """Handles multimodal generation using Gemini API"""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        logger.info("Initializing Gemini Multimodal Processor...")
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.last_image = None
        self.last_image_timestamp = 0
        self.lock = asyncio.Lock()
        self.generation_count = 0
    
    async def set_image(self, image_data):
        """Cache the most recent image received"""
        async with self.lock:
            try:
                image = Image.open(io.BytesIO(image_data))
                new_size = (int(image.size[0] * 0.75), int(image.size[1] * 0.75))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                self.last_image = image
                self.last_image_timestamp = time.time()
                return True
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                return False
    
    async def generate(self, text):
        """Generate a response using Gemini API with the latest image and text input"""
        async with self.lock:
            try:
                if not self.last_image:
                    logger.warning("No image available for multimodal generation")
                    return f"No image context: {text}"
                
                img_byte_arr = io.BytesIO()
                self.last_image.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()

                prompt = (
                    "You are a helpful assistant providing spoken responses about images. "
                    "Keep your answers concise, fluent, and conversational. "
                    "Use natural oral language that's easy to listen to. "
                    "Avoid lengthy explanations and focus on the most important information. "
                    "Limit your response to 2-3 short sentences when possible. "
                    "Ask user to repeat or clarify if the request content is not clear or broken.\n\n"
                    f"User input: {text}"
                )
                
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    [prompt, {"mime_type": "image/jpeg", "data": img_bytes}]
                )
                
                output_text = response.text.strip()
                self.generation_count += 1
                logger.info(f"Gemini generation result ({len(output_text)} chars)")
                return output_text
                
            except Exception as e:
                logger.error(f"Gemini generation error: {e}")
                return f"Error processing: {text}"

class KokoroTTSProcessor:
    """Handles streaming text-to-speech conversion using Kokoro model"""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        logger.info("Initializing Kokoro TTS processor...")
        try:
            self.pipeline = KPipeline(lang_code='a')
            self.default_voice = 'af_sarah'
            self.chunk_size = 24000  # 1 second chunks at 24kHz
            logger.info("Kokoro TTS processor initialized successfully")
            self.synthesis_count = 0
        except Exception as e:
            logger.error(f"Error initializing Kokoro TTS: {e}")
            self.pipeline = None
    
    async def synthesize_speech(self, text, websocket):
        """Stream speech synthesis to WebSocket client in real-time"""
        if not text or not self.pipeline:
            return None
        
        try:
            logger.info(f"Synthesizing speech for text: '{text[:50]}...'")
            
            generator = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipeline(
                    text,
                    voice=self.default_voice,
                    speed=1.1,
                    split_pattern=r'[.!?。！？]+'
                )
            )
            
            async def stream_audio():
                buffer = []
                total_samples = 0
                
                for _, _, audio in generator:
                    buffer.append(audio)
                    total_samples += len(audio)
                    
                    if total_samples >= self.chunk_size:
                        chunk = np.concatenate(buffer)
                        audio_bytes = (chunk * 32767).astype(np.int16).tobytes()
                        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                        
                        await websocket.send(json.dumps({"audio": base64_audio}))
                        buffer = []
                        total_samples = 0
                        await asyncio.sleep(0.01)
                
                if buffer:
                    chunk = np.concatenate(buffer)
                    audio_bytes = (chunk * 32767).astype(np.int16).tobytes()
                    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                    await websocket.send(json.dumps({"audio": base64_audio}))
            
            await stream_audio()
            self.synthesis_count += 1
            logger.info("Speech synthesis streaming complete")
            
        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
            return None

async def handle_client(websocket):
    """Handles WebSocket client connection"""
    try:
        await websocket.recv()
        logger.info("Client connected")
        
        detector = AudioSegmentDetector()
        transcriber = WhisperTranscriber.get_instance()
        gemini_processor = GeminiMultimodalProcessor.get_instance()
        tts_processor = KokoroTTSProcessor.get_instance()
        
        async def send_keepalive():
            while True:
                try:
                    await websocket.ping()
                    await asyncio.sleep(10)
                except Exception:
                    break
        
        async def detect_speech_segments():
            while True:
                try:
                    speech_segment = await detector.get_next_segment()
                    if speech_segment:
                        transcription = await transcriber.transcribe(speech_segment)
                        if transcription:
                            response = await gemini_processor.generate(transcription)
                            await detector.set_tts_playing(True)
                            
                            try:
                                await tts_processor.synthesize_speech(response, websocket)
                            except websockets.exceptions.ConnectionClosed:
                                break
                            finally:
                                await detector.set_tts_playing(False)
                    
                    await asyncio.sleep(0.005)
                except websockets.exceptions.ConnectionClosed:
                    break
                except Exception as e:
                    logger.error(f"Error detecting speech: {e}")
                    await detector.set_tts_playing(False)
        
        async def receive_audio_and_images():
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if "realtime_input" in data:
                        for chunk in data["realtime_input"]["media_chunks"]:
                            if chunk["mime_type"] == "audio/pcm":
                                audio_data = base64.b64decode(chunk["data"])
                                await detector.add_audio(audio_data)
                            elif chunk["mime_type"] == "image/jpeg" and not detector.tts_playing:
                                image_data = base64.b64decode(chunk["data"])
                                await gemini_processor.set_image(image_data)
                    
                    if "image" in data and not detector.tts_playing:
                        image_data = base64.b64decode(data["image"])
                        await gemini_processor.set_image(image_data)
                        
                except Exception as e:
                    logger.error(f"Error receiving data: {e}")
        
        await asyncio.gather(
            receive_audio_and_images(),
            detect_speech_segments(),
            send_keepalive(),
            return_exceptions=True
        )
        
    except websockets.exceptions.ConnectionClosed:
        logger.info("Connection closed")
    except Exception as e:
        logger.error(f"Session error: {e}")
    finally:
        await detector.set_tts_playing(False)

async def main():
    """Main function to start the WebSocket server"""
    try:
        transcriber = WhisperTranscriber.get_instance()
        gemini_processor = GeminiMultimodalProcessor.get_instance()
        tts_processor = KokoroTTSProcessor.get_instance()
        
        logger.info("Starting WebSocket server on 0.0.0.0:9073")
        async with websockets.serve(
            handle_client, 
            "0.0.0.0", 
            9073,
            ping_interval=20,
            ping_timeout=60,
            close_timeout=10
        ):
            logger.info("WebSocket server running on 0.0.0.0:9073")
            await asyncio.Future()
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
