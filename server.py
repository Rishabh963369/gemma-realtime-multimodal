import asyncio
import json
import websockets
import base64
import torch
# Corrected imports: Use AutoModelForCausalLM for Gemma
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    AutoModelForCausalLM, # <--- CHANGE HERE
    BitsAndBytesConfig
)
import numpy as np
import logging
import sys
import io
from PIL import Image
import time
import os
from datetime import datetime
# Import Kokoro TTS library
from kokoro import KPipeline
import re
import traceback # For detailed error logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    # Added module and line number for better debugging
    format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Constants ---
SAMPLE_RATE = 16000
ENERGY_THRESHOLD = 0.015  # Adjust based on mic sensitivity
SILENCE_DURATION = 0.7    # Shorter silence to feel more responsive
MIN_SPEECH_DURATION = 0.5 # Shorter min duration
MAX_SPEECH_DURATION = 10  # Shorter max duration for faster turns
WEBSOCKET_PORT = 9073

class AudioSegmentDetector:
    """Detects speech segments based on audio energy levels and manages processing state"""

    def __init__(self,
                 sample_rate=SAMPLE_RATE,
                 energy_threshold=ENERGY_THRESHOLD,
                 silence_duration=SILENCE_DURATION,
                 min_speech_duration=MIN_SPEECH_DURATION,
                 max_speech_duration=MAX_SPEECH_DURATION):

        self.sample_rate = sample_rate
        self.bytes_per_sample = 2 # 16-bit PCM
        self.energy_threshold = energy_threshold
        self.silence_samples = int(silence_duration * sample_rate)
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.max_speech_samples = int(max_speech_duration * sample_rate)

        # Internal state for VAD
        self.audio_buffer = bytearray()
        self.is_speech_active = False
        self.silence_counter = 0
        self.speech_start_idx = 0
        self.buffer_lock = asyncio.Lock() # Protects buffer access
        self.segment_queue = asyncio.Queue()

        # Assistant response control state
        self.assistant_is_responding = False
        self.state_lock = asyncio.Lock() # Protects assistant_is_responding and current_processing_task
        self.current_processing_task = None
        self.last_interrupt_time = 0 # To prevent rapid-fire interrupts


    async def get_responding_state(self):
        """Safely get the current responding state."""
        async with self.state_lock:
            return self.assistant_is_responding

    async def set_assistant_responding(self, is_responding: bool, task: asyncio.Task = None):
        """Set assistant response state and optionally store the task."""
        async with self.state_lock:
            self.assistant_is_responding = is_responding
            # Store task only when starting, clear when stopping
            self.current_processing_task = task if is_responding else None
            logger.info(f"Assistant responding state set to: {is_responding}")
            if not is_responding:
                # Clear any VAD state when assistant stops, prevents carry-over noise detection
                self.is_speech_active = False
                self.silence_counter = 0

    async def cancel_current_processing(self):
        """Cancel any ongoing generation and TTS task"""
        task_to_cancel = None
        was_responding = False
        async with self.state_lock:
            task_to_cancel = self.current_processing_task
            was_responding = self.assistant_is_responding
            # Immediately mark as not responding and clear task reference
            self.assistant_is_responding = False
            self.current_processing_task = None

        if task_to_cancel and not task_to_cancel.done():
            logger.info("Attempting to cancel ongoing processing task.")
            task_to_cancel.cancel()
            try:
                # Wait briefly for cancellation to register
                # Shielding prevents wait_for itself from being cancelled if the outer task is cancelled
                await asyncio.wait_for(asyncio.shield(task_to_cancel), timeout=0.2)
            except asyncio.CancelledError:
                logger.info("Ongoing processing task cancelled successfully.")
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for task cancellation acknowledgment (task might still be cancelling).")
            except Exception as e:
                 logger.error(f"Unexpected error during task cancellation await: {e}")
            finally:
                 # Ensure state is False even if cancellation had issues (already set under lock)
                 logger.info(f"Responding state confirmed False after cancellation attempt (was {was_responding}).")
                 # Reset VAD state after cancellation
                 self.is_speech_active = False
                 self.silence_counter = 0
        elif was_responding:
             # If state was responding but no task found, ensure state is False
             logger.info("Responding state was true but no active task found. Resetting state.")
             # State already set to False under lock


    async def add_audio(self, audio_bytes):
        """Add audio data to the buffer and check for speech segments"""
        async with self.buffer_lock:
            self.audio_buffer.extend(audio_bytes)
            buffer_len_samples = len(self.audio_buffer) // self.bytes_per_sample

            # Trim buffer if it gets excessively long (e.g., > 30 seconds)
            max_buffer_samples = 30 * self.sample_rate
            if buffer_len_samples > max_buffer_samples:
                 trim_amount_bytes = (buffer_len_samples - max_buffer_samples) * self.bytes_per_sample
                 # Create a new bytearray slice instead of modifying in place potentially
                 self.audio_buffer = self.audio_buffer[trim_amount_bytes:]
                 # Adjust speech_start_idx relative to the new buffer start
                 self.speech_start_idx = max(0, self.speech_start_idx - (trim_amount_bytes // self.bytes_per_sample * self.bytes_per_sample)) # Adjust index based on bytes removed
                 logger.warning(f"Audio buffer trimmed to {max_buffer_samples / self.sample_rate}s")
                 buffer_len_samples = len(self.audio_buffer) // self.bytes_per_sample # Update length after trim


            # Use only the newly added audio for energy calculation
            num_new_samples = len(audio_bytes) // self.bytes_per_sample
            if num_new_samples == 0:
                return None

            try:
                # Process only the new chunk for energy detection
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                # Handle potential silence resulting in NaN/Inf energy
                if not np.all(np.isfinite(audio_array)):
                     energy = 0.0 # Treat non-finite values as silence
                elif len(audio_array) > 0:
                     energy = np.sqrt(np.mean(audio_array**2))
                else:
                     energy = 0.0
            except Exception as e:
                 logger.error(f"Error calculating energy: {e}")
                 energy = 0.0 # Treat error as silence


            # --- Speech Detection Logic ---
            if not self.is_speech_active and energy > self.energy_threshold:
                # Speech start detected
                self.is_speech_active = True
                # Mark start relative to the beginning of the *current* buffer content
                # The start is just before the newly added chunk
                self.speech_start_idx = max(0, len(self.audio_buffer) - len(audio_bytes))
                self.silence_counter = 0
                logger.debug(f"Speech start detected (energy: {energy:.6f}) at buffer index {self.speech_start_idx}")

                # --- Interrupt Logic ---
                # Check if the assistant is currently responding
                is_responding = await self.get_responding_state()
                if is_responding:
                    # Debounce interrupts: Only cancel if enough time passed since last one
                    now = time.monotonic()
                    if now - self.last_interrupt_time > 1.0: # Only interrupt if > 1s since last
                         logger.info("User speech detected while assistant responding. Cancelling assistant.")
                         self.last_interrupt_time = now
                         # Run cancellation concurrently, don't block VAD
                         asyncio.create_task(self.cancel_current_processing())
                    else:
                         logger.debug("Ignoring potential interrupt signal (too soon after last).")


            elif self.is_speech_active:
                # Calculate length from the *marked* start index to the end of the buffer
                current_speech_len_bytes = len(self.audio_buffer) - self.speech_start_idx
                current_speech_len_samples = current_speech_len_bytes // self.bytes_per_sample

                if energy > self.energy_threshold:
                    # Continued speech
                    self.silence_counter = 0
                else:
                    # Potential end of speech (silence)
                    self.silence_counter += num_new_samples

                    # Check if enough silence to end speech segment
                    if self.silence_counter >= self.silence_samples:
                        # End of speech segment is *before* the accumulated silence
                        # speech_end_idx = (len(self.audio_buffer) // self.bytes_per_sample - self.silence_counter) * self.bytes_per_sample
                        speech_end_idx = len(self.audio_buffer) - (self.silence_counter * self.bytes_per_sample)

                        # Ensure start index is valid and before end index
                        if self.speech_start_idx >= 0 and self.speech_start_idx < speech_end_idx:
                            speech_segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])
                            segment_len_samples = len(speech_segment_bytes) // self.bytes_per_sample

                            # Keep only the audio *after* the detected segment (including the silence)
                            self.audio_buffer = self.audio_buffer[speech_end_idx:]
                            # Reset VAD state for next detection
                            self.is_speech_active = False
                            self.silence_counter = 0
                            self.speech_start_idx = 0 # Reset relative start index for the new buffer

                            # Only queue if speech segment is within valid duration
                            if self.min_speech_samples <= segment_len_samples <= self.max_speech_samples :
                                logger.info(f"Speech segment detected (silence end): {segment_len_samples / self.sample_rate:.2f}s")
                                await self.segment_queue.put(speech_segment_bytes)
                                return speech_segment_bytes # Indicate segment found
                            elif segment_len_samples > self.max_speech_samples:
                                logger.warning(f"Segment ended by silence EXCEEDED max duration ({segment_len_samples / self.sample_rate:.2f}s > {self.max_speech_duration}s). Still processing.")
                                await self.segment_queue.put(speech_segment_bytes) # Process it anyway? Or trim? Let's process for now.
                                return speech_segment_bytes
                            else: # segment_len_samples < self.min_speech_samples
                                logger.debug(f"Speech segment too short (silence end): {segment_len_samples / self.sample_rate:.2f}s. Discarding.")
                        else:
                             logger.warning(f"Invalid VAD indices on silence end: start={self.speech_start_idx}, end={speech_end_idx}, buffer_len={len(self.audio_buffer)}. Resetting buffer.")
                             self.is_speech_active = False
                             self.silence_counter = 0
                             self.audio_buffer = bytearray() # Clear buffer on inconsistency
                             self.speech_start_idx = 0


                # Check if speech segment exceeds maximum duration (Force cut)
                # This check needs self.is_speech_active to be true
                if self.is_speech_active and current_speech_len_samples >= self.max_speech_samples:
                    # Cut exactly at max duration samples from the start index
                    speech_end_idx = self.speech_start_idx + (self.max_speech_samples * self.bytes_per_sample)
                    # Clamp end index to current buffer length just in case (shouldn't happen if logic is right)
                    speech_end_idx = min(speech_end_idx, len(self.audio_buffer))

                    if self.speech_start_idx < speech_end_idx:
                        speech_segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])
                        segment_len_samples = len(speech_segment_bytes) // self.bytes_per_sample

                        logger.info(f"Max duration speech segment force cut: {segment_len_samples / self.sample_rate:.2f}s")

                        # Update buffer: Keep audio *after* the cut segment
                        self.audio_buffer = self.audio_buffer[speech_end_idx:]
                        # Reset relative start index for the *new* buffer content
                        self.speech_start_idx = 0
                        # We forced a cut, but speech might still be active if energy is high in the remaining buffer.
                        # Reset silence counter, but DON'T reset is_speech_active yet. Let the next chunk decide.
                        self.silence_counter = 0

                        # Queue the cut segment
                        if segment_len_samples >= self.min_speech_samples: # Check min duration again for the cut segment
                            await self.segment_queue.put(speech_segment_bytes)
                            return speech_segment_bytes # Indicate segment found
                        else:
                            logger.debug(f"Max duration cut resulted in segment too short ({segment_len_samples / self.sample_rate:.2f}s). Discarding.")

                    else:
                         logger.warning(f"Invalid VAD indices on max duration cut: start={self.speech_start_idx}, end={speech_end_idx}. Resetting buffer.")
                         self.is_speech_active = False
                         self.silence_counter = 0
                         self.audio_buffer = bytearray()
                         self.speech_start_idx = 0


        return None # No segment finalized in this call

    async def get_next_segment(self):
        """Get the next available speech segment from the queue"""
        try:
            # Use a small timeout to prevent blocking indefinitely if queue is empty
            return await asyncio.wait_for(self.segment_queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            return None

# --- Model Processors (Singleton Pattern) ---

class WhisperTranscriber:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Creating WhisperTranscriber instance")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if hasattr(self, 'pipe') and self.pipe:
            logger.debug("WhisperTranscriber instance already exists")
            return # Avoid re-init
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Whisper using device: {self.device}")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() and torch.cuda.get_device_capability(self.device)[0] >= 7 else torch.float32
        logger.info(f"Whisper using torch_dtype: {self.torch_dtype}")

        model_id = "openai/whisper-large-v3"
        logger.info(f"Loading Whisper model: {model_id}...")
        self.pipe = None
        self.model = None
        self.processor = None
        try:
            # Use bnb quantization for Whisper as well if desired, or just load normally
            # quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) # Optional: if VRAM is tight

            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                # quantization_config=quantization_config, # Add this if using quantization
                # device_map="auto" # Use device_map if using quantization
            )
            if self.device != "cpu" and not hasattr(self.model, 'hf_device_map'): # Only move if not using device_map
                 self.model.to(self.device)

            self.processor = AutoProcessor.from_pretrained(model_id)

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device if not hasattr(self.model, 'hf_device_map') else None, # Let pipeline use device_map if present
                chunk_length_s=30, # Process audio in chunks
                stride_length_s=5  # Overlap chunks
            )
            logger.info("Whisper model ready.")
            self.transcription_count = 0
        except Exception as e:
             logger.error(f"FATAL: Failed to load Whisper model: {e}", exc_info=True)
             # Clean up partially loaded model/processor
             del self.model
             del self.processor
             del self.pipe
             self.model, self.processor, self.pipe = None, None, None
             raise RuntimeError(f"Could not initialize Whisper model: {e}") from e

    async def transcribe(self, audio_bytes, sample_rate=SAMPLE_RATE):
        if not self.pipe or not audio_bytes: return ""
        try:
            # Convert bytes to numpy float32 array
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Check minimum length requirement (Whisper needs at least a few samples)
            min_len_s = 0.1
            if len(audio_np) < sample_rate * min_len_s:
                logger.debug(f"Skipping transcription for very short audio ({len(audio_np)/sample_rate:.2f}s)")
                return "" # Skip very short segments

            logger.debug(f"Transcribing audio segment: {len(audio_np)/sample_rate:.2f}s")
            start_time = time.monotonic()

            # Define generation arguments
            generate_kwargs = {
                "language": "english",
                "task": "transcribe",
                # "temperature": 0.0 # Use default temperature for now
            }

            # Run inference in executor thread
            result = await asyncio.get_event_loop().run_in_executor(
                None,  # Use default executor
                lambda: self.pipe(
                    audio_np.copy(), # Pass a copy
                    batch_size=8,    # Adjust based on VRAM/performance
                    generate_kwargs=generate_kwargs
                )
            )

            end_time = time.monotonic()
            text = result.get("text", "").strip() if result else ""
            self.transcription_count += 1
            logger.info(f"Transcription #{self.transcription_count} ({end_time - start_time:.2f}s): '{text}'")
            return text
        except Exception as e:
            logger.error(f"Transcription error: {e}\n{traceback.format_exc()}")
            return ""

class GemmaMultimodalProcessor:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Creating GemmaMultimodalProcessor instance")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if hasattr(self, 'model') and self.model:
            logger.debug("GemmaMultimodalProcessor instance already exists")
            return # Avoid re-init
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Gemma using device: {self.device}")
        # Use a smaller Gemma model if VRAM is limited, e.g., "google/gemma-2b-it"
        model_id = "google/gemma-2-9b-it"
        logger.info(f"Loading Gemma model: {model_id}...")
        self.model = None
        self.processor = None
        try:
            # Quantization configuration - adjust compute type if needed
            quantization_config = BitsAndBytesConfig(
                 load_in_4bit=True,
                 bnb_4bit_compute_dtype=torch.bfloat16 # bfloat16 recommended for Ampere+ GPUs
                 # bnb_4bit_compute_dtype=torch.float16 # Use float16 if bfloat16 is not supported/available
            )
            # Determine the correct torch_dtype based on compute_dtype for consistency
            model_dtype = quantization_config.bnb_4bit_compute_dtype

            # --- Use AutoModelForCausalLM ---
            self.model = AutoModelForCausalLM.from_pretrained( # <--- CHANGE HERE
                model_id,
                device_map="auto", # Handles placing parts of the model across devices (CPU/GPU)
                quantization_config=quantization_config,
                torch_dtype=model_dtype # Load weights in the compute dtype
            )
            # --- Load the Processor ---
            # For Gemma, the processor handles both text and image inputs
            self.processor = AutoProcessor.from_pretrained(model_id)
            logger.info("Gemma model and processor ready.")

            self.last_image = None
            self.image_lock = asyncio.Lock()
            # Initialize with system prompt
            self.system_prompt = """You are a helpful assistant providing spoken responses about images and engaging in natural conversation. Keep responses concise, fluent, and conversational (1-3 short sentences). Use natural language suitable for speaking aloud.

Guidelines:
1. If the user asks about the image, describe relevant parts concisely.
2. If the user's input isn't about the image, respond naturally without forcing image descriptions.
3. If unsure about the image context, ask for clarification politely (e.g., "What about the image would you like to know?").
4. Maintain conversation context. Refer to previous turns naturally if needed.
5. Avoid overly long or complex sentences."""
            # Store history including the system prompt (check if model expects it this way)
            # Some models expect system prompt separate, others as first message.
            # Gemma IT models usually work well with chat templates starting with user/assistant turns.
            self.message_history = [] # Start history empty, apply template later
            self.max_history_len = 4 # Turns (User + Assistant = 1 turn) -> Keeps 4 pairs
            self.history_lock = asyncio.Lock()
            self.generation_count = 0


        except Exception as e:
             logger.error(f"FATAL: Failed to load Gemma model: {e}", exc_info=True)
             del self.model
             del self.processor
             self.model, self.processor = None, None
             raise RuntimeError(f"Could not initialize Gemma model: {e}") from e

    async def set_image(self, image_data):
        async with self.image_lock:
            try:
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                self.last_image = image
                logger.info(f"New image received. Size: {image.size}")
                async with self.history_lock:
                    self.message_history = [] # Clear history on new image
                    logger.info("Message history cleared due to new image.")
                return True
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                self.last_image = None
                return False

    async def _build_chat(self, text):
        """Build the chat structure for Gemma, including history and image."""
        async with self.history_lock:
            # Get current history
            current_history = list(self.message_history)

            # Prepare the new user message content
            user_content = []
            image_to_include = None
            async with self.image_lock: # Access image under its lock
                # Only include the image if it's relevant (e.g., first turn after image upload, or explicitly asked about)
                # For simplicity here, let's include it if it exists and history is short (implies recent image)
                # A more complex logic could check if the user prompt mentions "image", "picture", "this", "that" etc.
                if self.last_image and len(current_history) < 2 : # Include image on first/second turn
                   image_to_include = self.last_image
                   logger.debug("Including image in Gemma input.")
                elif self.last_image:
                   logger.debug("Image exists but not including in this turn's input (history longer).")
                else:
                    logger.warning("Generating response without image context in prompt.")

            # The processor expects images *only* in the final user message content list
            # It cannot handle images in the history turns.
            user_content.append(text) # Add user text

        # Apply chat template - this handles roles, special tokens etc.
        # Crucially, pass the image *only* when calling the processor, not into the template directly.
        chat_formatted = current_history + [{"role": "user", "content": text}] # History (text only) + current user turn (text only)

        # The processor will handle combining text and the optional image
        prompt = self.processor.apply_chat_template(
            chat_formatted,
            tokenize=False,
            add_generation_prompt=True # Adds marker for assistant's turn
        )
        # Return the formatted prompt string and the image (if any) separately
        return prompt, image_to_include


    async def update_history(self, user_text, assistant_response):
        """Update message history, ensuring it doesn't exceed max length."""
        # Clean up responses: remove potential incomplete tags or markers if needed
        assistant_response = assistant_response.replace("<bos>", "").replace("<eos>", "").strip()

        async with self.history_lock:
            # Add user message (text only for history)
            self.message_history.append({"role": "user", "content": user_text})
            # Add assistant response
            self.message_history.append({"role": "assistant", "content": assistant_response})
            # Trim history: Keep the last N pairs (max_history_len * 2 messages)
            if len(self.message_history) > self.max_history_len * 2:
                 # Keep the most recent max_history_len*2 messages
                 self.message_history = self.message_history[-(self.max_history_len * 2):]
            logger.debug(f"History updated. Length: {len(self.message_history)}")

    async def generate(self, text):
        if not self.model or not self.processor:
             logger.error("Gemma model or processor not ready for generation.")
             return "Error: Gemma model not ready."

        # Build the prompt string and determine if an image should be included
        prompt_string, image_input = await self._build_chat(text)

        try:
            # Prepare inputs using the processor
            # Pass the prompt string and the PIL image object (if available)
            inputs = self.processor(
                text=prompt_string,
                images=image_input, # Pass the PIL image here, or None
                return_tensors="pt",
                # padding=True, # Processor handles padding
                # truncation=True # Processor handles truncation
             ).to(self.model.device) # Move inputs to the same device as the model

            logger.debug(f"Gemma input keys: {inputs.keys()}")
            logger.debug(f"Input IDs shape: {inputs['input_ids'].shape}")
            if 'pixel_values' in inputs:
                logger.debug(f"Pixel values shape: {inputs['pixel_values'].shape}")


            # Generate response using run_in_executor for the blocking model.generate call
            start_time = time.monotonic()
            generate_ids = await asyncio.get_event_loop().run_in_executor(
                None, # Use default executor
                lambda: self.model.generate(
                    **inputs,
                    max_new_tokens=150, # Limit response length
                    do_sample=True,     # Use sampling for more natural responses
                    temperature=0.7,    # Control randomness (creativity vs. focus)
                    # top_k=50,         # Consider top-k sampling
                    # top_p=0.95,       # Consider nucleus sampling
                    pad_token_id=self.processor.tokenizer.eos_token_id # Important for generation
                )
            )
            end_time = time.monotonic()

            # Decode the generated tokens (excluding the input prompt tokens)
            # Shape of generate_ids is (batch_size, sequence_length)
            # Shape of inputs['input_ids'] is (batch_size, prompt_length)
            input_ids_len = inputs['input_ids'].shape[1]
            output_ids = generate_ids[:, input_ids_len:] # Slice generated part

            # Decode the output IDs
            generated_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False # Gemma might need specific space handling
            )[0].strip() # Get the first (and only) response in the batch

            # Update history *only after* successful generation
            # Pass the original user text, not the full prompt
            await self.update_history(text, generated_text)

            self.generation_count += 1
            logger.info(f"Gemma generation #{self.generation_count} ({end_time - start_time:.2f}s): '{generated_text[:100]}...' ({len(generated_text)} chars)")

            # Optional: Clear cache if memory issues persist, but can slow down subsequent runs
            # torch.cuda.empty_cache()

            return generated_text

        except Exception as e:
            logger.error(f"Gemma generation error: {e}\n{traceback.format_exc()}")
            torch.cuda.empty_cache() # Attempt cache clear on error
            return "Sorry, I encountered an error generating a response."


class KokoroTTSProcessor:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Creating KokoroTTSProcessor instance")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if hasattr(self, 'pipeline') and self.pipeline:
            logger.debug("KokoroTTSProcessor instance already exists")
            return # Avoid re-init
        logger.info("Initializing Kokoro TTS processor...")
        self.pipeline = None
        self.is_ready = False
        self.target_sample_rate = None # Store expected sample rate
        try:
            # Check Kokoro's expected language codes and voices
            # You might need to install language models for Kokoro separately
            self.pipeline = KPipeline(lang_code='en') # Assuming English focus
            # List available voices for 'en' if unsure: print(self.pipeline.list_voices())
            self.default_voice = 'en-US-Standard-F' # Example standard voice, confirm availability
            logger.info(f"Attempting Kokoro TTS dummy synthesis with voice: {self.default_voice}")
            # Attempt a dummy synthesis to check readiness and get sample rate
            dummy_audio = self.pipeline.synthesize("test initialization", voice=self.default_voice)

            if isinstance(dummy_audio, np.ndarray) and dummy_audio.size > 0:
                 # Kokoro might return different sample rates depending on voice/config
                 self.target_sample_rate = self.pipeline.target_sample_rate # Store rate if available
                 if self.target_sample_rate is None:
                     # If rate isn't directly available, try to infer (less reliable)
                     # This depends on Kokoro's internal structure, might need adjustment
                     logger.warning("Kokoro pipeline did not provide target_sample_rate directly. Assuming default.")
                     # Use a common default if needed, but ideally Kokoro provides it
                     self.target_sample_rate = 22050 # Or 24000, check Kokoro docs/output

                 logger.info(f"Kokoro TTS processor initialized successfully. Voice: {self.default_voice}, Target Rate: {self.target_sample_rate}Hz")
                 self.is_ready = True
                 self.synthesis_count = 0
            else:
                 logger.error("Kokoro TTS dummy synthesis failed or returned empty audio.")
                 self.pipeline = None # Mark as unusable

        except ImportError:
             logger.error("Kokoro library not found. Please install it (`pip install kokoro-tts`).")
             self.pipeline = None
        except Exception as e:
            # Catch potential errors like missing voice models
            logger.error(f"Error initializing Kokoro TTS: {e}", exc_info=True)
            self.pipeline = None

    async def synthesize_speech(self, text):
        if not self.is_ready or not text:
            logger.warning(f"TTS skipped. Ready: {self.is_ready}, Text provided: {bool(text)}")
            return None

        # Pre-process text slightly for TTS (optional)
        text = text.replace("*", "").strip() # Remove markdown like '*' which might be spoken
        if not text:
             logger.warning("TTS skipped: Text empty after cleaning.")
             return None

        try:
            start_time = time.monotonic()
            logger.info(f"Synthesizing speech for text (first 80): '{text[:80]}...'")

            # Use run_in_executor for the blocking Kokoro call
            audio_data_float32 = await asyncio.get_event_loop().run_in_executor(
                None, # Use default executor
                lambda: self.pipeline.synthesize(
                    text,
                    voice=self.default_voice,
                    speed=1.0 # Adjust speed if needed
                )
            )
            end_time = time.monotonic()

            if isinstance(audio_data_float32, np.ndarray) and audio_data_float32.size > 0:
                self.synthesis_count += 1
                duration = len(audio_data_float32) / self.target_sample_rate if self.target_sample_rate else 'N/A'
                elapsed = end_time - start_time
                logger.info(f"Speech synthesis #{self.synthesis_count} complete. Samples: {len(audio_data_float32)}, Est. Duration: {duration:.2f}s, Time: {elapsed:.2f}s")

                # Ensure the output is float32 between -1.0 and 1.0
                if audio_data_float32.dtype != np.float32:
                    logger.warning(f"Kokoro output dtype is {audio_data_float32.dtype}, converting to float32.")
                    audio_data_float32 = audio_data_float32.astype(np.float32)
                    # Normalize if conversion resulted in values outside [-1, 1]
                    max_abs = np.max(np.abs(audio_data_float32))
                    if max_abs > 1.0:
                         logger.warning(f"Normalizing audio data from max abs value {max_abs}")
                         audio_data_float32 /= max_abs
                elif np.max(np.abs(audio_data_float32)) > 1.0:
                     logger.warning("Kokoro float32 output exceeds [-1, 1], clipping might occur.")
                     # Optionally clip: np.clip(audio_data_float32, -1.0, 1.0, out=audio_data_float32)

                # torch.cuda.empty_cache() # Optional: Clear GPU cache if TTS uses GPU resources indirectly

                # Return the float32 numpy array
                return audio_data_float32
            else:
                 logger.warning("Speech synthesis resulted in empty or invalid audio.")
                 return None

        except Exception as e:
            logger.error(f"Speech synthesis error: {e}\n{traceback.format_exc()}")
            # torch.cuda.empty_cache() # Optional cache clear on error
            return None


# --- WebSocket Handler & Pipeline ---

async def run_full_response_pipeline(speech_segment, detector, transcriber, gemma_processor, tts_processor, websocket):
     """Handles the ASR -> LLM -> TTS pipeline for a single speech segment."""
     # This function runs as its own task, managed by detector.current_processing_task
     pipeline_start_time = time.monotonic()
     task_name = asyncio.current_task().get_name() if asyncio.current_task() else "ResponsePipeline"

     try:
          # 1. Mark start of processing (state already set before task creation)
          logger.info(f"Task '{task_name}': Starting response pipeline...")

          # 2. Transcribe Speech
          transcription = await transcriber.transcribe(speech_segment)
          if not transcription:
               logger.info(f"Task '{task_name}': Skipping empty transcription.")
               return # End pipeline early

          # --- Basic Filtering ---
          # More robust filtering: remove punctuation, lowercase, check against common short utterances/fillers
          cleaned_transcription = re.sub(r'[^\w\s]', '', transcription).lower().strip()
          words = [w for w in cleaned_transcription.split() if w]
          # Expand filler words list slightly
          common_fillers = {'yes', 'no', 'ok', 'okay', 'um', 'uh', 'uhuh', 'huh', 'yeah', 'hmm', 'bye', 'hi', 'hello', 'thanks', 'thank you'}
          is_filler = not words or (len(words) <= 2 and all(w in common_fillers for w in words))

          if is_filler:
               logger.info(f"Task '{task_name}': Skipping filtered transcription (likely filler): '{transcription}'")
               return # End pipeline early

          # 3. Send Interrupt Signal to Client (optional but good practice)
          # Tells client to stop playing any previous audio immediately
          logger.info(f"Task '{task_name}': Sending interrupt signal to client before new response.")
          try:
               # Ensure websocket is still open before sending
               if not websocket.closed:
                    await websocket.send(json.dumps({"interrupt": True}))
               else:
                   logger.warning(f"Task '{task_name}': Cannot send interrupt, connection closed.")
                   return # Cannot continue if connection is closed
          except websockets.exceptions.ConnectionClosed:
               logger.warning(f"Task '{task_name}': Connection closed while sending interrupt.")
               return # Cannot continue if connection is closed
          except Exception as send_err:
                logger.error(f"Task '{task_name}': Error sending interrupt signal: {send_err}")
                # Decide whether to continue or stop based on the error


          # 4. Generate Response using Gemma
          logger.info(f"Task '{task_name}': Generating response with Gemma for transcription: '{transcription}'")
          generated_text = await gemma_processor.generate(transcription) # Pass original transcription
          if not generated_text or "error" in generated_text.lower():
               logger.error(f"Task '{task_name}': Gemma generation failed or returned error: '{generated_text}'")
               # TODO: Optionally send a generic fallback audio message
               # Example: synthesized_fallback = await tts_processor.synthesize_speech("Sorry, I had trouble responding.")
               # Then send synthesized_fallback if not None
               return

          # 5. Synthesize Speech using Kokoro TTS
          logger.info(f"Task '{task_name}': Synthesizing speech with Kokoro TTS...")
          audio_response_float32 = await tts_processor.synthesize_speech(generated_text)

          # 6. Send Audio Response (if synthesis successful)
          if audio_response_float32 is not None and audio_response_float32.size > 0:
               try:
                   # Ensure the target sample rate from TTS matches our desired output rate (or resample if needed)
                   # Assuming detector.sample_rate (SAMPLE_RATE) is the desired output rate
                   tts_rate = tts_processor.target_sample_rate
                   if tts_rate != SAMPLE_RATE:
                       logger.warning(f"Task '{task_name}': TTS sample rate ({tts_rate}Hz) differs from target ({SAMPLE_RATE}Hz). Resampling needed (Not implemented). Sending at TTS rate.")
                       # Add resampling code here if necessary (e.g., using librosa or torchaudio.transforms.Resample)
                       # For now, we'll send at the TTS rate and assume the client can handle it, or configure TTS for the right rate.
                       pass # Placeholder for resampling

                   # Convert float32 numpy array [-1, 1] to int16 bytes
                   audio_int16 = np.clip(audio_response_float32 * 32767, -32768, 32767).astype(np.int16)
                   audio_bytes = audio_int16.tobytes()
                   base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

                   logger.info(f"Task '{task_name}': Sending synthesized audio ({len(audio_bytes)} bytes, rate={tts_rate}Hz) to client.")
                   if not websocket.closed:
                       await websocket.send(json.dumps({
                           "audio": base64_audio,
                           "sample_rate": tts_rate # Inform client about the sample rate
                        }))
                       logger.info(f"Task '{task_name}': Audio sent successfully.")
                   else:
                       logger.warning(f"Task '{task_name}': Connection closed before audio could be sent.")

               except websockets.exceptions.ConnectionClosed:
                    logger.warning(f"Task '{task_name}': Connection closed during audio send.")
               except Exception as send_err:
                    logger.error(f"Task '{task_name}': Error sending audio to client: {send_err}")
          else:
               logger.warning(f"Task '{task_name}': TTS synthesis failed or produced empty audio, no audio response sent.")

     except asyncio.CancelledError:
          # This specific task was cancelled (likely due to interruption)
          logger.info(f"Task '{task_name}': Response pipeline task was CANCELLED.")
          # State is reset in the finally block below
          raise # Re-raise cancellation to allow caller (detector state management) to know

     except Exception as e:
          logger.error(f"Task '{task_name}': Unhandled error in response pipeline: {e}\n{traceback.format_exc()}")

     finally:
          # 7. VERY IMPORTANT: Reset State ONLY IF this task is still the CURRENT task
          # This prevents a late-finishing cancelled task from resetting the state
          # if a *new* task has already started.
          async with detector.state_lock:
              if detector.current_processing_task is asyncio.current_task():
                  logger.info(f"Task '{task_name}': Pipeline finished. Resetting responding state.")
                  detector.assistant_is_responding = False
                  detector.current_processing_task = None
                  # Also reset VAD state here after assistant finishes normally
                  detector.is_speech_active = False
                  detector.silence_counter = 0
              else:
                   # This might happen if cancellation was slow and a new task was already set
                   logger.warning(f"Task '{task_name}': Pipeline finished, but it's no longer the current task. State NOT reset by this task.")

          pipeline_end_time = time.monotonic()
          logger.info(f"Task '{task_name}': Pipeline duration: {pipeline_end_time - pipeline_start_time:.2f}s")


async def process_speech_segments(detector, transcriber, gemma_processor, tts_processor, websocket):
    """Continuously checks for and processes detected speech segments."""
    task_name = asyncio.current_task().get_name() if asyncio.current_task() else "SegmentProcessor"
    logger.info(f"Task '{task_name}': Starting segment processing loop.")
    while True:
        try:
            # Check connection state first
            if websocket.closed:
                 logger.info(f"Task '{task_name}': WebSocket closed, stopping segment processing.")
                 break

            speech_segment = await detector.get_next_segment()

            if speech_segment:
                segment_start_time = time.monotonic()
                segment_id = f"Seg-{segment_start_time:.2f}" # Unique ID for logging this segment processing attempt
                logger.info(f"Task '{task_name}': [{segment_id}] Got segment ({len(speech_segment)/SAMPLE_RATE/2:.2f}s). Checking state...")

                # --- State Check ---
                # Check if the assistant is *already* responding.
                # Use the lock to ensure atomicity of check and potential state update.
                should_process = False
                async with detector.state_lock:
                    if not detector.assistant_is_responding:
                        logger.info(f"Task '{task_name}': [{segment_id}] Assistant is idle. Preparing to process.")
                        # Tentatively mark as responding and create task *within the lock*
                        # to prevent race condition if another segment arrives quickly.
                        detector.assistant_is_responding = True # Mark busy
                        should_process = True
                    else:
                        logger.info(f"Task '{task_name}': [{segment_id}] Assistant is busy with task {detector.current_processing_task.get_name() if detector.current_processing_task else 'Unknown'}. Discarding new segment.")
                        should_process = False # Do not process

                if should_process:
                    # --- Start Processing the New Segment ---
                    # Create and store the new task *outside* the lock to avoid holding it during task creation
                    pipeline_task = None
                    try:
                        logger.info(f"Task '{task_name}': [{segment_id}] Creating new task for response pipeline.")
                        pipeline_task = asyncio.create_task(
                            run_full_response_pipeline(
                                speech_segment, detector, transcriber, gemma_processor, tts_processor, websocket
                            ),
                            # Give task a name including segment ID for traceability
                            name=f"ResponsePipeline-{segment_id}"
                        )
                        # Now store the created task reference back into the detector state
                        async with detector.state_lock:
                            # Check again if state is still 'responding' and no other task took over
                            if detector.assistant_is_responding and detector.current_processing_task is None:
                                detector.current_processing_task = pipeline_task
                                logger.info(f"Task '{task_name}': [{segment_id}] Stored new pipeline task: {pipeline_task.get_name()}")
                            else:
                                # This case is unlikely if lock logic above is correct, but handle defensively
                                logger.warning(f"Task '{task_name}': [{segment_id}] State changed unexpectedly after check. Cancelling newly created task.")
                                pipeline_task.cancel()
                                # If state is no longer responding, reset it properly just in case
                                if not detector.assistant_is_responding:
                                     detector.current_processing_task = None


                    except Exception as task_creation_err:
                        logger.error(f"Task '{task_name}': [{segment_id}] Failed to create pipeline task: {task_creation_err}. Resetting state.", exc_info=True)
                        # Ensure state is reset if task creation failed
                        async with detector.state_lock:
                            detector.assistant_is_responding = False
                            detector.current_processing_task = None
                            # Reset VAD
                            detector.is_speech_active = False
                            detector.silence_counter = 0


            # Small sleep even if no segment, prevents tight loop, allows other tasks (like VAD) to run
            await asyncio.sleep(0.02)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Task '{task_name}': Connection closed during segment processing loop.")
            break # Exit loop
        except asyncio.CancelledError:
             logger.info(f"Task '{task_name}': Segment processing task cancelled.")
             break # Exit loop
        except Exception as e:
            logger.error(f"Task '{task_name}': Error in speech segment processing loop: {e}\n{traceback.format_exc()}")
            # Attempt to reset state cautiously after unexpected error
            try:
                 async with detector.state_lock:
                     # Only reset if state seems inconsistent
                     if detector.assistant_is_responding and (not detector.current_processing_task or detector.current_processing_task.done()):
                          logger.warning(f"Task '{task_name}': Resetting responding state due to loop error.")
                          detector.assistant_is_responding = False
                          detector.current_processing_task = None
                          detector.is_speech_active = False
                          detector.silence_counter = 0
            except Exception as reset_err:
                 logger.error(f"Task '{task_name}': Error trying to reset state after loop error: {reset_err}")

            await asyncio.sleep(1) # Pause after error before continuing loop

    logger.info(f"Task '{task_name}': Segment processing loop finished.")


async def receive_messages(detector, gemma_processor, websocket):
    """Handles incoming WebSocket messages (audio chunks, images)."""
    task_name = asyncio.current_task().get_name() if asyncio.current_task() else "Receiver"
    logger.info(f"Task '{task_name}': Starting message receive loop.")
    while True:
        message = None # Initialize message to None
        try:
            message = await websocket.recv()
            data = json.loads(message)

            # We don't necessarily need the responding state here, VAD handles interrupts based on audio energy.
            # is_responding_now = await detector.get_responding_state() # Maybe not needed

            # Handle image data
            if "image" in data:
                # Process image regardless of assistant state? Or queue it?
                # Current logic: set_image clears history, so maybe only do when idle.
                 is_responding = await detector.get_responding_state()
                 if not is_responding:
                    try:
                        image_data = base64.b64decode(data["image"])
                        logger.info(f"Task '{task_name}': Received standalone image data ({len(image_data)} bytes).")
                        success = await gemma_processor.set_image(image_data)
                        if success:
                            logger.info(f"Task '{task_name}': Image processed and history cleared.")
                        else:
                            logger.error(f"Task '{task_name}': Failed to process received image.")
                    except (base64.binascii.Error, ValueError, TypeError) as decode_err:
                        logger.error(f"Task '{task_name}': Error decoding base64 image: {decode_err}")
                    except Exception as img_err:
                        logger.error(f"Task '{task_name}': Error processing received image: {img_err}", exc_info=True)
                 else:
                    logger.info(f"Task '{task_name}': Ignoring image received while assistant is responding.")


            # Handle audio data (always add to VAD buffer)
            # Client should send audio chunks under "audio_data" key
            elif "audio_data" in data:
                try:
                    audio_bytes = base64.b64decode(data["audio_data"])
                    # logger.debug(f"Task '{task_name}': Received audio chunk: {len(audio_bytes)} bytes")
                    # Add audio to detector. This might trigger VAD -> Interrupt logic inside add_audio
                    await detector.add_audio(audio_bytes)
                except (base64.binascii.Error, ValueError, TypeError) as decode_err:
                     logger.error(f"Task '{task_name}': Error decoding base64 audio: {decode_err}")
                except Exception as audio_err:
                     logger.error(f"Task '{task_name}': Error processing received audio: {audio_err}", exc_info=True)

            # Handle explicit interrupt signal from client (if implemented by client)
            elif "interrupt" in data and data["interrupt"] is True:
                 logger.info(f"Task '{task_name}': Received explicit interrupt request from client.")
                 # Debounce like the VAD interrupt
                 now = time.monotonic()
                 if now - detector.last_interrupt_time > 1.0: # Use the same debounce timeout
                      is_responding = await detector.get_responding_state()
                      if is_responding:
                          detector.last_interrupt_time = now
                          logger.info(f"Task '{task_name}': Explicit interrupt triggered. Cancelling assistant.")
                          # Create task for cancellation, don't await it here
                          asyncio.create_task(detector.cancel_current_processing())
                      else:
                          logger.info(f"Task '{task_name}': Explicit interrupt received but assistant not responding.")
                 else:
                     logger.debug(f"Task '{task_name}': Ignoring explicit interrupt signal (too soon after last).")

            # Handle other message types if needed
            else:
                 logger.warning(f"Task '{task_name}': Received unknown message format: {list(data.keys())}")


        except websockets.exceptions.ConnectionClosedOK:
             logger.info(f"Task '{task_name}': WebSocket connection closed normally (OK).")
             break # Exit loop cleanly
        except websockets.exceptions.ConnectionClosedError as e:
             logger.warning(f"Task '{task_name}': WebSocket connection closed with error: {e.code} {e.reason}")
             break # Exit loop
        except websockets.exceptions.ConnectionClosed as e: # Catch any other close reason
             logger.warning(f"Task '{task_name}': WebSocket connection closed unexpectedly: {e.code} {e.reason}")
             break # Exit loop
        except json.JSONDecodeError:
            logger.error(f"Task '{task_name}': Received invalid JSON: {message[:200] if message else 'None'}...")
            # Decide whether to continue or break on invalid JSON
        except asyncio.CancelledError:
             logger.info(f"Task '{task_name}': Receive messages task cancelled.")
             break # Exit loop
        except Exception as e:
            logger.error(f"Task '{task_name}': Error processing received message: {e}\n{traceback.format_exc()}")
            # Decide if the loop should continue or break on other errors
            # Let's break on most other errors to be safe
            break

    logger.info(f"Task '{task_name}': Receive message loop finished.")

async def send_keepalive(websocket):
    """Sends WebSocket pings periodically to keep connection alive."""
    task_name = asyncio.current_task().get_name() if asyncio.current_task() else "Keepalive"
    logger.info(f"Task '{task_name}': Starting keepalive loop.")
    while True:
        try:
            if websocket.closed:
                logger.info(f"Task '{task_name}': Websocket closed, stopping keepalive.")
                break
            await websocket.ping()
            # logger.debug(f"Task '{task_name}': Sent ping.")
            await asyncio.sleep(15) # Send ping every 15 seconds
        except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
            logger.info(f"Task '{task_name}': Keepalive task stopped (connection closed or cancelled).")
            break
        except Exception as e:
             logger.error(f"Task '{task_name}': Error in keepalive task: {e}", exc_info=True)
             await asyncio.sleep(5) # Wait before retrying after an error

    logger.info(f"Task '{task_name}': Keepalive loop finished.")


async def handle_client(websocket):
    """Main handler for a single WebSocket client connection."""
    client_ip = websocket.remote_address
    logger.info(f"Client connected from {client_ip}")

    # Initialize components for this client session
    # Singletons are fetched, detector is instantiated per client
    detector = None
    transcriber = None
    gemma_processor = None
    tts_processor = None
    try:
        detector = AudioSegmentDetector()
        transcriber = WhisperTranscriber.get_instance()
        gemma_processor = GemmaMultimodalProcessor.get_instance()
        tts_processor = KokoroTTSProcessor.get_instance()

        # Crucial check: Ensure TTS is actually ready before proceeding
        if not tts_processor or not tts_processor.is_ready:
             logger.error(f"TTS Processor is not ready for client {client_ip}. Cannot provide audio responses.")
             # Attempt to send an error message before closing
             try:
                 await websocket.send(json.dumps({"error": "TTS service unavailable", "code": "TTS_INIT_FAILED"}))
             except Exception: pass # Ignore errors sending error message
             await websocket.close(code=1011, reason="Server TTS component unavailable") # Close gracefully
             return

        # Also check other critical components
        if not transcriber or not transcriber.pipe:
             logger.error(f"ASR Processor is not ready for client {client_ip}.")
             await websocket.close(code=1011, reason="Server ASR component unavailable")
             return
        if not gemma_processor or not gemma_processor.model:
             logger.error(f"LLM Processor is not ready for client {client_ip}.")
             await websocket.close(code=1011, reason="Server LLM component unavailable")
             return

        logger.info(f"All components ready for client {client_ip}.")

    except Exception as init_err:
         logger.error(f"Failed to initialize components for client {client_ip}: {init_err}", exc_info=True)
         try:
             await websocket.close(code=1011, reason="Server component initialization failed")
         except Exception: pass # Ignore errors during close
         # Ensure cleanup even if init fails partially
         del detector, transcriber, gemma_processor, tts_processor
         return


    # --- Background Tasks for this client ---
    receive_task = None
    segment_proc_task = None
    keepalive_task = None
    all_tasks = [] # Keep track of tasks to cancel later

    try:
        # Assign names to tasks for easier debugging
        receive_task = asyncio.create_task(receive_messages(detector, gemma_processor, websocket), name=f"Receiver-{client_ip}")
        segment_proc_task = asyncio.create_task(process_speech_segments(detector, transcriber, gemma_processor, tts_processor, websocket), name=f"SegmentProcessor-{client_ip}")
        keepalive_task = asyncio.create_task(send_keepalive(websocket), name=f"Keepalive-{client_ip}")
        all_tasks = [receive_task, segment_proc_task, keepalive_task]

        # Wait for any task to complete (usually receive_task on disconnect, or potentially segment_proc on error)
        done, pending = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)

        # Log results/exceptions from the completed task(s) for debugging
        for task in done:
            task_name = task.get_name()
            try:
                # Calling result() re-raises exceptions if the task failed
                result = task.result()
                logger.info(f"Task '{task_name}' completed successfully for client {client_ip}.")
            except asyncio.CancelledError:
                 logger.info(f"Task '{task_name}' was cancelled for client {client_ip}.")
            except websockets.exceptions.ConnectionClosedOK:
                 logger.info(f"Task '{task_name}' finished due to normal connection closure (OK) for client {client_ip}.")
            except websockets.exceptions.ConnectionClosedError as e:
                 logger.warning(f"Task '{task_name}' finished due to connection closure error ({e.code}) for client {client_ip}.")
            except Exception as e:
                 logger.error(f"Task '{task_name}' failed with an unhandled exception for client {client_ip}: {e}", exc_info=True)


    except Exception as e:
        # Catch errors in the main handler setup/wait logic itself
        logger.error(f"Unhandled exception in client handler main logic for {client_ip}: {e}", exc_info=True)
    finally:
        logger.info(f"Cleaning up client session for {client_ip}...")

        # 1. Explicitly cancel any ongoing processing pipeline task stored in the detector
        # This is important because the pipeline task isn't directly in all_tasks list
        if detector: # Ensure detector was initialized
            try:
                 logger.info(f"Client {client_ip}: Requesting cancellation of any active processing task.")
                 await detector.cancel_current_processing()
            except Exception as final_cancel_err:
                 logger.error(f"Client {client_ip}: Error during final cancel_current_processing call: {final_cancel_err}")

        # 2. Cancel all background tasks associated with this client
        logger.info(f"Client {client_ip}: Cancelling remaining background tasks: {[t.get_name() for t in all_tasks if t and not t.done()]}")
        cancelled_tasks = []
        for task in all_tasks:
             if task and not task.done():
                  task.cancel()
                  cancelled_tasks.append(task)

        # 3. Wait briefly for tasks to acknowledge cancellation
        if cancelled_tasks:
            logger.info(f"Client {client_ip}: Waiting for {len(cancelled_tasks)} tasks to finish cancelling...")
            # Use gather to wait for all cancellations, return_exceptions=True prevents one failed cancel from stopping others
            await asyncio.gather(*cancelled_tasks, return_exceptions=True)
            logger.info(f"Client {client_ip}: Finished waiting for task cancellations.")
        else:
             logger.info(f"Client {client_ip}: No pending background tasks needed cancellation.")

        # 4. Ensure WebSocket is closed
        if websocket and not websocket.closed:
            logger.info(f"Client {client_ip}: Closing WebSocket connection during cleanup.")
            try:
                await websocket.close(code=1001, reason="Client session ended")
            except Exception as close_err:
                logger.warning(f"Client {client_ip}: Error closing websocket during cleanup: {close_err}")

        # 5. Clean up resources (optional, depends if they need explicit release)
        # Python's GC should handle most objects like the detector instance when it goes out of scope.
        del detector # Explicitly remove reference if desired

        logger.info(f"Client {client_ip} cleanup complete.")


async def main():
    """Initializes models and starts the WebSocket server."""
    logger.info("--- Starting Server Initialization ---")
    try:
        # Pre-initialize singleton instances - crucial to catch errors early
        logger.info("Initializing Whisper Transcriber...")
        WhisperTranscriber.get_instance()
        logger.info("Initializing Gemma Processor...")
        GemmaMultimodalProcessor.get_instance()
        logger.info("Initializing Kokoro TTS Processor...")
        KokoroTTSProcessor.get_instance() # This also checks TTS readiness
        logger.info("--- Core Model Initialization Complete (or attempted) ---")
    except Exception as init_err:
         # Log the specific error during initialization
         logger.critical(f"FATAL: Core component initialization failed: {init_err}", exc_info=True)
         logger.critical("Server cannot start due to component initialization failure.")
         sys.exit(1) # Exit if core components fail

    # Double-check TTS readiness after initialization attempt
    tts_instance = KokoroTTSProcessor.get_instance()
    if not tts_instance or not tts_instance.is_ready:
        logger.warning("Kokoro TTS processor is not ready after initialization. Audio output will be disabled.")
        # Decide if server should run without TTS or exit
        # For now, let it run but log the warning prominently. Clients will be disconnected if they connect.
        # If TTS is absolutely essential, uncomment the sys.exit(1) line below.
        # logger.critical("Exiting because TTS is essential and failed to initialize.")
        # sys.exit(1)


    logger.info(f"Attempting to start WebSocket server on 0.0.0.0:{WEBSOCKET_PORT}")
    server = None
    try:
        # Define server behavior parameters
        server = await websockets.serve(
            handle_client,
            "0.0.0.0",
            WEBSOCKET_PORT,
            ping_interval=20,    # Send pings every 20s
            ping_timeout=30,     # Wait 30s for pong response
            close_timeout=10,    # Wait 10s for graceful close handshake
            # Increase max_size if large images/audio chunks are expected (default is 1MB)
            max_size=2*1024*1024 # Example: 2MB limit
        )
        logger.info(f"WebSocket server running successfully on ws://0.0.0.0:{WEBSOCKET_PORT}")
        await asyncio.Future()  # Keep the server running indefinitely until stopped

    except OSError as e:
         logger.error(f"SERVER STARTUP FAILED: Could not bind to address/port ({e}).")
         logger.error(f"Please check if port {WEBSOCKET_PORT} is already in use or if you have permission to bind.")
         sys.exit(1)
    except Exception as e:
        logger.error(f"SERVER STARTUP FAILED: An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # This block executes when the server stops (e.g., Ctrl+C)
        if server and server.is_serving():
            logger.info("Server shutting down...")
            server.close()
            await server.wait_closed()
            logger.info("Server shutdown complete.")

if __name__ == "__main__":
    # Optional: Increase asyncio debug logging if needed
    # loop = asyncio.get_event_loop()
    # loop.set_debug(True)
    # logging.getLogger('asyncio').setLevel(logging.DEBUG)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped manually via KeyboardInterrupt (Ctrl+C).")
    except Exception as main_err:
        # Catch any unexpected errors during asyncio.run() itself
        logger.critical(f"CRITICAL ERROR in main execution: {main_err}", exc_info=True)
        sys.exit(1) # Exit with error status
    finally:
        logger.info("--- Server Process Exiting ---")
