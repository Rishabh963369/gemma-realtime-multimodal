class GemmaMultimodalProcessor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        from accelerate import Accelerator
        self.accelerator = Accelerator()  # Automatically handles multi-GPU
        self.device = self.accelerator.device
        model_id = "google/gemma-3-12b-it"
        
        # Load model with auto device mapping for multi-GPU
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",  # Splits model across available GPUs
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Prepare model with accelerate for multi-GPU
        self.model = self.accelerator.prepare(self.model)
        
        self.last_image = None
        self.last_image_timestamp = 0
        self.lock = asyncio.Lock()
        self.message_history = []
        self.generation_count = 0
        logger.info("Gemma model loaded with multi-GPU support and bfloat16")

    async def generate_streaming(self, text):
        async with self.lock:
            try:
                messages = self._build_messages(text)
                inputs = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                ).to(self.model.device)
                
                from transformers import TextIteratorStreamer
                streamer = TextIteratorStreamer(self.processor.tokenizer, skip_special_tokens=True, skip_prompt=True)
                
                # Use accelerate to handle multi-GPU generation
                generation_kwargs = dict(
                    **inputs,
                    max_new_tokens=128,  # Reduced for speed
                    do_sample=False,
                    use_cache=True,
                    streamer=streamer
                )
                
                # Run generation in a separate thread
                import threading
                threading.Thread(
                    target=self.model.generate,
                    kwargs=generation_kwargs
                ).start()
                
                initial_text = ""
                for chunk in streamer:
                    initial_text += chunk
                    if len(initial_text) > 10 or "." in chunk or "," in chunk:
                        break
                self.generation_count += 1
                logger.info(f"Generated initial text: '{initial_text}'")
                return streamer, initial_text
            except Exception as e:
                logger.error(f"Gemma streaming error: {e}")
                return None, f"Sorry, I couldn’t process that due to an error."
