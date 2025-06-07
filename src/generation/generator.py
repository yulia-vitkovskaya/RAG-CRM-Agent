from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms.base import LLM
from pydantic import BaseModel
import torch

# === 1. LocalLLMGenerator ===
class LocalLLMGenerator:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()

    def generate_answer(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# === 2. Обёртка для LangChain ===
class TinyLlamaLLM(LLM, BaseModel):
    generator: LocalLLMGenerator

    def _call(self, prompt: str, stop=None) -> str:
        return self.generator.generate_answer(prompt)

    @property
    def _llm_type(self) -> str:
        return "tinyllama-local"
