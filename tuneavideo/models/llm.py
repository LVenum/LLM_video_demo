from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class DetailedPromptGenerator:
    def __init__(self, model_name="gpt2", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device

    def generate(self, prompt, max_length=512, do_sample=True, temperature=0.3):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        detailed_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return detailed_text
