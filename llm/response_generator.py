import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResponseGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("./maitri_llm")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "./maitri_llm"
        ).to(DEVICE)

    def generate(self, emotion, user_text):
        prompt = f"Emotion: {emotion}\nUser: {user_text}"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        ).to(DEVICE)

        outputs = self.model.generate(
            **inputs,
            max_length=100,
            temperature=0.7
        )

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return response