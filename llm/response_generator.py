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

        prompt = f"""
You are MAITRI, an AI assistant supporting astronauts during space missions.

Detected emotional state: {emotion}
User message: "{user_text}"

Respond with:
1. One empathetic supportive sentence.
2. One small coping suggestion.
3. One gentle follow-up question.

Keep it short and caring.
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(DEVICE)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3
        )

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # Safety fallback
        if user_text.lower() in response.lower() or len(response.strip()) < 20:
            return "I understand this feels heavy. Let’s pause and take a slow breath together. Would you like to share more about what’s been most difficult?"

        return response