import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ResponseGenerator:
    def __init__(self):
        print("Loading TinyLlama Chat (Optimized)...")

        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )

        self.model.to(DEVICE)
        self.model.eval()

   
    def generate(self, face_emotion, audio_emotion, text_emotion, user_text):

        stress_level = self._compute_stress(
            face_emotion,
            audio_emotion,
            text_emotion
        )

        
        prompt = f"""
You are MAITRI, an AI assistant for astronauts.

STRICT RULES:
- Respond in ONLY 1–2 sentences
- DO NOT ask questions
- DO NOT simulate conversation
- DO NOT include labels like 'User' or 'Astronaut'
- Give direct emotional support only

Tone Guidance:
- HIGH stress → very calming
- MODERATE → reassuring
- LOW → supportive

Emotional State:
Face: {face_emotion}
Audio: {audio_emotion}
Text: {text_emotion}
Stress: {stress_level}

User:
{user_text}

Answer:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.6,
                top_p=0.8,
                repetition_penalty=1.3,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        if "Answer:" in response:
            response = response.split("Answer:")[-1]

        bad_phrases = [
            "Astronaut message",
            "User:",
            "Assistant:",
            "<|",
            "You are MAITRI"
        ]

        for phrase in bad_phrases:
            response = response.replace(phrase, "")

        response = response.strip()

        if "?" in response:
            response = response.split("?")[0] + "."

        sentences = response.split(".")
        response = ".".join(sentences[:2]).strip()

        response = response.replace("\n", " ").strip()

        if response == "" or len(response) < 5:
            response = "Stay calm and focused. You are not alone."

        return response
    
    def _compute_stress(self, face, audio, text):

        negative = ["sad", "fear", "angry", "disgust"]
        positive = ["happy", "surprise"]

        score = 0

        if face in negative:
            score += 2
        if audio in negative:
            score += 2
        if text in negative:
            score += 2

        if face in positive:
            score -= 1
        if audio in positive:
            score -= 1

        if score >= 4:
            return "HIGH"
        elif score >= 2:
            return "MODERATE"
        elif score <= 0:
            return "LOW"
        else:
            return "STABLE"