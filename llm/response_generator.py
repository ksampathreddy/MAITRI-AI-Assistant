import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ResponseGenerator:
    def __init__(self):
        print("Loading TinyLlama Chat (CPU Optimized)...")

        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32  # Full CPU compatibility
        )

        self.model.to(DEVICE)
        self.model.eval()

    # =====================================
    # MAIN GENERATE FUNCTION (Optimized)
    # =====================================
    def generate(self, face_emotion, audio_emotion, text_emotion, user_text):

        stress_level = self._compute_stress(
            face_emotion,
            audio_emotion,
            text_emotion
        )

        prompt = f"""<|system|>
    You are MAITRI.
    You provide psychological and physiological support to astronauts.
    Do NOT introduce yourself.
    Do NOT simulate a conversation.
    Respond in exactly 2-3 sentences only.
    Be calm, professional, and mission-aware.
    <|user|>
    Current emotional state:
    - Facial emotion: {face_emotion}
    - Voice emotion: {audio_emotion}
    - Text emotion: {text_emotion}
    - Stress level: {stress_level}

    Astronaut message:
    {user_text}
    <|assistant|>
    """

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=15,      # Reduced further
                do_sample=False,
                repetition_penalty=1.2, # Prevent repetition
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]

        # Remove unwanted patterns
        response = response.replace("Astronaut response", "")
        response = response.strip()

        return response

    # =====================================
    # STRESS COMPUTATION (Deterministic)
    # =====================================
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