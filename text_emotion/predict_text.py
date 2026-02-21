from transformers import pipeline

# Load emotion pipeline (public model)
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False
)

def predict_text_emotion(text):

    result = emotion_pipeline(text)

    return result[0]['label']