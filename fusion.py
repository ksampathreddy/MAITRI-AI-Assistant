import numpy as np

def weighted_fusion(face_pred, audio_pred, face_weight=0.5, audio_weight=0.5):
    fused = (face_weight * face_pred) + (audio_weight * audio_pred)
    return np.argmax(fused)
