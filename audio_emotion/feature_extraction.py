import librosa
import numpy as np

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050)

        if np.abs(audio).mean() < 0.001:
            return None

        audio = audio / (np.max(np.abs(audio)) + 1e-6)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        max_len = 174

        if mfcc.shape[1] < max_len:
            pad = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad)))
        else:
            mfcc = mfcc[:, :max_len]

        return mfcc

    except Exception as e:
        print("Feature Error:", e)
        return None