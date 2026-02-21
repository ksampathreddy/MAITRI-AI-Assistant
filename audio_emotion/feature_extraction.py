import librosa
import numpy as np

def extract_features(file_path):

    audio, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    if mfcc.shape[1] < 174:
        pad = 174 - mfcc.shape[1]
        mfcc = np.pad(mfcc,((0,0),(0,pad)))
    else:
        mfcc = mfcc[:,:174]

    return mfcc