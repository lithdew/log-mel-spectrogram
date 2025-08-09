# log-mel spectrogram (128 x N frames)
# Written by Kenta Iwasaki. All rights reserved.
# 2025-08-10

import librosa
import json

mel_filterbank = librosa.filters.mel(sr=16000, n_fft=400, n_mels=128)
print(mel_filterbank.shape)
print(mel_filterbank)

with open("mel_filterbank.json", "w") as f:
    json.dump(mel_filterbank.tolist(), f)