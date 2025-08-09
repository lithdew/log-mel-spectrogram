# log-mel spectrogram (128 x N frames)
# Written by Kenta Iwasaki. All rights reserved.
# 2025-08-10

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import json

N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * 16000

def log_mel_spectrogram(
    audio: np.ndarray,
    n_mels: int = 80,
    padding: int = 0,
):
    audio = torch.from_numpy(audio)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=N_FFT, n_mels=n_mels))
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

audio, _ = librosa.load("samples_jfk.wav", sr=16000, dtype=np.float32)
print(audio.shape)
print(audio)

mel_spec = log_mel_spectrogram(audio, n_mels=128, padding=N_SAMPLES)
print(mel_spec.shape)
print(mel_spec)

with open("mel_example.json", "w") as f:
    json.dump(mel_spec.numpy().tolist(), f)