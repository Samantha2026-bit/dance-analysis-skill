from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .models import AudioFeatures


def analyze_audio(path: str | Path) -> AudioFeatures:
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    if y.size == 0:
        return AudioFeatures(0.0, np.array([]), np.array([]), 0.0)
    onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_envelope, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_envelope, sr=sr)
    tempo_value = float(np.asarray(tempo).reshape(-1)[0]) if np.size(tempo) else 0.0
    return AudioFeatures(
        tempo=tempo_value,
        beat_times=librosa.frames_to_time(beat_frames, sr=sr),
        onset_times=librosa.frames_to_time(onset_frames, sr=sr),
        duration=float(librosa.get_duration(y=y, sr=sr)),
    )
