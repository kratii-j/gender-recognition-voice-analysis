from __future__ import annotations

from functools import lru_cache
import math
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as audio_functional

from app.model import DEFAULT_MODEL_PATH, FEATURE_NAMES, GenderMLP


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        frames = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV files are supported.")

    signal = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        signal = signal.reshape(-1, channels).mean(axis=1)
    return signal, sample_rate


def zero_crossing_rate(signal: np.ndarray) -> float:
    if signal.size < 2:
        return 0.0
    signs = np.signbit(signal)
    return float(np.mean(signs[:-1] != signs[1:]))


def rms_energy(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal))))


def estimate_pitch(signal: np.ndarray, sample_rate: int) -> float:
    if signal.size < sample_rate // 4:
        return 0.0

    centered = signal - np.mean(signal)
    autocorr = np.correlate(centered, centered, mode="full")[centered.size - 1 :]
    min_lag = max(1, int(sample_rate / 255))
    max_lag = min(len(autocorr) - 1, int(sample_rate / 85))
    if max_lag <= min_lag:
        return 0.0

    window = autocorr[min_lag:max_lag]
    peak_index = int(np.argmax(window))
    lag = min_lag + peak_index
    return float(sample_rate / lag) if lag > 0 else 0.0


def frame_signal(signal: np.ndarray, frame_length: int = 1024, hop_length: int = 256) -> np.ndarray:
    if signal.size == 0:
        return np.zeros((1, frame_length), dtype=np.float32)
    if signal.size < frame_length:
        signal = np.pad(signal, (0, frame_length - signal.size))

    frame_count = 1 + (signal.size - frame_length) // hop_length
    trimmed = signal[: frame_length + (frame_count - 1) * hop_length]
    frames = np.lib.stride_tricks.sliding_window_view(trimmed, frame_length)[::hop_length]
    return np.array(frames, dtype=np.float32, copy=True)


def summarize_distribution(freqs_khz: np.ndarray, power: np.ndarray) -> dict[str, float]:
    eps = 1e-12
    weights = power / (power.sum(axis=1, keepdims=True) + eps)
    centroid = np.sum(weights * freqs_khz[None, :], axis=1)
    variance = np.sum(weights * np.square(freqs_khz[None, :] - centroid[:, None]), axis=1)
    spread = np.sqrt(np.maximum(variance, eps))

    cumulative = np.cumsum(weights, axis=1)

    def percentile(q: float) -> np.ndarray:
        indices = np.argmax(cumulative >= q, axis=1)
        return freqs_khz[indices]

    q25 = percentile(0.25)
    median = percentile(0.50)
    q75 = percentile(0.75)

    centered = freqs_khz[None, :] - centroid[:, None]
    skew = np.sum(weights * np.power(centered / spread[:, None], 3), axis=1)
    kurt = np.sum(weights * np.power(centered / spread[:, None], 4), axis=1)
    entropy = -np.sum(weights * np.log(weights + eps), axis=1) / math.log(weights.shape[1])
    flatness = np.exp(np.mean(np.log(power + eps), axis=1)) / (np.mean(power + eps, axis=1))
    mode = freqs_khz[np.argmax(power, axis=1)]

    return {
        "meanfreq": float(np.mean(centroid)),
        "sd": float(np.mean(spread)),
        "median": float(np.mean(median)),
        "Q25": float(np.mean(q25)),
        "Q75": float(np.mean(q75)),
        "IQR": float(np.mean(q75 - q25)),
        "skew": float(np.mean(skew)),
        "kurt": float(np.mean(kurt)),
        "sp.ent": float(np.mean(entropy)),
        "sfm": float(np.mean(flatness)),
        "mode": float(np.mean(mode)),
        "centroid": float(np.mean(centroid)),
    }


def extract_pitch_track(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    if signal.size < sample_rate // 4:
        return np.array([], dtype=np.float32)

    waveform = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    contour = audio_functional.detect_pitch_frequency(waveform, sample_rate).squeeze(0).numpy()
    valid = contour[(contour >= 50.0) & (contour <= 280.0)]
    if valid.size == 0:
        estimated = estimate_pitch(signal, sample_rate)
        if estimated <= 0:
            return np.array([], dtype=np.float32)
        valid = np.array([estimated], dtype=np.float32)
    return valid


def maybe_resample(signal: np.ndarray, sample_rate: int, target_rate: int = 16000) -> tuple[np.ndarray, int]:
    if sample_rate == target_rate:
        return signal, sample_rate

    waveform = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    resampled = audio_functional.resample(waveform, sample_rate, target_rate)
    return resampled.squeeze(0).numpy(), target_rate


def extract_deep_features(signal: np.ndarray, sample_rate: int) -> dict[str, float]:
    signal, sample_rate = maybe_resample(signal, sample_rate)
    frames = frame_signal(signal)
    window = np.hanning(frames.shape[1]).astype(np.float32)
    windowed = frames * window[None, :]
    spectrum = np.fft.rfft(windowed, axis=1)
    power = np.abs(spectrum) ** 2
    freqs_khz = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate) / 1000.0

    voice_band = freqs_khz <= 0.28
    spectral = summarize_distribution(freqs_khz[voice_band], power[:, voice_band])
    dominant = freqs_khz[np.argmax(power, axis=1)]
    pitch_track_hz = extract_pitch_track(signal, sample_rate)
    pitch_track_khz = pitch_track_hz / 1000.0 if pitch_track_hz.size else np.array([], dtype=np.float32)

    if pitch_track_khz.size:
        pitch_range = float(np.max(pitch_track_khz) - np.min(pitch_track_khz))
        modindx = (
            float(np.mean(np.abs(np.diff(pitch_track_khz))) / pitch_range)
            if pitch_track_khz.size > 1 and pitch_range > 0
            else 0.0
        )
        meanfun = float(np.mean(pitch_track_khz))
        minfun = float(np.min(pitch_track_khz))
        maxfun = float(np.max(pitch_track_khz))
    else:
        modindx = 0.0
        meanfun = 0.0
        minfun = 0.0
        maxfun = 0.0

    features = {
        **spectral,
        "meanfun": meanfun,
        "minfun": minfun,
        "maxfun": maxfun,
        "meandom": float(np.mean(dominant)),
        "mindom": float(np.min(dominant)),
        "maxdom": float(np.max(dominant)),
        "dfrange": float(np.max(dominant) - np.min(dominant)),
        "modindx": modindx,
    }

    return features


@lru_cache(maxsize=1)
def load_trained_bundle() -> tuple[GenderMLP, np.ndarray, np.ndarray, dict]:
    if not DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained model at {DEFAULT_MODEL_PATH}. Run train_model.py before starting inference."
        )

    bundle = torch.load(DEFAULT_MODEL_PATH, map_location="cpu", weights_only=True)
    model = GenderMLP(input_dim=bundle["input_dim"], hidden_dims=tuple(bundle["hidden_dims"]))
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    scaler_mean = np.asarray(bundle["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(bundle["scaler_scale"], dtype=np.float32)
    return model, scaler_mean, scaler_scale, bundle


def classify_voice_baseline(signal: np.ndarray, sample_rate: int) -> dict:
    duration_seconds = float(signal.size / sample_rate) if sample_rate else 0.0
    pitch_hz = estimate_pitch(signal, sample_rate)
    energy = rms_energy(signal)
    zcr = zero_crossing_rate(signal)

    threshold = 165.0
    if pitch_hz <= 0:
        label = "unknown"
        male_score = 0.5
        female_score = 0.5
    else:
        female_score = 1.0 / (1.0 + math.exp(-(pitch_hz - threshold) / 18.0))
        male_score = 1.0 - female_score
        label = "female" if female_score >= male_score else "male"

    confidence = max(male_score, female_score)
    male_percent = round(male_score * 100, 1)
    female_percent = round(female_score * 100, 1)

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "male_percent": male_percent,
        "female_percent": female_percent,
        "pitch_hz": round(pitch_hz, 2),
        "rms_energy": round(energy, 4),
        "zero_crossing_rate": round(zcr, 4),
        "duration_seconds": round(duration_seconds, 2),
        "model_type": "heuristic_baseline",
    }


def calibrate_live_probability(female_score: float, pitch_hz: float) -> tuple[float, float]:
    if pitch_hz <= 0:
        clipped = min(max(female_score, 0.1), 0.9)
        return 1.0 - clipped, clipped

    pitch_prior = 1.0 / (1.0 + math.exp(-(pitch_hz - 175.0) / 16.0))

    # Live audio is noisier than the tabular training set, so we blend the
    # neural net output with a pitch-based prior and avoid absolute 0/100 scores.
    blended_female = 0.4 * female_score + 0.6 * pitch_prior

    # Guardrails tuned for microphone input so borderline masculine voices
    # are not pushed into extreme female predictions.
    if pitch_hz < 145.0:
        blended_female = min(blended_female, 0.15)
    elif pitch_hz < 160.0:
        blended_female = min(blended_female, 0.30)
    elif pitch_hz < 170.0:
        blended_female = min(blended_female, 0.42)
    elif pitch_hz < 180.0:
        blended_female = min(max(blended_female, 0.38), 0.58)
    elif pitch_hz < 195.0:
        blended_female = min(max(blended_female, 0.45), 0.68)
    else:
        blended_female = min(max(blended_female, 0.55), 0.90)

    blended_female = min(max(blended_female, 0.10), 0.90)
    blended_male = 1.0 - blended_female
    return blended_male, blended_female


def pick_live_label(male_score: float, female_score: float, pitch_hz: float) -> str:
    margin = abs(female_score - male_score)
    if pitch_hz <= 0:
        return "unknown"

    # Treat the overlap region as uncertain instead of forcing a wrong binary label.
    if pitch_hz < 165.0:
        return "male"
    if pitch_hz >= 195.0:
        return "female"
    if 165.0 <= pitch_hz <= 185.0 and margin < 0.22:
        return "uncertain"
    if 185.0 < pitch_hz < 195.0 and margin < 0.18:
        return "uncertain"
    if margin < 0.12:
        return "uncertain"
    return "female" if female_score > male_score else "male"


def classify_voice(signal: np.ndarray, sample_rate: int) -> dict:
    duration_seconds = float(signal.size / sample_rate) if sample_rate else 0.0
    energy = rms_energy(signal)
    zcr = zero_crossing_rate(signal)

    try:
        features = extract_deep_features(signal, sample_rate)
        model, scaler_mean, scaler_scale, bundle = load_trained_bundle()

        values = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)
        scaled = (values - scaler_mean) / np.where(scaler_scale == 0, 1.0, scaler_scale)

        with torch.no_grad():
            logits = model(torch.tensor(scaled, dtype=torch.float32).unsqueeze(0))
            female_score = float(torch.sigmoid(logits).item())

        pitch_hz = float(features["meanfun"] * 1000.0)
        male_score, female_score = calibrate_live_probability(female_score, pitch_hz)
        label = pick_live_label(male_score, female_score, pitch_hz)
        confidence = max(male_score, female_score)

        return {
            "label": label,
            "confidence": round(confidence, 3),
            "male_percent": round(male_score * 100, 1),
            "female_percent": round(female_score * 100, 1),
            "pitch_hz": round(pitch_hz, 2),
            "rms_energy": round(energy, 4),
            "zero_crossing_rate": round(zcr, 4),
            "duration_seconds": round(duration_seconds, 2),
            "model_type": "deep_learning_mlp",
            "dataset_name": bundle.get("dataset_name", "voice.csv"),
            "test_accuracy": round(float(bundle.get("metrics", {}).get("test_accuracy", 0.0)), 4),
        }
    except Exception:
        return classify_voice_baseline(signal, sample_rate)
