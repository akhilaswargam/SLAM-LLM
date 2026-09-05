#!/usr/bin/env python3
"""Synthetic audio generator for controlled noise evaluation in SmartSLAM.

Generates:
1. Clean speech simulation (harmonic speech-like formants with natural amplitude envelope)
2. Low noise condition (clean speech + Gaussian / pink noise at ~22 dB SNR)
3. Moderate noise condition (clean speech + noise at ~12 dB SNR)
4. Strong noise condition (clean speech + noise at ~3 dB SNR)
"""

import os
import sys
import numpy as np
import soundfile as sf


def generate_speech_like_signal(duration: float = 3.0, sr: int = 16000) -> np.ndarray:
    """Synthesize clean speech-like harmonic signal with natural pauses and modulation."""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    
    # Fundamental frequency pitch glide typical of human speech (120 Hz to 220 Hz)
    f0 = 150.0 + 30.0 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr

    # Harmonics (voice formants F1, F2, F3)
    signal = (
        0.50 * np.sin(phase) +
        0.30 * np.sin(2 * phase) +
        0.20 * np.sin(3 * phase) +
        0.15 * np.sin(4 * phase) +
        0.10 * np.sin(5 * phase)
    )

    # Apply speech-like syllable modulation envelope (2.5 Hz syllable rate) with pauses
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 2.5 * t)) ** 2
    # Add silent pauses at the beginning and end
    pause_samples = int(0.3 * sr)
    envelope[:pause_samples] = 0.0
    envelope[-pause_samples:] = 0.0

    speech = signal * envelope
    # Normalize speech signal
    speech = speech / (np.max(np.abs(speech)) + 1e-8) * 0.7
    return speech.astype(np.float32)


def add_noise(clean: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Add Gaussian noise at target SNR (dB)."""
    clean_power = np.mean(clean**2)
    noise_power_target = clean_power / (10.0 ** (target_snr_db / 10.0))
    
    noise = np.random.normal(0, np.sqrt(noise_power_target), size=len(clean))
    noisy = clean + noise
    return noisy.astype(np.float32)


def generate_colored_noise(num_samples: int, color: str = "pink") -> np.ndarray:
    """Generate colored noise (pink / brown / low-frequency hum) using spectral filtering."""
    white = np.random.normal(0, 1.0, size=num_samples)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(num_samples)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0  # avoid zero division

    if color == "pink":
        # 1/f power spectrum -> 1/sqrt(f) amplitude
        filter_curve = 1.0 / np.sqrt(freqs)
    elif color == "brown":
        # 1/f^2 power spectrum -> 1/f amplitude
        filter_curve = 1.0 / freqs
    else:
        filter_curve = np.ones_like(freqs)

    filter_curve = filter_curve / np.max(filter_curve)
    colored_X = X * filter_curve
    colored = np.fft.irfft(colored_X, n=num_samples)
    colored = colored / (np.std(colored) + 1e-8)
    return colored.astype(np.float32)


def add_nonstationary_transient_noise(clean: np.ndarray, sr: int = 16000, target_snr_db: float = 8.0) -> np.ndarray:
    """Add non-stationary transient noise bursts (door slams, clicks, intermittent background clatter)."""
    num_samples = len(clean)
    clean_power = np.mean(clean**2)
    noise_power_target = clean_power / (10.0 ** (target_snr_db / 10.0))

    base_noise = np.random.normal(0, np.sqrt(noise_power_target * 0.3), size=num_samples)

    # Generate 3-4 transient bursts across the audio duration
    burst_mask = np.zeros(num_samples, dtype=np.float32)
    burst_centers = [int(num_samples * 0.25), int(num_samples * 0.55), int(num_samples * 0.80)]
    burst_width = int(sr * 0.08)  # 80 ms burst
    for bc in burst_centers:
        start = max(0, bc - burst_width // 2)
        end = min(num_samples, bc + burst_width // 2)
        window = np.hanning(end - start)
        burst_mask[start:end] += window * 3.0

    transient_noise = np.random.normal(0, np.sqrt(noise_power_target * 2.0), size=num_samples) * burst_mask
    total_noise = base_noise + transient_noise
    # Scale total noise to exact target SNR
    curr_noise_power = np.mean(total_noise**2) + 1e-10
    total_noise = total_noise * np.sqrt(noise_power_target / curr_noise_power)

    noisy = clean + total_noise
    return noisy.astype(np.float32)


def add_colored_environmental_noise(clean: np.ndarray, target_snr_db: float = 10.0) -> np.ndarray:
    """Add colored environmental background noise (pink/ambient ventilation noise)."""
    clean_power = np.mean(clean**2)
    noise_power_target = clean_power / (10.0 ** (target_snr_db / 10.0))

    pink_noise = generate_colored_noise(len(clean), color="pink")
    curr_power = np.mean(pink_noise**2) + 1e-10
    pink_noise = pink_noise * np.sqrt(noise_power_target / curr_power)

    noisy = clean + pink_noise
    return noisy.astype(np.float32)


def create_test_suite(output_dir: str, sr: int = 16000):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    clean = generate_speech_like_signal(duration=3.0, sr=sr)
    low_noise = add_noise(clean, target_snr_db=22.0)
    mod_noise = add_noise(clean, target_snr_db=12.0)
    str_noise = add_noise(clean, target_snr_db=3.0)
    nonstat_noise = add_nonstationary_transient_noise(clean, sr=sr, target_snr_db=8.0)
    colored_noise = add_colored_environmental_noise(clean, target_snr_db=10.0)

    files = {
        "1_clean.wav": clean,
        "2_low_noise.wav": low_noise,
        "3_moderate_noise.wav": mod_noise,
        "4_strong_noise.wav": str_noise,
        "5_nonstationary_noise.wav": nonstat_noise,
        "6_colored_environmental_noise.wav": colored_noise,
    }

    created = {}
    for fname, wav in files.items():
        path = os.path.join(output_dir, fname)
        sf.write(path, np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")
        created[fname] = path

    return created


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/audio"
    created = create_test_suite(out_dir)
    print(f"Generated test audio files in: {out_dir}")
    for name, p in created.items():
        print(f"  - {name} -> {p}")

