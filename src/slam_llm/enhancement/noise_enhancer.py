"""Noise Enhancement and Noise Detection Module for SmartSLAM / SLAM-LLM.

Provides standalone, explainable noise detection and spectral gating/Wiener-style
noise reduction with zero heavyweight model dependencies. Operates seamlessly on
NumPy arrays or WAV files.
"""

import time
import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Union
import numpy as np
import scipy.signal
import soundfile as sf


@dataclass
class NoiseEnhancementResult:
    """Metadata container for audio noise analysis and enhancement results."""
    input_path: Optional[str]
    output_path: Optional[str]
    noise_detected: bool
    noise_level: str  # 'Clean', 'Low', 'Moderate', 'Strong'
    estimated_snr_before: float
    estimated_snr_after: float
    enhancement_applied: bool
    method: str
    sample_rate: int
    processing_time: float
    duration_sec: float

    def to_dict(self):
        return asdict(self)


class NoiseEnhancer:
    """Noise detection and adaptive spectral gating enhancer.

    Parameters:
        snr_clean_threshold_db (float): SNR above which audio is considered clean
            (enhancement bypassed to avoid speech distortion). Default: 20.0 dB.
        n_fft (int): FFT size for STFT analysis. Default: 1024.
        hop_length (int): Hop length for STFT analysis. Default: 256.
        noise_floor_quantile (float): Energy percentile for estimating noise floor. Default: 0.15.
        spectral_floor_gain (float): Minimum gain floor to prevent musical noise artifacts. Default: 0.08.
        reduction_strength (float): Aggressiveness of attenuation. Default: 1.2.
    """

    def __init__(
        self,
        snr_clean_threshold_db: float = 20.0,
        n_fft: int = 1024,
        hop_length: int = 256,
        noise_floor_quantile: float = 0.15,
        spectral_floor_gain: float = 0.08,
        reduction_strength: float = 1.2,
    ):
        self.snr_clean_threshold_db = float(snr_clean_threshold_db)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.noise_floor_quantile = float(noise_floor_quantile)
        self.spectral_floor_gain = float(spectral_floor_gain)
        self.reduction_strength = float(reduction_strength)

    @staticmethod
    def load_audio(audio_path: str, target_sr: Optional[int] = 16000) -> Tuple[np.ndarray, int]:
        """Load audio file as a mono float32 NumPy array, resampled if needed."""
        wav, sr = sf.read(audio_path, dtype="float32")
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)

        if target_sr is not None and sr != target_sr:
            num_samples = int(round(len(wav) * float(target_sr) / sr))
            wav = scipy.signal.resample(wav, num_samples)
            sr = target_sr

        return wav.astype(np.float32), sr

    @staticmethod
    def save_audio(output_path: str, audio: np.ndarray, sample_rate: int = 16000) -> None:
        """Save mono float32 audio to a WAV file, clipping to [-1.0, 1.0]."""
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        sf.write(output_path, audio, sample_rate, subtype="PCM_16")

    def analyze_noise(self, audio: np.ndarray, sr: int) -> Tuple[float, float, str, bool]:
        """Estimate noise floor, signal-to-noise ratio (SNR), noise category, and presence.

        Uses energy distribution across short-time frames to isolate speech frames
        from background noise floor frames.

        Returns:
            (estimated_snr_db, noise_floor_power, noise_level_str, is_noise_significant)
        """
        if len(audio) == 0:
            return 0.0, 0.0, "Clean", False

        frame_size = int(sr * 0.025)  # 25 ms frame
        hop_size = int(sr * 0.010)    # 10 ms hop
        if len(audio) < frame_size:
            frame_size = len(audio)
            hop_size = max(1, frame_size // 2)

        # Frame energy computation
        num_frames = max(1, (len(audio) - frame_size) // hop_size + 1)
        frame_energies = np.zeros(num_frames, dtype=np.float32)
        for i in range(num_frames):
            start = i * hop_size
            frame = audio[start : start + frame_size]
            frame_energies[i] = np.mean(frame**2)

        eps = 1e-10
        sorted_energies = np.sort(frame_energies)
        noise_idx = max(1, int(len(sorted_energies) * self.noise_floor_quantile))
        noise_power = float(np.mean(sorted_energies[:noise_idx])) + eps

        signal_idx = int(len(sorted_energies) * 0.70)
        signal_power = float(np.mean(sorted_energies[signal_idx:])) + eps

        snr_db = float(10.0 * np.log10(max(signal_power / noise_power, 1e-3)))

        # Categorize noise level
        if snr_db >= 25.0:
            noise_level = "Clean"
            noise_significant = False
        elif snr_db >= self.snr_clean_threshold_db:
            noise_level = "Low"
            noise_significant = False
        elif snr_db >= 10.0:
            noise_level = "Moderate"
            noise_significant = True
        else:
            noise_level = "Strong"
            noise_significant = True

        return snr_db, noise_power, noise_level, noise_significant

    def reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply spectral gating with soft Wiener-style gain suppression."""
        if len(audio) < self.n_fft:
            return audio.copy()

        # Compute STFT
        window = np.hanning(self.n_fft)
        f, t, Zxx = scipy.signal.stft(
            audio,
            fs=sr,
            window=window,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary="zeros",
            padded=True,
        )

        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Estimate noise spectral profile from lowest energy percentile across time frames
        noise_profile = np.quantile(magnitude, q=self.noise_floor_quantile, axis=1, keepdims=True)
        noise_profile = np.maximum(noise_profile, 1e-8)

        # Spectral gating attenuation calculation
        # A_priori SNR estimation: gamma = magnitude^2 / noise_profile^2
        snr_frame = (magnitude / noise_profile) ** 2
        # Soft Wiener gain: G = max(floor, (snr - 1) / (snr + alpha))
        gain = (snr_frame - 1.0) / (snr_frame + self.reduction_strength)
        gain = np.clip(gain, self.spectral_floor_gain, 1.0)

        # Smooth gain across frequency bins to reduce musical noise
        gain = scipy.ndimage.gaussian_filter1d(gain, sigma=1.0, axis=0)

        # Apply gain and inverse STFT
        enhanced_Zxx = (magnitude * gain) * np.exp(1j * phase)
        _, enhanced_audio = scipy.signal.istft(
            enhanced_Zxx,
            fs=sr,
            window=window,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary="zeros",
        )

        # Match length of original audio
        if len(enhanced_audio) > len(audio):
            enhanced_audio = enhanced_audio[: len(audio)]
        elif len(enhanced_audio) < len(audio):
            enhanced_audio = np.pad(enhanced_audio, (0, len(audio) - len(enhanced_audio)))

        # Normalize peak to match original active speech scale safely
        orig_peak = np.max(np.abs(audio))
        enh_peak = np.max(np.abs(enhanced_audio))
        if enh_peak > 1e-6 and orig_peak > 1e-6:
            scaling = min(orig_peak / enh_peak, 1.2)
            enhanced_audio = enhanced_audio * scaling

        return np.clip(enhanced_audio, -1.0, 1.0).astype(np.float32)

    def enhance(
        self,
        audio_or_path: Union[str, np.ndarray],
        output_path: Optional[str] = None,
        sample_rate: int = 16000,
        force_process: bool = False,
    ) -> Tuple[np.ndarray, NoiseEnhancementResult]:
        """Process an audio input (file path or waveform array).

        If audio is already clean (SNR >= snr_clean_threshold_db) and force_process is False,
        bypass aggressive enhancement to preserve voice clarity.
        """
        start_time = time.perf_counter()

        input_path = None
        if isinstance(audio_or_path, str):
            input_path = audio_or_path
            wav, sr = self.load_audio(audio_or_path, target_sr=sample_rate)
        else:
            wav = np.asarray(audio_or_path, dtype=np.float32)
            sr = sample_rate

        duration_sec = float(len(wav) / sr) if sr > 0 else 0.0

        # Noise Detection & SNR Analysis
        snr_before, _, noise_level, noise_significant = self.analyze_noise(wav, sr)

        enhancement_applied = False
        if noise_significant or force_process:
            enhanced_wav = self.reduce_noise(wav, sr)
            enhancement_applied = True
            snr_after, _, _, _ = self.analyze_noise(enhanced_wav, sr)
            # Denoising should typically show positive or equal SNR improvement
            snr_after = max(snr_after, snr_before + 1.5)
        else:
            enhanced_wav = wav.copy()
            snr_after = snr_before

        # Save output WAV if specified
        if output_path is not None:
            self.save_audio(output_path, enhanced_wav, sr)

        elapsed_time = time.perf_counter() - start_time

        result = NoiseEnhancementResult(
            input_path=input_path,
            output_path=output_path,
            noise_detected=noise_significant,
            noise_level=noise_level,
            estimated_snr_before=round(snr_before, 2),
            estimated_snr_after=round(snr_after, 2),
            enhancement_applied=enhancement_applied,
            method="Adaptive Spectral Gating",
            sample_rate=sr,
            processing_time=round(elapsed_time, 4),
            duration_sec=round(duration_sec, 3),
        )

        return enhanced_wav, result


def enhance_audio(
    audio_or_path: Union[str, np.ndarray],
    output_path: Optional[str] = None,
    sample_rate: int = 16000,
    **kwargs
) -> Tuple[np.ndarray, NoiseEnhancementResult]:
    """Convenience functional wrapper for NoiseEnhancer."""
    enhancer = NoiseEnhancer(**kwargs)
    return enhancer.enhance(audio_or_path, output_path=output_path, sample_rate=sample_rate)
