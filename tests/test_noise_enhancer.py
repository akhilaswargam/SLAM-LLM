"""Unit tests for the Noise Enhancement module."""

import os
import tempfile
import numpy as np
import pytest
import soundfile as sf

from slam_llm.enhancement import NoiseEnhancer, enhance_audio


@pytest.fixture
def synthetic_audio_signals():
    """Generate sample clean, low, and strong noise audio vectors."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    # Speech-like tone harmonic signal
    clean = (0.6 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    # Add envelope
    envelope = (0.5 * (1.0 + np.sin(2 * np.pi * 3 * t))) ** 2
    clean = clean * envelope

    # Generate noisy signal
    np.random.seed(42)
    strong_noise = (clean + np.random.normal(0, 0.25, size=len(clean))).astype(np.float32)
    
    return {"sr": sr, "clean": clean, "noisy": strong_noise}


def test_clean_audio_handling(synthetic_audio_signals):
    """Test that clean audio is recognized as clean and enhancement is not forced."""
    enhancer = NoiseEnhancer(snr_clean_threshold_db=20.0)
    clean = synthetic_audio_signals["clean"]
    sr = synthetic_audio_signals["sr"]

    enhanced_wav, result = enhancer.enhance(clean, sample_rate=sr)

    assert result.noise_detected is False
    assert result.enhancement_applied is False
    assert result.noise_level in ["Clean", "Low"]
    assert np.allclose(enhanced_wav, clean, atol=1e-5)


def test_noisy_audio_detection_and_enhancement(synthetic_audio_signals):
    """Test that noisy audio triggers noise detection and enhancement."""
    enhancer = NoiseEnhancer(snr_clean_threshold_db=20.0)
    noisy = synthetic_audio_signals["noisy"]
    sr = synthetic_audio_signals["sr"]

    enhanced_wav, result = enhancer.enhance(noisy, sample_rate=sr)

    assert result.noise_detected is True
    assert result.enhancement_applied is True
    assert result.estimated_snr_after > result.estimated_snr_before
    assert len(enhanced_wav) == len(noisy)


def test_output_wav_file_validity(synthetic_audio_signals):
    """Test that saving output WAV results in valid, readable audio on disk."""
    noisy = synthetic_audio_signals["noisy"]
    sr = synthetic_audio_signals["sr"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_wav = os.path.join(tmp_dir, "input.wav")
        output_wav = os.path.join(tmp_dir, "output.wav")
        
        sf.write(input_wav, noisy, sr)
        assert os.path.exists(input_wav)

        _, result = enhance_audio(input_wav, output_path=output_wav)

        assert os.path.exists(output_wav)
        assert os.path.getsize(output_wav) > 0
        
        read_audio, read_sr = sf.read(output_wav)
        assert read_sr == 16000
        assert len(read_audio) > 0
        assert np.max(np.abs(read_audio)) <= 1.0


def test_metadata_structure(synthetic_audio_signals):
    """Test that result object conforms to required metadata specification."""
    noisy = synthetic_audio_signals["noisy"]
    sr = synthetic_audio_signals["sr"]

    _, result = enhance_audio(noisy, sample_rate=sr)
    res_dict = result.to_dict()

    expected_keys = [
        "input_path",
        "output_path",
        "noise_detected",
        "noise_level",
        "estimated_snr_before",
        "estimated_snr_after",
        "enhancement_applied",
        "method",
        "sample_rate",
        "processing_time",
        "duration_sec",
    ]
    for key in expected_keys:
        assert key in res_dict
    assert isinstance(result.processing_time, float)
    assert result.processing_time >= 0.0


def test_sample_rate_resampling(synthetic_audio_signals):
    """Test that audio recorded at different sample rate (e.g., 44.1 kHz) resamples correctly."""
    clean = synthetic_audio_signals["clean"]
    sr_orig = 44100
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_wav = os.path.join(tmp_dir, "sr44100.wav")
        sf.write(input_wav, clean, sr_orig)

        enhancer = NoiseEnhancer()
        loaded_wav, loaded_sr = enhancer.load_audio(input_wav, target_sr=16000)

        assert loaded_sr == 16000
        assert len(loaded_wav) > 0
