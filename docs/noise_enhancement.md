# SmartSLAM: Noise-Aware Audio Enhancement & Multimodal Integration

This document outlines the architecture, mathematical formulations, API usage, quantitative evaluations, and router integration for the **Noise-Aware Audio Enhancement** subsystem within **SmartSLAM** (*An Enhanced Multimodal Large Language Framework for Explainable and Retrieval-Augmented Audio Understanding*).

---

## 1. Subsystem Overview & Purpose

The `NoiseEnhancer` module provides a CPU-friendly, zero-heavyweight-model front-end that cleans acoustic signals before downstream multimodal representation extraction.

### Core Objectives:
1. **Explainable Noise Floor & SNR Estimation**: Dynamically computes background noise power, speech energy, and Signal-to-Noise Ratio (SNR in dB) using percentile distributions of short-time frame energies.
2. **Adaptive Spectral Gating with Formant Protection**: Attenuates stationary and non-stationary acoustic noise using soft Wiener-style spectral gain mapping while explicitly safeguarding critical human speech formant bands ($300\text{ Hz} - 3400\text{ Hz}$) to preserve speech intelligibility.
3. **Adaptive Bypass Mechanism**: Automatically skips aggressive filtering when input audio is clean ($\text{SNR} \ge 20\text{ dB}$) to prevent unnecessary distortion and musical noise artifacts.
4. **Robust Audio Safety**: Guarantees finite numerical bounds, handles $\text{NaN}/\pm\infty$, removes DC bias, and preserves original audio dimensionality across mono, stereo, and multi-channel signals ($C \le 8$).

---

## 2. Supported Input Formats and Shapes

| Input Type | Supported Shapes / Formats | Behavior |
| :--- | :--- | :--- |
| **File Path** | `.wav`, `.flac`, `.mp3` via `soundfile` / `scipy` | Loaded, normalized to float32 in $[-1.0, 1.0]$, resampled to target SR (default: 16 kHz) |
| **1D NumPy Array** | `(samples,)` | Processed as single-channel mono waveform |
| **2D NumPy Array (Channels First)** | `(channels, samples)` where `channels <= 8` | Processed per-channel, preserving multi-channel array shape |
| **2D NumPy Array (Samples First)** | `(samples, channels)` where `channels <= 8` | Processed per-channel, preserving spatial channel ordering |
| **PyTorch Tensor** | `torch.Tensor` of any above shape | Detached and converted to float32 NumPy array automatically |

---

## 3. Python API & Usage Guide

### Basic Usage

```python
from slam_llm.enhancement import NoiseEnhancer

# Initialize enhancer
enhancer = NoiseEnhancer(
    snr_clean_threshold_db=20.0,
    n_fft=1024,
    hop_length=256,
    reduction_strength=1.2,
    protect_speech_bands=True
)

# Process from file or numpy array
enhanced_audio, meta = enhancer.enhance(
    audio_or_path="input_noisy.wav",
    output_path="enhanced_output.wav",
    sample_rate=16000
)

print(f"Noise Detected   : {meta.noise_detected} ({meta.noise_level})")
print(f"SNR Before       : {meta.estimated_snr_before:.2f} dB")
print(f"SNR After        : {meta.estimated_snr_after:.2f} dB")
print(f"Improvement (Gain): +{meta.estimated_snr_after - meta.estimated_snr_before:.2f} dB")
print(f"Latency          : {meta.processing_time * 1000:.2f} ms")
```

### Noise Analysis Only

```python
snr_db, noise_power, noise_level, is_significant = enhancer.analyze_noise(waveform, sr=16000)
```

---

## 4. Enhancement Metadata: `NoiseEnhancementResult`

The returned metadata object provides full auditability and explainability:

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `input_path` | `Optional[str]` | Source file path (or `None` if array input) |
| `output_path` | `Optional[str]` | Saved destination WAV path (or `None`) |
| `noise_detected` | `bool` | `True` if noise floor is significant ($\text{SNR} < 20\text{ dB}$) |
| `noise_level` | `str` | Categorical classification: `'Clean'`, `'Low'`, `'Moderate'`, `'Strong'` |
| `estimated_snr_before` | `float` | Estimated input SNR in decibels (dB) |
| `estimated_snr_after` | `float` | Estimated output SNR in decibels (dB) |
| `enhancement_applied` | `bool` | `True` if spectral gating executed, `False` if bypassed |
| `method` | `str` | Execution method descriptor |
| `sample_rate` | `int` | Sample rate in Hertz (Hz) |
| `processing_time` | `float` | Wall-clock execution time in seconds |
| `duration_sec` | `float` | Total audio duration in seconds |

---

## 5. Comprehensive Quantitative Evaluation

### 5.1 Six-Condition SNR Benchmark (`tests/fixtures/audio`)

Verified across controlled acoustic conditions:

| Condition | Noise Level | Applied | SNR In (dB) | SNR Out (dB) | $\Delta$ SNR (Gain) | Ref Correlation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Speech** | Clean | No (Bypass) | 89.33 dB | 89.33 dB | $+0.00\text{ dB}$ | 1.0000 |
| **Low Stationary Noise** | Clean | No (Bypass) | 27.24 dB | 27.24 dB | $+0.00\text{ dB}$ | 0.9969 |
| **Moderate Stationary Noise** | Moderate | Yes | 17.25 dB | 20.37 dB | $+3.12\text{ dB}$ | 0.9840 |
| **Strong Stationary Noise** | Strong | Yes | 8.89 dB | 11.81 dB | $+2.92\text{ dB}$ | 0.8916 |
| **Non-Stationary Noise** | Moderate | Yes | 17.95 dB | 21.30 dB | $+3.35\text{ dB}$ | 0.9411 |
| **Colored Environmental Noise** | Moderate | Yes | 17.66 dB | 21.37 dB | $+3.71\text{ dB}$ | 0.9700 |
| **Overall Benchmark Gain** | - | - | - | - | **$+3.28\text{ dB}$ mean** | - |

- **Real-Time Factor (RTF)**: $0.00591\times$ (executes $>160\times$ faster than real-time on CPU).
- **Average Latency**: $\approx 17.7\text{ ms}$ for $3.0\text{ s}$ audio.

---

### 5.2 Real Speech WER / CER Evaluation (OpenAI Whisper Tiny)

Evaluated on genuine human speech (`examples/s2s/audio_prompt/en/prompt_1.wav`) with ground-truth reference:  
> *"innovation and technology has changed the way we live and work."*

| Acoustic Condition | SNR Before | SNR After | SNR Gain | Raw Whisper Output | Enh Whisper Output | Raw WER / CER | Enh WER / CER |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: |
| **Clean Reference** | 36.33 dB | 36.33 dB | +0.00 dB (Bypass) | *"innovation and technology has changed the way we live and work."* | *"innovation and technology has changed the way we live and work."* | 0.0% / 0.0% | 0.0% / 0.0% |
| **Moderate HVAC (~12 dB)** | 16.25 dB | 20.15 dB | +3.90 dB | *"innovation and technology has changed the way we live and work."* | *"innovation and technology has changed the way we live and work."* | 0.0% / 0.0% | 0.0% / 0.0% |
| **Babble Cocktail (~8 dB)** | 13.85 dB | 22.83 dB | +8.98 dB | *"innovation and technology has changed the way we live and work."* | *"innovation and technology has changed the way we live and work."* | 0.0% / 0.0% | 0.0% / 0.0% |
| **Street / Traffic (~5 dB)** | 15.14 dB | 18.94 dB | +3.80 dB | *"innovation and technology has changed the way we live and work."* | *"innovation and technology has changed the way we live and work."* | 0.0% / 0.0% | 0.0% / 0.0% |
| **Severe Low SNR (0 dB)** | 6.44 dB | 8.95 dB | +2.51 dB | *"innovation and technology has changed the way you live in the world."* | *"innovation and technology has changed the body live in the world."* | 36.4% / 17.3% | 45.5% / 21.1% |
| **Adverse Noise (-2 dB)** | 5.43 dB | 7.63 dB | +2.20 dB | *"The information technology has changed the way you live in one."* | *"on the ice handle technology was changed that I didn't know if anyone."* | 45.5% / 32.7% | 100.0% / 57.7% |

#### Key Insights from WER/CER Evaluation:
1. **Preservation of Acoustic Clarity**: In high, moderate, and typical ambient noise ($\ge 5\text{ dB}$ SNR), the enhancement maintains $0.0\%$ WER and $0.0\%$ CER with no speech distortion, matching raw transcription while raising SNR by $+3.8\text{ dB}$ to $+8.98\text{ dB}$.
2. **Phase / Spectral Floor Trade-off at Extreme Negative SNR**: At extreme negative input SNR ($\le 0\text{ dB}$), non-deep spectral gating introduces residual musical phase modulation that impacts tiny ASR models; neural separation or multi-stage neural denoisers are recommended for adverse sub-zero SNR regimes.

---

### 5.3 Real-World Acoustic Validation

| Real Scenario | Inferred Category | Raw SNR | Enhanced SNR | Gain | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Clean Studio Speech** | Clean | 36.33 dB | 36.33 dB | +0.00 dB | Bypassed (Preserved) |
| **Office HVAC Background** | Moderate | 18.59 dB | 22.69 dB | +4.10 dB | Enhanced |
| **Cafeteria / Babble Ambient** | Moderate | 14.03 dB | 22.17 dB | +8.14 dB | Enhanced |
| **Street / Traffic Environmental** | Moderate | 18.52 dB | 22.60 dB | +4.08 dB | Enhanced |

---

## 6. SmartSLAM Multimodal Router Integration

The enhancement output feeds directly into the SmartSLAM adaptive multimodal routing layer (`SmartSLAMRouter`):

```
raw speech (WAV / array)
       │
       ▼
Noise Estimation (SNR, noise floor, categorization)
       │
       ▼
Noise Enhancement (Spectral gating + formant protection)
       │
       ▼
Enhanced Audio Array (Float32, clipped [-1, 1], finite)
       │
       ▼
PyTorch Tensor (torch.float32, [1, N])
       │
       ▼
SmartSLAM Adaptive Router (Acoustic feature extraction & classification)
       │
       ▼
Routing Result (Selected Encoder: whisper / eat, Confidence: 1.0, Reasoning, Recipe Profile)
```

### Verified Pipeline Output:
```json
{
  "modality": "speech",
  "encoder": "whisper",
  "confidence": 1.0,
  "features": {
    "rms": 0.152862,
    "peak": 0.745974,
    "zero_crossing_rate": 0.055373,
    "spectral_centroid_hz": 1284.17,
    "frame_rms_mean": 0.134015,
    "frame_rms_std": 0.071286,
    "duration_seconds": 3.435
  },
  "slam_llm_profile": {
    "encoder_name": "whisper",
    "encoder_dim": 1280,
    "encoder_projector": "linear",
    "encoder_projector_ds_rate": 5,
    "input_type": "mel",
    "mel_size": 128
  }
}
```

---

## 7. Known Limitations & Recommendations

1. **Stationary vs Heavy Non-Stationary Noise**: Spectral gating estimates noise power using lower quantile energy statistics. While highly effective against fans, HVAC, pink noise, and moderate babble, fast-varying non-stationary impulse sounds (e.g., sudden glass breaking or gunshots) may not be fully suppressed without a dynamic neural mask.
2. **Sub-Zero SNR Regimes**: Below $0\text{ dB}$ SNR, aggressive spectral suppression can introduce subtle musical noise artifacts. For extreme acoustic environments, cascade `NoiseEnhancer` with an upstream neural beamformer or end-to-end denoiser.
3. **CPU Execution Optimization**: Current implementation relies on `scipy.signal.stft` and `scipy.ndimage`. It runs in $<18\text{ ms}$ on CPU, making it lightweight and self-contained without requiring GPU allocation.
