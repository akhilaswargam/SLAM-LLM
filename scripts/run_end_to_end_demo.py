#!/usr/bin/env python3
"""End-to-end Demonstration of the Complete SmartSLAM Pipeline:
raw speech
   ↓
noise estimation
   ↓
noise enhancement
   ↓
enhanced audio
   ↓
PyTorch tensor
   ↓
SmartSLAM adaptive router
   ↓
routing result
"""

import os
import sys
import json
import numpy as np
import scipy.signal
import soundfile as sf
import torch

sys.path.insert(0, os.path.abspath("src"))
from slam_llm.enhancement import NoiseEnhancer
from slam_llm.models.smartslam_router import SmartSLAMRouter


def main():
    print("=" * 75)
    print("SMARTSLAM END-TO-END PIPELINE DEMONSTRATION")
    print("=" * 75)

    # 1. Raw speech audio input
    wav_path = "examples/s2s/audio_prompt/en/prompt_1.wav"
    enhancer = NoiseEnhancer(snr_clean_threshold_db=20.0)
    clean_audio, sr = enhancer.load_audio(wav_path, target_sr=16000)

    # Inject realistic environmental background noise
    np.random.seed(42)
    t = np.linspace(0, len(clean_audio) / sr, len(clean_audio), endpoint=False)
    white = np.random.randn(len(clean_audio))
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    pink = scipy.signal.lfilter(b, a, white)
    env_noise = pink * (0.8 + 0.3 * np.sin(2 * np.pi * 1.5 * t))
    p_sig = np.mean(clean_audio ** 2)
    p_n = np.mean(env_noise ** 2)
    scaling = np.sqrt(p_sig / (p_n * (10 ** (10.0 / 10.0))))
    raw_speech = np.clip(clean_audio + scaling * env_noise, -1.0, 1.0).astype(np.float32)

    print("[Step 1] Raw Speech Input:")
    print(f"  Source file       : {wav_path}")
    print(f"  Duration          : {len(raw_speech) / sr:.3f} s")
    print(f"  Sampling rate     : {sr} Hz")
    print(f"  Data type / shape : {type(raw_speech).__name__}, {raw_speech.shape}")

    # 2. Noise estimation
    snr_est, n_pow, n_level, n_sig = enhancer.analyze_noise(raw_speech, sr)
    print("\n[Step 2] Noise Estimation:")
    print(f"  Noise detected    : {n_sig}")
    print(f"  Noise severity    : {n_level}")
    print(f"  Estimated SNR     : {snr_est:.2f} dB")
    print(f"  Noise floor power : {n_pow:.6e}")

    # 3. Noise enhancement
    print("\n[Step 3] Noise Enhancement:")
    enhanced_audio, meta = enhancer.enhance(raw_speech, sample_rate=sr)
    print(f"  Enhancement method: {meta.method}")
    print(f"  Enhancement applied: {meta.enhancement_applied}")
    print(f"  Processing latency: {meta.processing_time * 1000:.2f} ms")
    print(f"  SNR After         : {meta.estimated_snr_after:.2f} dB (Gain: +{meta.estimated_snr_after - meta.estimated_snr_before:.2f} dB)")

    # 4. Enhanced audio
    print("\n[Step 4] Enhanced Audio Array:")
    print(f"  Peak amplitude    : {np.max(np.abs(enhanced_audio)):.4f}")
    print(f"  RMS energy        : {np.sqrt(np.mean(enhanced_audio ** 2)):.4f}")
    print(f"  Finite / bounded  : np.isfinite = {np.all(np.isfinite(enhanced_audio))}")

    # 5. Conversion to PyTorch tensor
    print("\n[Step 5] PyTorch Tensor Conversion:")
    torch_tensor = torch.from_numpy(enhanced_audio).float().unsqueeze(0)
    print(f"  Tensor shape      : {tuple(torch_tensor.shape)}")
    print(f"  Tensor dtype      : {torch_tensor.dtype}")
    print(f"  Tensor device     : {torch_tensor.device}")

    # 6. SmartSLAM adaptive router
    print("\n[Step 6] SmartSLAM Adaptive Multimodal Router:")
    router = SmartSLAMRouter()
    routing_result = router.route(torch_tensor)

    # 7. Routing result
    print("\n[Step 7] Routing Decision & SLAM-LLM Execution Profile:")
    print(f"  Detected modality : {routing_result['modality']}")
    print(f"  Selected encoder  : {routing_result['encoder']}")
    print(f"  Confidence score  : {routing_result['confidence']:.4f}")
    print("  Routing reasoning :")
    for r in routing_result["reasoning"]:
        print(f"    - {r}")
    print("  SLAM-LLM Profile  :")
    for k, v in routing_result["slam_llm_profile"].items():
        print(f"    {k:26s}: {v}")

    print("=" * 75)

    os.makedirs("results", exist_ok=True)
    out_json = "results/end_to_end_demo_evidence.json"
    evidence_payload = {
        "raw_speech": {
            "source": wav_path,
            "duration_sec": round(len(raw_speech) / sr, 3),
            "sample_rate": sr,
        },
        "noise_estimation": {
            "noise_detected": bool(n_sig),
            "noise_level": n_level,
            "estimated_snr_db": round(snr_est, 2),
            "noise_power": float(n_pow),
        },
        "noise_enhancement": meta.to_dict(),
        "pytorch_tensor": {
            "shape": list(torch_tensor.shape),
            "dtype": str(torch_tensor.dtype),
            "device": str(torch_tensor.device),
        },
        "routing_result": routing_result,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(evidence_payload, f, indent=2)
    print(f"Demonstration log saved to: {out_json}\n")


if __name__ == "__main__":
    main()
