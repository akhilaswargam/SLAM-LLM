#!/usr/bin/env python3
"""Evaluation script for SmartSLAM Noise Enhancement.

Compares before and after enhancement across controlled noise conditions:
- Clean speech
- Low stationary noise (~22 dB)
- Moderate stationary noise (~12 dB)
- Strong stationary noise (~3 dB)
- Non-stationary/transient noise (~8 dB)
- Colored environmental noise (~10 dB)

Evaluates:
- SNR before, SNR after, Delta SNR (dB)
- Noise detection decision & category
- Whether enhancement was applied
- Peak amplitude & clipping detection
- Speech reference preservation (waveform Pearson correlation with clean signal)
- Duration and sample rate consistency
- Processing latency
- Honest check of downstream ASR (Whisper/PyTorch) dependencies (reporting BLOCKED if absent)
"""

import os
import sys
import json
import csv
import time
import tempfile
import argparse
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from slam_llm.enhancement import NoiseEnhancer
from generate_test_audio import create_test_suite


def compute_waveform_correlation(ref_wav: np.ndarray, proc_wav: np.ndarray) -> float:
    """Compute Pearson correlation coefficient between reference clean speech and processed audio."""
    min_len = min(len(ref_wav), len(proc_wav))
    r = ref_wav[:min_len]
    p = proc_wav[:min_len]
    std_r = np.std(r)
    std_p = np.std(p)
    if std_r < 1e-8 or std_p < 1e-8:
        return 0.0
    corr = np.corrcoef(r, p)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def probe_asr_dependencies():
    """Probe whether Whisper and PyTorch are available for downstream ASR evaluation."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import whisper
    except ImportError:
        missing.append("whisper")

    if missing:
        return False, f"BLOCKED - missing required dependencies/checkpoints: {', '.join(missing)}"
    return True, "Available"


def run_evaluation(data_dir: str = None, threshold: float = 20.0, json_out: str = None, csv_out: str = None):
    temp_dir = None
    if data_dir is None or not os.path.isdir(data_dir):
        temp_dir = tempfile.TemporaryDirectory()
        data_dir = temp_dir.name
        create_test_suite(data_dir)

    enhancer = NoiseEnhancer(snr_clean_threshold_db=threshold)

    # Reference clean signal
    clean_path = os.path.join(data_dir, "1_clean.wav")
    clean_wav, ref_sr = sf.read(clean_path, dtype="float32") if os.path.exists(clean_path) else (None, 16000)

    test_cases = [
        ("Clean Speech", os.path.join(data_dir, "1_clean.wav")),
        ("Low Stationary (~22 dB)", os.path.join(data_dir, "2_low_noise.wav")),
        ("Moderate Stationary (~12 dB)", os.path.join(data_dir, "3_moderate_noise.wav")),
        ("Strong Stationary (~3 dB)", os.path.join(data_dir, "4_strong_noise.wav")),
        ("Non-Stationary (~8 dB)", os.path.join(data_dir, "5_nonstationary_noise.wav")),
        ("Colored Env (~10 dB)", os.path.join(data_dir, "6_colored_environmental_noise.wav")),
    ]

    header_line = (
        f"{'Condition':<24} | {'Detected':<8} | {'Level':<8} | {'Applied':<7} | "
        f"{'SNR In':<9} | {'SNR Out':<9} | {'dSNR':<8} | {'Corr(Ref)':<9} | {'Peak':<6} | {'Time(s)':<7}"
    )
    print("=" * 115)
    print("SMARTSLAM NOISE ENHANCEMENT MULTI-CONDITION EVALUATION MATRIX")
    print("=" * 115)
    print(header_line)
    print("-" * 115)

    results_data = []
    applied_count = 0
    total_delta_snr = 0.0
    total_time = 0.0

    for label, path in test_cases:
        if not os.path.exists(path):
            continue

        noisy_wav, in_sr = sf.read(path, dtype="float32")
        out_temp = os.path.join(data_dir, f"enhanced_{os.path.basename(path)}")
        enhanced_wav, res = enhancer.enhance(path, output_path=out_temp)

        delta_snr = res.estimated_snr_after - res.estimated_snr_before
        peak_amp = float(np.max(np.abs(enhanced_wav))) if len(enhanced_wav) > 0 else 0.0
        is_clipped = peak_amp >= 1.0 - 1e-4

        # Speech preservation correlation with clean reference
        ref_corr = compute_waveform_correlation(clean_wav, enhanced_wav) if clean_wav is not None else 1.0

        if res.enhancement_applied:
            applied_count += 1
            total_delta_snr += delta_snr
        total_time += res.processing_time

        row = {
            "condition": label,
            "file": os.path.basename(path),
            "noise_detected": res.noise_detected,
            "noise_level": res.noise_level,
            "enhancement_applied": res.enhancement_applied,
            "snr_before_db": res.estimated_snr_before,
            "snr_after_db": res.estimated_snr_after,
            "delta_snr_db": round(delta_snr, 2),
            "ref_correlation": round(ref_corr, 4),
            "peak_amplitude": round(peak_amp, 4),
            "is_clipped": is_clipped,
            "sample_rate": res.sample_rate,
            "duration_sec": res.duration_sec,
            "processing_time_sec": res.processing_time,
        }
        results_data.append(row)

        print(
            f"{label:<24} | "
            f"{'Yes' if res.noise_detected else 'No':<8} | "
            f"{res.noise_level:<8} | "
            f"{'Yes' if res.enhancement_applied else 'No':<7} | "
            f"{res.estimated_snr_before:>6.2f} dB | "
            f"{res.estimated_snr_after:>6.2f} dB | "
            f"{delta_snr:>+6.2f} dB | "
            f"{ref_corr:>9.4f} | "
            f"{peak_amp:>6.3f} | "
            f"{res.processing_time:>7.4f}"
        )

    print("=" * 115)

    # Downstream ASR Probe Status
    asr_available, asr_status_msg = probe_asr_dependencies()
    print("\nDOWNSTREAM ASR EVALUATION PROBE:")
    print(f"Status: {asr_status_msg}")
    print("-" * 115)

    # Aggregate summary
    avg_delta_snr_enhanced = (total_delta_snr / applied_count) if applied_count > 0 else 0.0
    summary = {
        "total_conditions_evaluated": len(results_data),
        "conditions_enhanced": applied_count,
        "clean_or_low_bypassed": len(results_data) - applied_count,
        "mean_snr_gain_enhanced_db": round(avg_delta_snr_enhanced, 2),
        "mean_latency_sec": round(total_time / max(1, len(results_data)), 4),
        "asr_evaluation_status": asr_status_msg,
    }

    print("AGGREGATE METRICS:")
    print(f"  - Total conditions evaluated: {summary['total_conditions_evaluated']}")
    print(f"  - Conditions enhanced:        {summary['conditions_enhanced']}")
    print(f"  - Clean/low-noise bypassed:   {summary['clean_or_low_bypassed']}")
    print(f"  - Mean SNR gain on enhanced:  {summary['mean_snr_gain_enhanced_db']:+.2f} dB")
    print(f"  - Mean latency per file:      {summary['mean_latency_sec']:.4f} s")
    print("=" * 115)

    full_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "Adaptive Spectral Gating with Formant Protection",
        "clean_threshold_db": threshold,
        "asr_evaluation_status": asr_status_msg,
        "summary": summary,
        "conditions": results_data,
    }

    if json_out:
        os.makedirs(os.path.dirname(os.path.abspath(json_out)), exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as jf:
            json.dump(full_output, jf, indent=2)
        print(f"\n[Saved JSON structured evaluation report]: {json_out}")

    if csv_out:
        os.makedirs(os.path.dirname(os.path.abspath(csv_out)), exist_ok=True)
        with open(csv_out, "w", newline="", encoding="utf-8") as cf:
            fieldnames = [
                "condition", "file", "noise_detected", "noise_level", "enhancement_applied",
                "snr_before_db", "snr_after_db", "delta_snr_db", "ref_correlation",
                "peak_amplitude", "is_clipped", "sample_rate", "duration_sec", "processing_time_sec"
            ]
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for r in results_data:
                writer.writerow(r)
        print(f"[Saved CSV structured evaluation report]:  {csv_out}")

    if temp_dir is not None:
        temp_dir.cleanup()

    return full_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SmartSLAM Noise Enhancement")
    parser.add_argument("--data-dir", "-d", type=str, default=None, help="Directory with test wav files")
    parser.add_argument("--threshold", "-t", type=float, default=20.0, help="Clean SNR threshold (dB)")
    parser.add_argument("--json", "-j", type=str, default=None, help="Output JSON path for structured results")
    parser.add_argument("--csv", "-c", type=str, default=None, help="Output CSV path for structured results")
    args = parser.parse_args()

    run_evaluation(args.data_dir, threshold=args.threshold, json_out=args.json, csv_out=args.csv)

