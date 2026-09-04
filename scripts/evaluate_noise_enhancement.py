#!/usr/bin/env python3
"""Evaluation script for SmartSLAM Noise Enhancement.

Compares before and after enhancement across controlled noise conditions:
- Clean speech
- Low noise
- Moderate noise
- Strong noise

Reports:
- input condition
- noise detected
- enhancement applied
- SNR before
- SNR after
- processing time
"""

import os
import sys
import tempfile
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from slam_llm.enhancement import NoiseEnhancer
from generate_test_audio import create_test_suite


def run_evaluation(data_dir: str = None, threshold: float = 20.0):
    temp_dir = None
    if data_dir is None or not os.path.isdir(data_dir):
        temp_dir = tempfile.TemporaryDirectory()
        data_dir = temp_dir.name
        create_test_suite(data_dir)

    enhancer = NoiseEnhancer(snr_clean_threshold_db=threshold)

    test_cases = [
        ("Clean Speech", os.path.join(data_dir, "1_clean.wav")),
        ("Low Noise (~22 dB)", os.path.join(data_dir, "2_low_noise.wav")),
        ("Moderate Noise (~12 dB)", os.path.join(data_dir, "3_moderate_noise.wav")),
        ("Strong Noise (~3 dB)", os.path.join(data_dir, "4_strong_noise.wav")),
    ]

    print("=" * 88)
    print(f"{'Condition':<22} | {'Noise?':<7} | {'Level':<9} | {'Applied?':<9} | {'SNR Before':<11} | {'SNR After':<11} | {'Time (s)':<8}")
    print("-" * 88)

    results = []
    for label, path in test_cases:
        if not os.path.exists(path):
            continue
        out_temp = os.path.join(data_dir, f"enhanced_{os.path.basename(path)}")
        _, res = enhancer.enhance(path, output_path=out_temp)
        results.append((label, res))

        print(
            f"{label:<22} | "
            f"{'Yes' if res.noise_detected else 'No':<7} | "
            f"{res.noise_level:<9} | "
            f"{'Yes' if res.enhancement_applied else 'No':<9} | "
            f"{res.estimated_snr_before:>7.2f} dB | "
            f"{res.estimated_snr_after:>7.2f} dB | "
            f"{res.processing_time:>8.4f}"
        )

    print("=" * 88)

    if temp_dir is not None:
        temp_dir.cleanup()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Noise Enhancement")
    parser.add_argument("--data-dir", "-d", type=str, default=None, help="Directory with test wav files")
    parser.add_argument("--threshold", "-t", type=float, default=20.0, help="Clean SNR threshold (dB)")
    args = parser.parse_args()
    run_evaluation(args.data_dir, threshold=args.threshold)
