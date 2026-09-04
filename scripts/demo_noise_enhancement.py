#!/usr/bin/env python3
"""Standalone Demo for SmartSLAM Noise Enhancement.

Usage:
    python scripts/demo_noise_enhancement.py --input <noisy_audio.wav> [--output <enhanced_audio.wav>]
"""

import sys
import os
import argparse

# Ensure src/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from slam_llm.enhancement import NoiseEnhancer


def parse_args():
    parser = argparse.ArgumentParser(description="SmartSLAM Noise Enhancement Demo")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input audio WAV")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save enhanced audio WAV")
    parser.add_argument("--threshold", "-t", type=float, default=20.0, help="Clean SNR threshold (dB)")
    parser.add_argument("--force", "-f", action="store_true", help="Force enhancement even if detected clean")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_enhanced{ext or '.wav'}"

    enhancer = NoiseEnhancer(snr_clean_threshold_db=args.threshold)
    _, result = enhancer.enhance(
        audio_or_path=args.input,
        output_path=output_path,
        sample_rate=16000,
        force_process=args.force,
    )

    print("========================================")
    print("SmartSLAM Noise Enhancement")
    print("========================================")
    print(f"Input: {result.input_path}")
    print(f"Noise detected: {'Yes' if result.noise_detected else 'No'}")
    print(f"Noise level: {result.noise_level}")
    print(f"Enhancement applied: {'Yes' if result.enhancement_applied else 'No'}")
    print(f"Method: {result.method}")
    print(f"Before SNR: {result.estimated_snr_before:.2f} dB")
    print(f"After SNR: {result.estimated_snr_after:.2f} dB")
    print(f"Output: {result.output_path}")
    print(f"Processing time: {result.processing_time:.4f} s")
    print("========================================")


if __name__ == "__main__":
    main()
