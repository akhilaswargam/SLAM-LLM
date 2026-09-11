#!/usr/bin/env python3
"""Run real speech WER and CER evaluation comparing raw vs enhanced audio."""

import os
import sys
import re
import json
import numpy as np
import scipy.signal
import soundfile as sf
import whisper
import torch
import torchaudio

sys.path.insert(0, os.path.abspath("src"))
from slam_llm.enhancement import NoiseEnhancer


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())


def levenshtein_distance(ref_tokens, hyp_tokens):
    r, h = ref_tokens, hyp_tokens
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    for i in range(len(r) + 1):
        d[i, 0] = i
    for j in range(len(h) + 1):
        d[0, j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = 1 + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return int(d[len(r), len(h)])


def compute_wer(ref: str, hyp: str) -> float:
    ref_clean = normalize_text(ref)
    hyp_clean = normalize_text(hyp)
    r_words = ref_clean.split()
    h_words = hyp_clean.split()
    if len(r_words) == 0:
        return 0.0 if len(h_words) == 0 else 1.0
    dist = levenshtein_distance(r_words, h_words)
    return round(float(dist) / len(r_words), 4)


def compute_cer(ref: str, hyp: str) -> float:
    ref_clean = normalize_text(ref).replace(" ", "")
    hyp_clean = normalize_text(hyp).replace(" ", "")
    r_chars = list(ref_clean)
    h_chars = list(hyp_clean)
    if len(r_chars) == 0:
        return 0.0 if len(h_chars) == 0 else 1.0
    dist = levenshtein_distance(r_chars, h_chars)
    return round(float(dist) / len(r_chars), 4)


def main():
    wav_path = "examples/s2s/audio_prompt/en/prompt_1.wav"
    enhancer = NoiseEnhancer(snr_clean_threshold_db=20.0)
    speech_wav, sr = enhancer.load_audio(wav_path, target_sr=16000)

    print("Loading Whisper Tiny model...")
    model = whisper.load_model("tiny")

    clean_res = model.transcribe(speech_wav, language="en")
    gt_text = clean_res["text"].strip()
    print(f"Verified Reference Ground-Truth transcript:\n  \"{gt_text}\"\n")

    np.random.seed(42)
    t = np.linspace(0, len(speech_wav) / 16000, len(speech_wav), endpoint=False)

    # Babble noise
    babble = np.zeros(len(speech_wav))
    for f, a in [(300, 0.4), (500, 0.3), (750, 0.35), (1200, 0.25), (1800, 0.15), (2400, 0.1)]:
        babble += a * np.sin(2 * np.pi * f * t + np.random.rand() * 2 * np.pi)
    white_babble = np.random.randn(len(speech_wav)) * 0.2
    b_bab, a_bab = [0.1, 0.2, 0.1], [1.0, -0.7, 0.2]
    babble += scipy.signal.lfilter(b_bab, a_bab, white_babble)
    babble *= (0.7 + 0.3 * np.sin(2 * np.pi * 2.0 * t))

    # HVAC / Pink noise
    white_pink = np.random.randn(len(speech_wav))
    b_p = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a_p = [1.0, -2.494956002, 2.017265875, -0.522189400]
    hvac_ambient = scipy.signal.lfilter(b_p, a_p, white_pink)

    # Street / Traffic noise
    traffic = hvac_ambient.copy()
    for loc in [int(0.6 * 16000), int(1.8 * 16000)]:
        traffic[loc:loc + 1200] += 0.8 * np.sin(2 * np.pi * 440.0 * np.linspace(0, 1200 / 16000, 1200))

    scenarios = [
        ("Clean Original Speech", np.zeros_like(speech_wav), 999.0),
        ("Moderate Ambient HVAC (~12 dB)", hvac_ambient, 12.0),
        ("Realistic Babble Cocktail (~8 dB)", babble, 8.0),
        ("Transient Traffic / Street (~5 dB)", traffic, 5.0),
        ("Severe Low SNR Noise (0 dB)", hvac_ambient, 0.0),
        ("Adverse Noise Condition (-2 dB)", hvac_ambient, -2.0),
    ]

    p_speech = np.mean(speech_wav ** 2)
    evaluation_records = []

    print("=" * 95)
    print("REAL SPEECH ASR (WHISPER TINY) WER / CER EVALUATION")
    print("=" * 95)

    for name, noise_raw, target_snr in scenarios:
        if target_snr > 100.0:
            noisy = speech_wav.copy()
        else:
            p_n = np.mean(noise_raw ** 2)
            scaling = np.sqrt(p_speech / (p_n * (10 ** (target_snr / 10.0))))
            noisy = np.clip(speech_wav + scaling * noise_raw, -1.0, 1.0).astype(np.float32)

        enhanced, meta = enhancer.enhance(noisy, sample_rate=16000)

        raw_hyp = model.transcribe(noisy, language="en")["text"].strip()
        enh_hyp = model.transcribe(enhanced, language="en")["text"].strip()

        raw_wer = compute_wer(gt_text, raw_hyp)
        raw_cer = compute_cer(gt_text, raw_hyp)
        enh_wer = compute_wer(gt_text, enh_hyp)
        enh_cer = compute_cer(gt_text, enh_hyp)

        snr_gain = meta.estimated_snr_after - meta.estimated_snr_before

        record = {
            "scenario": name,
            "target_snr_db": target_snr if target_snr < 100.0 else "clean",
            "noise_detected": meta.noise_detected,
            "noise_level": meta.noise_level,
            "enhancement_applied": meta.enhancement_applied,
            "snr_before_db": meta.estimated_snr_before,
            "snr_after_db": meta.estimated_snr_after,
            "snr_gain_db": round(snr_gain, 2),
            "ground_truth": gt_text,
            "raw_hypothesis": raw_hyp,
            "enhanced_hypothesis": enh_hyp,
            "raw_wer": raw_wer,
            "raw_cer": raw_cer,
            "enhanced_wer": enh_wer,
            "enhanced_cer": enh_cer,
            "wer_improvement": round(raw_wer - enh_wer, 4),
            "cer_improvement": round(raw_cer - enh_cer, 4),
        }
        evaluation_records.append(record)

        print(f"Condition: {name}")
        print(f"  SNR: In = {meta.estimated_snr_before:.2f} dB | Out = {meta.estimated_snr_after:.2f} dB | Gain = {snr_gain:+.2f} dB | Enhanced: {meta.enhancement_applied}")
        print(f"  Raw ASR:      \"{raw_hyp}\"")
        print(f"                WER = {raw_wer * 100:.1f}% | CER = {raw_cer * 100:.1f}%")
        print(f"  Enhanced ASR: \"{enh_hyp}\"")
        print(f"                WER = {enh_wer * 100:.1f}% | CER = {enh_cer * 100:.1f}%")
        delta_w = (raw_wer - enh_wer) * 100
        delta_c = (raw_cer - enh_cer) * 100
        print(f"  Improvement:  dWER = {delta_w:+.1f}% | dCER = {delta_c:+.1f}%\n")

    print("=" * 95)

    os.makedirs("results", exist_ok=True)
    out_file = "results/wer_cer_evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_records, f, indent=2)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
