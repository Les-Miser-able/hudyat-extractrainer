"""Deterministic timing variation of complete multi-frame landmark sequences."""
import hashlib
import numpy as np
from .sequences import resample
from .augmentation_options import extra_samples, time_warp

VERSION = 1
MAX_ATTEMPTS_PER_SAMPLE = 10


def configuration(count, strength, seed):
    count, strength = extra_samples(count), time_warp(strength)
    if not count:
        return None
    return {"version": VERSION, "algorithm": "piecewise_linear_timing",
            "eligible_inputs": "videos_and_multi_frame_landmarks", "split": "train",
            "extra_samples": count, "max_displacement": strength, "seed": seed,
            "max_attempts_per_sample": MAX_ATTEMPTS_PER_SAMPLE}


def _sample_fingerprint(sample):
    normalized = sample.copy()
    normalized[normalized == 0] = 0  # +0 and -0 are the same coordinates.
    return hashlib.sha256(normalized.tobytes()).digest()


def augment_sequence(sequence, source, seed, count, strength):
    samples = []
    seen = {_sample_fingerprint(resample(sequence))}
    anchors = np.array([0., 1/3, 2/3, 1.])
    detail = {"source": source, "requested_extra_samples": count,
              "generated_extra_samples": 0, "unfilled_extra_samples": 0,
              "duplicate_candidates": 0, "invalid_candidates": 0,
              "shortfall_reasons": []}
    for candidate in range(count * MAX_ATTEMPTS_PER_SAMPLE):
        if len(samples) == count:
            break
        identity = f"{seed}:{source}:{candidate}".encode("utf-8")
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(identity).digest(), "big"))
        mapped = anchors.copy()
        mapped[1:3] += rng.uniform(-strength, strength, 2)
        timeline = np.interp(np.linspace(0, 1, 32), anchors, mapped) * (len(sequence) - 1)
        sample = resample(sequence, timeline)
        if not np.isfinite(sample).all() or not sample[:, 126:].any():
            detail["invalid_candidates"] += 1
            continue
        fingerprint = _sample_fingerprint(sample)
        if fingerprint in seen:
            detail["duplicate_candidates"] += 1
            continue
        seen.add(fingerprint)
        samples.append((sample, {"kind": "temporal_warp", "version": VERSION,
                                 "candidate": candidate, "anchors": mapped.tolist()}))
    detail["generated_extra_samples"] = len(samples)
    detail["unfilled_extra_samples"] = count - len(samples)
    if detail["unfilled_extra_samples"]:
        for field, reason in (("duplicate_candidates", "duplicate samples"),
                              ("invalid_candidates", "invalid samples")):
            if detail[field]:
                detail["shortfall_reasons"].append(reason)
    return samples, detail


def build_report(settings, sources, original_samples):
    totals = {field: sum(source[field] for source in sources) for field in
              ("requested_extra_samples", "generated_extra_samples", "unfilled_extra_samples",
               "duplicate_candidates", "invalid_candidates")}
    return {"version": VERSION, "settings": settings, "original_sequence_samples": original_samples,
            **totals, "sources": sources}
