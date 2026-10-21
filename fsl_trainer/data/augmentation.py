"""Deterministic timing and optional opposite-hand copies of training sequences."""
import hashlib
import numpy as np
from .sequences import resample
from .augmentation_options import extra_samples, time_warp, mirror_hands as validate_mirroring
from .preprocessing import IMAGE, preparation_normalization, mirror_sequence

VERSION = 1
MAX_ATTEMPTS_PER_SAMPLE = 10


def configuration(count, strength, seed, *, mirror_hands=False, normalization=IMAGE):
    count, strength = extra_samples(count), time_warp(strength)
    mirror_hands = validate_mirroring(mirror_hands)
    if not count and not mirror_hands:
        return None
    settings = {"version": VERSION, "algorithm": "piecewise_linear_timing",
            "eligible_inputs": "videos_and_multi_frame_landmarks", "split": "train",
            "extra_samples": count, "max_displacement": strength, "seed": seed,
            "max_attempts_per_sample": MAX_ATTEMPTS_PER_SAMPLE}
    if mirror_hands:
        settings.update(version=2, mirror_hands=True, mirror_normalization=preparation_normalization(normalization),
                        algorithm='piecewise_linear_timing_with_hand_mirroring' if count else 'hand_mirroring')
    return settings


def _sample_fingerprint(sample):
    normalized = sample.copy()
    normalized[normalized == 0] = 0  # +0 and -0 are the same coordinates.
    return hashlib.sha256(normalized.tobytes()).digest()


def augment_sequence(sequence, source, seed, count, strength, *, mirror_hands=False, normalization=IMAGE):
    settings = configuration(count, strength, seed, mirror_hands=mirror_hands, normalization=normalization)
    count, strength = extra_samples(count), time_warp(strength)
    samples = []
    original = resample(sequence)
    seen = {_sample_fingerprint(original)}
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
    if settings and settings.get('mirror_hands'):
        timing_count = len(samples)
        parents = [(original, None), *samples]
        detail.update(requested_timing_samples=count, generated_timing_samples=timing_count,
                      unfilled_timing_samples=count - timing_count, requested_mirrored_samples=count + 1,
                      generated_mirrored_samples=0, mirrored_duplicate_candidates=0)
        for variant, (sample, _) in enumerate(parents):
            mirrored = mirror_sequence(sample, settings['mirror_normalization'])
            fingerprint = _sample_fingerprint(mirrored)
            if fingerprint in seen:
                detail['duplicate_candidates'] += 1
                detail['mirrored_duplicate_candidates'] += 1
                continue
            seen.add(fingerprint)
            samples.append((mirrored, {'kind': 'hand_mirror', 'version': 1,
                                      'normalization': settings['mirror_normalization'],
                                      'mirror_of_variant': variant}))
            detail['generated_mirrored_samples'] += 1
        detail['unfilled_mirrored_samples'] = count + 1 - detail['generated_mirrored_samples']
        detail['requested_extra_samples'] = count * 2 + 1
    detail["generated_extra_samples"] = len(samples)
    detail["unfilled_extra_samples"] = detail['requested_extra_samples'] - len(samples)
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
    if settings.get('mirror_hands'):
        totals.update({field: sum(source[field] for source in sources) for field in
                       ('requested_timing_samples', 'generated_timing_samples', 'unfilled_timing_samples',
                        'requested_mirrored_samples', 'generated_mirrored_samples', 'unfilled_mirrored_samples',
                        'mirrored_duplicate_candidates')})
    return {"version": settings['version'], "settings": settings, "original_sequence_samples": original_samples,
            **totals, "sources": sources}
