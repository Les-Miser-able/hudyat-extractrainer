"""Source-group-disjoint dataset splits."""
import numpy as np

def split_sources(records, val_fraction, test_fraction, seed):
    """Group-disjoint splits. Groups can represent source images, sessions or signers."""
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1
            and val_fraction + test_fraction < 1):
        raise ValueError("validation and test fractions must be positive and sum to <1")
    labels = sorted({r["label"] for r in records})
    groups = sorted({r["group"] for r in records})
    for label in labels:
        if len({r["group"] for r in records if r["label"] == label}) < 3:
            raise ValueError(f"{label}: need at least 3 independent source groups "
                             "for train/validation/test; synthetic variants do not count")
    rng = np.random.default_rng(seed)
    # Class-local groups admit a direct stratified assignment.
    memberships = {g: {r["label"] for r in records if r["group"] == g} for g in groups}
    assignment = {}
    if all(len(v) == 1 for v in memberships.values()):
        for label in labels:
            local = np.array([g for g in groups if label in memberships[g]], dtype=object)
            rng.shuffle(local)
            nv = max(1, round(len(local) * val_fraction))
            nt = max(1, round(len(local) * test_fraction))
            while nv + nt >= len(local):
                if nv >= nt and nv > 1:
                    nv -= 1
                elif nt > 1:
                    nt -= 1
                else:
                    raise ValueError("Cannot allocate three nonempty splits")
            for i, group in enumerate(local):
                assignment[group] = "validation" if i < nv else "test" if i < nv + nt else "train"
        return assignment
    # Shared signer/session IDs must remain together across every class.
    matrix = np.array([[sum(r["group"] == g and r["label"] == label for r in records)
                        for label in labels] for g in groups])
    fractions = np.array([1 - val_fraction - test_fraction, val_fraction, test_fraction])
    best, best_score = None, float("inf")
    for _ in range(3000):
        choices = rng.choice(3, size=len(groups), p=fractions)
        counts = np.array([matrix[choices == k].sum(axis=0) for k in range(3)])
        if (counts == 0).any():
            continue
        score = np.mean((counts / matrix.sum(axis=0) - fractions[:, None]) ** 2)
        if score < best_score:
            best, best_score = choices.copy(), score
    if best is None:
        raise ValueError("Could not find group-disjoint splits containing every class. "
                         "Add independent groups or revise group IDs/fractions.")
    return {g: ["train", "validation", "test"][int(k)] for g, k in zip(groups, best)}

