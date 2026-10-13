"""Inference with the saved training preprocessing contract."""
import hashlib
import json
from pathlib import Path
from fsl_trainer.models.network import tensorflow
from fsl_trainer.preparation.sequences import load_sequence, resample

def predict(args):
    tf = tensorflow()
    root = Path(args.run)
    classes = json.loads((root / "classes.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(root / "best.keras")
    if Path(args.sample).suffix.lower() == ".npy":
        x = resample(load_sequence(args.sample))
    else:
        from fsl_trainer.preparation.extraction import Extractor, DEFAULT_MODEL
        metadata = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))
        settings = metadata.get("extraction")
        if not settings:
            raise ValueError("This model has no raw-media extraction settings; supply normalized .npy input")
        with Extractor(model=getattr(args, "hand_model", None) or DEFAULT_MODEL,
                       confidence=settings["detection_confidence"],
                       swap_hands=settings["swap_hands"]) as extractor:
            if hashlib.sha256(extractor.model.read_bytes()).hexdigest() != settings["model_sha256"]:
                raise ValueError("Hand-landmark model differs from training; use --hand-model with the original asset")
            sequence, _ = extractor.extract(args.sample)
            x = resample(sequence)
    probabilities = model.predict(x[None], verbose=0)[0]
    print(json.dumps({label: float(probabilities[i]) for i, label in
                      sorted(enumerate(classes), key=lambda pair: -probabilities[pair[0]])}, indent=2))

