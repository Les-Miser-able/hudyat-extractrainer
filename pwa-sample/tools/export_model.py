"""Export this trainer's Keras checkpoint for the sample's TensorFlow.js Layers model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fsl_trainer.training.trainer import build_model, tensorflow
from fsl_trainer.data.preprocessing import model_normalization, preprocess_sequence
from fsl_trainer.data.sequences import resample


def normalized(config):
    if isinstance(config, dict):
        return {k: normalized(v) for k,v in config.items() if k != "name"}
    if isinstance(config, (list, tuple)):
        return [normalized(v) for v in config]
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--hand-model", type=Path, default=ROOT/"models"/"hand_landmarker.task")
    parser.add_argument("--site", type=Path, default=ROOT/"pwa-sample"/"dist")
    args = parser.parse_args(argv)
    if not args.category or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in args.category):
        parser.error("Use a lowercase category ID containing letters, digits, hyphens or underscores.")
    classes = json.loads((args.run/"classes.json").read_text(encoding="utf-8"))
    preprocessing = json.loads((args.run/"preprocessing.json").read_text(encoding="utf-8"))
    normalization = model_normalization(preprocessing)
    extraction = preprocessing.get("extraction")
    if not extraction:
        parser.error("Browser camera inference needs raw-media extraction metadata from training.")
    hand_bytes = args.hand_model.read_bytes()
    hand_hash = hashlib.sha256(hand_bytes).hexdigest()
    if hand_hash != extraction["model_sha256"]:
        parser.error("The landmark .task asset does not match the model used for extraction.")
    tf = tensorflow()
    model = tf.keras.models.load_model(args.run/"best.keras", compile=False)
    expected = build_model(len(classes))
    if normalized(model.get_config()) != normalized(expected.get_config()):
        parser.error("Only the exact configured Conv1D + BiLSTM architecture is supported.")
    del expected
    chunks, specs, offset = [], [], 0
    for weight in model.get_weights():
        raw = np.asarray(weight, dtype="<f4").tobytes(order="C")
        specs.append({"shape":list(weight.shape), "dtype":"float32", "offset":offset, "bytes":len(raw)})
        chunks.append(raw)
        offset += len(raw)
    binary = b"".join(chunks)
    version = hashlib.sha256(binary+json.dumps([classes,preprocessing],sort_keys=True).encode()).hexdigest()[:16]
    dest = args.site/"models"/args.category/version
    dest.mkdir(parents=True,exist_ok=True)
    hand_dest = args.site/"shared"/f"{hand_hash}.task"
    hand_dest.parent.mkdir(parents=True,exist_ok=True)
    hand_dest.write_bytes(hand_bytes)
    metadata = {"format":"fsl-conv1d-bilstm-v1", "category":args.category, "version":version,
                "inputShape":[32,128], "classes":classes, "preprocessing":preprocessing,
                "weights":specs, "weightBytes":len(binary),
                "weightSha256":hashlib.sha256(binary).hexdigest()}
    (dest/"model.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    (dest/"weights.bin").write_bytes(binary)
    # Synthetic parity probes, never source images or user recordings.
    rng = np.random.default_rng(42)
    probes = np.zeros((3,32,128),dtype=np.float32)
    probes[0,:,:63] = rng.uniform(0.1,0.8,(1,63)); probes[0,:,126]=1
    probes[1,:,:126] = rng.uniform(-0.1,0.8,(32,126)); probes[1,:,126:]=1
    probes[2,:,63:126] = rng.uniform(0.1,0.8,(32,63)); probes[2,:,127]=1
    probabilities = model(probes,training=False).numpy()
    (dest/"parity.json").write_text(json.dumps({"inputs":probes.tolist(),"expected":probabilities.tolist()}),encoding="utf-8")
    camera_cases = []
    for sample, (width, height) in zip(probes, [(640, 480), (1280, 720), (480, 640)]):
        expected_input = resample(preprocess_sequence(sample, width, height, normalization))
        camera_cases.append({"frames": sample.tolist(), "width": width, "height": height,
                             "expected": expected_input.tolist()})
    (dest/"camera-preprocessing-parity.json").write_text(json.dumps(
        {"preprocessing": {"normalization": normalization}, "cases": camera_cases}), encoding="utf-8")
    catalog_path = args.site/"catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {"categories":[]}
    entries = {entry["id"]:entry for entry in catalog["categories"]}
    for folder in sorted((ROOT/"data").iterdir()):
        if folder.is_dir() and not folder.name.startswith("."):
            entries.setdefault(folder.name,{"id":folder.name,"name":folder.name.replace("_"," ").title(),"available":False})
    base = f"./models/{args.category}/{version}"
    entries[args.category] = {"id":args.category,"name":args.category.replace("_"," ").title(),
        "available":True, "version":version,"modelUrl":base+"/model.json","weightsUrl":base+"/weights.bin",
        "modelSha256":hashlib.sha256((dest/"model.json").read_bytes()).hexdigest(),
        "weightsSha256":metadata["weightSha256"],"bytes":len(binary)+(dest/"model.json").stat().st_size,
        "handUrl":f"./shared/{hand_hash}.task","handSha256":hand_hash,"handBytes":len(hand_bytes),
        "classes":len(classes)}
    catalog["categories"] = sorted(entries.values(),key=lambda e:e["name"])
    temporary = catalog_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(catalog,indent=2),encoding="utf-8")
    temporary.replace(catalog_path)
    print(f"Exported {args.category}: {len(classes)} classes, {len(binary)/1024/1024:.2f} MiB weights")
    print(f"Browser files: {dest}")


if __name__ == "__main__":
    main()
