"""Local UI for the FSL trainer. Start with: python dashboard.py"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path
import secrets
import subprocess
import sys
import threading
from urllib.parse import parse_qs, unquote, urlparse

from ..paths import PROJECT_ROOT

BASE = PROJECT_ROOT
UI = Path(__file__).resolve().parent / "ui"
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
             ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".npy"}


def local_path(value, root=None):
    """Resolve only repository-local paths, including symlink targets."""
    root = BASE if root is None else root
    if root.resolve() != BASE.resolve() and BASE.resolve() not in root.resolve().parents:
        raise ValueError("Paths must stay inside the project folder.")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Enter a project-relative path.")
    path = (root / value).resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise ValueError("Paths must stay inside the project folder.")
    return path


def folder_name(value):
    if (not isinstance(value, str) or not value.strip() or value in {".", ".."}
            or value.startswith(".") or value.endswith((".", " "))
            or any(c in value for c in '/\\:*?"<>|')
            or any(ord(c) < 32 for c in value)):
        raise ValueError("Use a single folder name, without path separators.")
    return value


def number(data, key, default, minimum, maximum, integer=False):
    raw = data.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(f"Invalid {key}.")
    try:
        value = float(raw)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid {key}.") from None
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}.")
    if integer and not value.is_integer():
        raise ValueError(f"{key} must be a whole number.")
    return int(value) if integer else value


def read_json(path, fallback=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def directories(root):
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
                  and BASE in p.resolve().parents)


def source_classes(source):
    classes = []
    for folder in directories(source):
        count = sum(1 for p in folder.rglob("*") if p.is_file()
                    and p.suffix.lower() in SUPPORTED
                    and not any(part.startswith(".") for part in p.relative_to(folder).parts)
                    and BASE in p.resolve().parents)
        classes.append({"name": folder.name, "samples": count})
    return classes


def preparation_issue(source, classes):
    location = source.relative_to(BASE).as_posix()
    if not classes:
        return (f"No gesture class folders found in {location}. Add at least two, "
                f"for example {location}/A/ and {location}/B/, with at least three "
                "independent source recordings or images per class. Then scan folders again.")
    if len(classes) < 2:
        return (f"{location} has only one gesture class folder. Add a second class; "
                "each class needs at least three independent sources.")
    short = [f"{c['name']} ({c['samples']})" for c in classes if c["samples"] < 3]
    if short:
        return ("At least three source files per gesture are needed before splitting. "
                f"Add images, videos, or .npy files to: {', '.join(short)}. "
                "Use independent recordings or images, not copies of the same source.")
    return None


def inventory(data="data", datasets="datasets_augmented"):
    sources, prepared = local_path(data), local_path(datasets)
    names = sorted({p.name for p in directories(sources)} | {p.name for p in directories(prepared)})
    rows = []
    for name in names:
        classes = source_classes(sources / name)
        target = prepared / name
        rows.append({"name": name, "classes": classes,
                     "samples": sum(c["samples"] for c in classes),
                     "preparation_issue": preparation_issue(sources / name, classes),
                     "prepared": (target / "manifest.json").is_file(),
                     "summary": read_json(target / "augmentation_summary.json", {})})
    return rows


def runs():
    rows = []
    for category in directories(BASE / "runs"):
        for run in directories(category):
            if not (run / "classes.json").is_file():
                continue
            metrics = read_json(run / "test_metrics.json", {})
            classes = read_json(run / "classes.json", [])
            rows.append({"id": run.relative_to(BASE).as_posix(), "name": run.name,
                         "category": category.name, "classes": len(classes),
                         "accuracy": metrics.get("accuracy"), "f1": metrics.get("macro_f1"),
                         "complete": (run / "test_metrics.json").is_file(),
                         "checkpoint": (run / "best.keras").is_file(),
                         "modified": run.stat().st_mtime})
    return sorted(rows, key=lambda r: r["modified"], reverse=True)


def build_command(data):
    action = data.get("action")
    command = [sys.executable, "-u", "-B", str(BASE / "trainer.py")]
    if action in {"prepare", "train", "inspect"}:
        category = folder_name(data.get("category"))
        datasets = local_path(data.get("datasets", "datasets_augmented"))
        dataset = local_path(category, datasets)
    if action == "prepare":
        source = local_path(category, local_path(data.get("data", "data")))
        if not source.is_dir():
            raise ValueError("This category has no source folder.")
        if source == dataset or source in dataset.parents or dataset in source.parents:
            raise ValueError("Source and dataset folders must be separate and non-nested.")
        if dataset.exists() and (not dataset.is_dir() or any(dataset.iterdir())):
            raise ValueError("Dataset output already exists. Choose a new dataset root.")
        validation = number(data, "validation", .15, .001, .99)
        test = number(data, "test", .15, .001, .99)
        if validation + test >= 1:
            raise ValueError("Validation and test fractions must sum to less than 1.")
        issue = preparation_issue(source, source_classes(source))
        if issue:
            raise ValueError(issue)
        command += ["prepare", "--source", str(source), "--output", str(dataset),
                    "--validation", str(validation), "--test", str(test),
                    "--static-variants", str(number(data, "variants", 8, 1, 1000, True)),
                    "--jitter", str(number(data, "jitter", .0075, 0, 1)),
                    "--detection-confidence", str(number(data, "confidence", .5, 0, 1))]
        if data.get("swap_hands") is True:
            command += ["--swap-hands"]
        if data.get("groups"):
            groups = local_path(data["groups"])
            if not groups.is_file():
                raise ValueError("Group CSV not found.")
            command += ["--groups", str(groups)]
    elif action in {"train", "inspect"}:
        if not (dataset / "manifest.json").is_file():
            raise ValueError("Prepare this category in the selected dataset root first.")
        command += [action, "--dataset", str(dataset)]
        if action == "train":
            output = local_path(folder_name(data.get("run")), local_path(category, BASE / "runs"))
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise ValueError("Run already exists. Choose a new run name.")
            command += ["--output", str(output),
                        "--epochs", str(number(data, "epochs", 100, 1, 10000, True)),
                        "--batch-size", str(number(data, "batch_size", 32, 1, 65536, True)),
                        "--learning-rate", str(number(data, "learning_rate", .001, 1e-10, 1)),
                        "--patience", str(number(data, "patience", 12, 1, 10000, True))]
            if data.get("class_weights") is True:
                command += ["--class-weights"]
    elif action == "predict":
        run = local_path(data.get("run"), BASE)
        if BASE / "runs" not in run.parents or not (run / "best.keras").is_file():
            raise ValueError("Select a run with a trained checkpoint.")
        sample = local_path(data.get("sample"))
        if not sample.is_file() or sample.suffix.lower() not in SUPPORTED:
            raise ValueError("Choose an existing project-local image, video, or .npy sample.")
        command += ["predict", "--run", str(run), "--sample", str(sample)]
    else:
        raise ValueError("Unknown job action.")
    if action in {"prepare", "train"}:
        command += ["--seed", str(number(data, "seed", 42, 0, 2**32 - 1, True))]
    if action in {"prepare", "predict"} and data.get("hand_model"):
        model = local_path(data["hand_model"])
        if not model.is_file() or model.suffix != ".task":
            raise ValueError("Choose an existing .task hand model.")
        command += ["--hand-model", str(model)]
    return command


class Jobs:
    """One worker at a time; only explicit UI requests start expensive work."""
    def __init__(self):
        self.lock = threading.Lock()
        self.items = deque(maxlen=30)
        self.process = None

    def start(self, data):
        with self.lock:
            if self.process and self.process.poll() is None:
                raise ValueError("A job is already running. Wait or stop it first.")
            command = build_command(data)
            process = subprocess.Popen(command, cwd=BASE, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                       errors="replace", bufsize=1,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            job = {"id": secrets.token_hex(8), "action": data["action"],
                   "category": data.get("category", "prediction"), "status": "running",
                   "started": datetime.now().isoformat(timespec="seconds"),
                   "command": subprocess.list2cmdline(command), "exit_code": None,
                   "logs": deque(maxlen=1800)}
            self.process = process
            self.items.appendleft(job)
            threading.Thread(target=self.collect, args=(process, job), daemon=True).start()
            return job["id"]

    def collect(self, process, job):
        try:
            for line in process.stdout:
                with self.lock:
                    job["logs"].append(line.rstrip("\r\n"))
        finally:
            process.stdout.close()
            code = process.wait()
            with self.lock:
                job["exit_code"] = code
                if job["status"] != "cancelled":
                    job["status"] = "completed" if code == 0 else "failed"

    def cancel(self, job_id):
        with self.lock:
            if (not self.items or self.items[0]["id"] != job_id or not self.process
                    or self.process.poll() is not None):
                raise ValueError("This job is no longer running.")
            self.process.terminate()
            self.items[0]["status"] = "cancelled"
            self.items[0]["logs"].append("Stopped by user. Partial outputs were kept; use a new output name to retry.")

    def snapshot(self):
        with self.lock:
            return [{**item, "logs": list(item["logs"])} for item in self.items]

    def shutdown(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()


class Handler(BaseHTTPRequestHandler):
    def reply(self, value, status=200):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def valid_host(self):
        return self.headers.get("Host") in {
            f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

    def do_GET(self):
        try:
            if not self.valid_host():
                return self.reply({"error": "Invalid host."}, 403)
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path == "/api/state":
                return self.reply({"categories": inventory(query.get("data", ["data"])[0],
                                                          query.get("datasets", ["datasets_augmented"])[0]),
                                   "runs": runs(), "token": self.server.token,
                                   "python": sys.version.split()[0]})
            if url.path == "/api/jobs":
                return self.reply(self.server.jobs.snapshot())
            if url.path == "/api/run":
                run_id = query.get("id", [""])[0]
                if run_id not in {r["id"] for r in runs()}:
                    raise ValueError("Run not found.")
                root = local_path(run_id)
                return self.reply({"metrics": read_json(root / "test_metrics.json", {}),
                                   "history": read_json(root / "history.json", {}),
                                   "classes": read_json(root / "classes.json", [])})
            path = local_path(unquote(url.path).lstrip("/") or "index.html", UI)
            if not path.is_file():
                return self.reply({"error": "Not found."}, 404)
            payload = path.read_bytes()
            self.send_response(200)
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_header("Content-Type", mime + ("; charset=utf-8" if mime.startswith("text/") else ""))
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)
        except (ValueError, OSError) as error:
            self.reply({"error": str(error)}, 400)

    def do_POST(self):
        try:
            if not self.valid_host() or not secrets.compare_digest(
                    self.headers.get("X-Dashboard-Token", ""), self.server.token):
                return self.reply({"error": "Refresh the dashboard before submitting."}, 403)
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://localhost:{self.server.server_port}",
                                         f"http://127.0.0.1:{self.server.server_port}"}:
                return self.reply({"error": "Invalid origin."}, 403)
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 16384:
                raise ValueError("Invalid request size.")
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object.")
            if self.path == "/api/jobs":
                return self.reply({"id": self.server.jobs.start(data)}, 201)
            if self.path == "/api/cancel":
                self.server.jobs.cancel(data.get("id"))
                return self.reply({"ok": True})
            self.reply({"error": "Not found."}, 404)
        except (ValueError, OSError) as error:
            self.reply({"error": str(error)}, 400)

    def log_message(self, format, *args):
        if args and str(args[1]) not in {"200", "201"}:
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.token = secrets.token_hex(32)
    server.jobs = Jobs()
    print(f"FSL Studio: http://localhost:{server.server_port}", flush=True)
    print(f"Worker Python: {sys.executable}\nPress Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.jobs.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
