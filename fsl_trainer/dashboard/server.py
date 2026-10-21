"""Local UI for the FSL trainer. Start with: python dashboard.py"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import io
import math
import mimetypes
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
from urllib.parse import parse_qs, unquote, urlparse

from ..reporting.benchmark import read_benchmark, benchmark_csv
from ..paths import PROJECT_ROOT
from ..data.augmentation_options import extra_samples, time_warp, mirror_hands
from ..data.source_selection import SUPPORTED, VIDEO_EXTENSIONS, source_files, folder_selection
from ..data.dataset_paths import dataset_name, prepared_datasets, select_dataset
from ..data.signer_splits import options as split_options, signer_from_source, split_signers
from ..data.preprocessing import IMAGE, HAND_RELATIVE, preparation_normalization
from ..data.source_selection import IMAGE_EXTENSIONS
from .camera_timing import camera_session, save_camera_timing
from .live import LiveInference
from .timing import capture_timing
from .model_records import run_metadata, read_records, save_record
from .prediction_tests import read_tests, save_test, export_model, append_trial
from .prediction_trials import trials_csv, metrics_csv
from .model_exports import native_archive, saved_run, export_options, matching_hand_model, download_name

BASE = PROJECT_ROOT
UI = Path(__file__).resolve().parent / "ui"


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


def source_classes(source, source_folders=None, videos_only=False):
    classes = []
    for folder in directories(source):
        files = [p for p in source_files(folder, source_folders, videos_only)
                 if BASE in p.resolve().parents]
        subfolders = {}
        for subfolder in directories(folder):
            members = [p for p in files if p.relative_to(folder).parts[0] == subfolder.name
                       and len(p.relative_to(folder).parts) > 1]
            subfolders[subfolder.name] = {"samples": len(members),
                                         "videos": sum(p.suffix.lower() in VIDEO_EXTENSIONS for p in members)}
        classes.append({"name": folder.name, "samples": len(files),
                        "videos": sum(p.suffix.lower() in VIDEO_EXTENSIONS for p in files),
                        "folders": subfolders})
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
        versions = [{"name": p.name if p != target else "Original dataset",
                     "path": p.relative_to(BASE).as_posix(), "legacy": p == target,
                     "evaluation_protocol": read_json(p / 'preprocessing.json', {}).get('evaluation_protocol'),
                     "summary": read_json(p / "augmentation_summary.json", {})}
                    for p in prepared_datasets(target)
                    if p.resolve() == BASE or BASE in p.resolve().parents]
        rows.append({"name": name, "classes": classes,
                     "samples": sum(c["samples"] for c in classes),
                     "preparation_issue": preparation_issue(sources / name, classes),
                     "prepared": bool(versions), "datasets": versions,
                     "summary": versions[0]["summary"] if len(versions) == 1 else {}})
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
                         "category": category.name, "classes": len(classes), "class_labels": classes,
                         "metadata": run_metadata(run),
                         "benchmark": read_benchmark(run, metrics, classes, category.name),
                         "exports": export_options(run),
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
        category_root = local_path(category, datasets)
    if action == "prepare":
        strategy, test_signer, validation_signer = split_options(
            data.get('split_strategy', 'grouped'), data.get('test_signer'), data.get('validation_signer'))
        dataset = local_path(dataset_name(data.get("dataset_name")), category_root)
        selected = folder_selection(data.get("source_folders"))
        videos_only = data.get("videos_only", False)
        if not isinstance(videos_only, bool):
            raise ValueError("videos_only must be a boolean.")
        video_count = extra_samples(data.get("video_extra_samples", 0))
        video_strength = time_warp(data.get("video_time_warp", .05))
        video_mirroring = mirror_hands(data.get('video_mirror_hands', False))
        source = local_path(category, local_path(data.get("data", "data")))
        if not source.is_dir():
            raise ValueError("This category has no source folder.")
        if source == dataset or source in dataset.parents or dataset in source.parents:
            raise ValueError("Source and dataset folders must be separate and non-nested.")
        if dataset.exists() and (not dataset.is_dir() or any(dataset.iterdir())):
            raise ValueError("Dataset output already exists. Choose a new dataset name.")
        validation = number(data, "validation", .15, .001, .99)
        test = number(data, "test", .15, .001, .99)
        if strategy == 'grouped' and validation + test >= 1:
            raise ValueError("Validation and test fractions must sum to less than 1.")
        classes = source_classes(source, selected, videos_only)
        if selected is not None or videos_only:
            classes = [row for row in classes if row["samples"]]
        issue = preparation_issue(source, classes)
        if issue:
            raise ValueError(issue)
        if strategy == 'loso':
            files = [p for folder in source.iterdir() if folder.is_dir() and not folder.name.startswith('.')
                     for p in source_files(folder, selected, videos_only)]
            split_signers([{'source': p.relative_to(source).as_posix(), 'label': p.relative_to(source).parts[0],
                            'group': p.relative_to(source).as_posix(),
                            'signer': signer_from_source(p.relative_to(source).as_posix())} for p in files],
                          test_signer, validation_signer)
        has_media = any(p.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
                        for folder in source.iterdir() if folder.is_dir()
                        for p in source_files(folder, selected, videos_only))
        normalization = preparation_normalization(data.get("normalization", "auto"), category, has_media)
        jitter_data = data if data.get("jitter") not in (None, "") else {k: v for k, v in data.items() if k != "jitter"}
        command += ["prepare", "--source", str(source), "--output", str(dataset),
                    "--validation", str(validation), "--test", str(test),
                    "--static-variants", str(number(data, "variants", 8, 1, 1000, True)),
                    "--jitter", str(number(jitter_data, "jitter", .03 if normalization == HAND_RELATIVE else .0075, 0, 1)),
                    "--detection-confidence", str(number(data, "confidence", .5, 0, 1))]
        if normalization != IMAGE:
            command += ["--normalization", normalization]
        if strategy == 'loso':
            command += ['--split-strategy', 'loso', '--test-signer', test_signer]
            if validation_signer:
                command += ['--validation-signer', validation_signer]
        if video_count:
            command += ["--video-extra-samples", str(video_count),
                        "--video-time-warp", str(video_strength)]
        if video_mirroring:
            command.append('--video-mirror-hands')
        for name in selected or []:
            command += ["--source-folder", name]
        if videos_only:
            command += ["--videos-only"]
        if data.get("swap_hands") is True:
            command += ["--swap-hands"]
        if data.get("groups"):
            groups = local_path(data["groups"])
            if not groups.is_file():
                raise ValueError("Group CSV not found.")
            command += ["--groups", str(groups)]
    elif action in {"train", "inspect"}:
        if "dataset" in data:
            dataset = local_path(data["dataset"])
            if dataset not in {p.resolve() for p in prepared_datasets(category_root)}:
                raise ValueError("Choose a prepared dataset from the selected category and dataset root.")
        else:
            dataset = select_dataset(category_root, data.get("dataset_name"))
        if dataset is None:
            raise ValueError("Prepare a named dataset for this category first.")
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
    elif action == "hudyat-timing":
        run, row = saved_run(BASE, data.get("run"), runs())
        if not export_options(run)["hudyat"]:
            raise ValueError(export_options(run)["reason"])
        project = data.get("hudyat_project") or str(BASE.parent / "HUDYAT")
        if not isinstance(project, str) or len(project) > 2000:
            raise ValueError("Enter the HUDYAT project folder path.")
        project = (BASE / project).resolve()
        session = data.get("_job_id") or secrets.token_hex(8)
        if not isinstance(session, str) or len(session) != 16 or any(c not in "0123456789abcdef" for c in session):
            raise ValueError("Invalid camera session identifier.")
        return [sys.executable, "-u", "-B", str(BASE / "tools/prepare_hudyat_timing.py"),
                "--run", row["id"], "--project", str(project), "--session", session]
    elif action == "benchmark":
        run_id = data.get("run")
        if run_id not in {row["id"] for row in runs()}:
            raise ValueError("Select an existing saved model for benchmarking.")
        run = local_path(run_id)
        if not all((run / name).is_file() for name in ("best.keras", "test_metrics.json",
                   "training_config.json", "split_manifest.json")):
            raise ValueError("Benchmarking needs a checkpoint, test metrics, training config, and split manifest.")
        command += ["benchmark", "--run", str(run)]
    elif action == "paper-report":
        command += ["paper-report"]
        if data.get("run"):
            if data["run"] not in {row["id"] for row in runs()}:
                raise ValueError("Choose an existing model run for its paper report.")
            command += ["--run", str(local_path(data["run"]))]
        elif data.get("dataset"):
            from ..reporting.paper import saved_datasets
            dataset = local_path(data["dataset"])
            dataset_root = local_path(data.get("datasets", "datasets_augmented"))
            if dataset not in {p.resolve() for p in saved_datasets(dataset_root)}:
                raise ValueError("Choose an existing prepared dataset for its paper report.")
            command += ["--dataset", str(dataset)]
        else:
            command += ["--all", "--datasets", str(local_path(data.get("datasets", "datasets_augmented")))]
    elif action == "export":
        run, row = saved_run(BASE, data.get("run"), runs())
        options = export_options(run)
        if not options['hudyat']:
            raise ValueError(options['reason'])
        detector = matching_hand_model(BASE, run, data.get('hand_model'))
        export_id = data.get('_job_id') or secrets.token_hex(8)
        if not isinstance(export_id, str) or len(export_id) != 16 or any(c not in '0123456789abcdef' for c in export_id):
            raise ValueError('Invalid export job identifier.')
        output = local_path('exports/hudyat/' + export_id + '.zip')
        return [sys.executable, '-u', '-B', str(BASE / 'tools/export_hudyat_model.py'),
                '--run', row['id'], '--output', str(output), '--hand-model', str(detector)]
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
        self.live_active = False
        self.closed = False

    def start(self, data):
        with self.lock:
            if self.closed:
                raise ValueError("The dashboard is closing.")
            if self.live_active:
                raise ValueError("Stop live camera prediction before starting a job.")
            if self.process and self.process.poll() is None:
                raise ValueError("A job is already running. Wait or stop it first.")
            job_id = secrets.token_hex(8)
            command = build_command({**data, '_job_id': job_id} if data.get('action') in {'export', 'hudyat-timing'} else data)
            process = subprocess.Popen(command, cwd=BASE, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                       errors="replace", bufsize=1,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            job = {"id": job_id, "action": data["action"],
                   "category": Path(data['run']).parent.name if data.get('run') and data['action'] in {'export', 'benchmark', 'hudyat-timing', 'paper-report'}
                   else data.get("category", "paper reports" if data['action'] == 'paper-report' else "prediction"),
                   "status": "running",
                   "dataset": data.get("dataset") or data.get("dataset_name"),
                   "run": data.get("run") if data["action"] in {"predict", "export", "benchmark", "hudyat-timing", "paper-report"} else None,
                   "sample": data.get("sample") if data["action"] == "predict" else None,
                   "started": datetime.now().isoformat(timespec="seconds"),
                   "command": subprocess.list2cmdline(command), "exit_code": None,
                   "logs": deque(maxlen=1800)}
            self.process = process
            self.items.appendleft(job)
            threading.Thread(target=self.collect, args=(process, job), daemon=True).start()
            return job["id"]

    def delete(self, data):
        with self.lock:
            if self.closed:
                raise ValueError('The dashboard is closing.')
            if self.live_active or self.process and self.process.poll() is None:
                raise ValueError('Stop the running job or camera prediction before deleting saved artifacts.')
            from .deletion import delete_artifact
            return delete_artifact(BASE, data)

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

    def snapshot(self, *, blocking=True):
        if not self.lock.acquire(blocking=blocking):
            return None
        try:
            return [{**item, "logs": list(item["logs"])} for item in self.items]
        finally:
            self.lock.release()

    def shutdown(self):
        with self.lock:
            self.closed = True
            process = self.process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def request_close(self, confirm):
        """Freeze new jobs on exit, asking first when a worker is running."""
        with self.lock:
            if not self.process or self.process.poll() is not None:
                self.closed = True
                return True
        # Never hold the job lock while a human is answering a dialog.
        if not confirm():
            return False
        with self.lock:
            self.closed = True
        return True


class Handler(BaseHTTPRequestHandler):
    def download(self, handle, filename, content_type="application/zip"):
        size = handle.seek(0, 2)
        handle.seek(0)
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.send_header('Content-Length', str(size))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        while chunk := handle.read(1024 * 1024):
            self.wfile.write(chunk)

    def reply(self, value, status=200, filename=None):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False,
                             indent=2 if filename else None).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
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
                                   "runs": runs(), "model_records": read_records(BASE),
                                   "prediction_tests": read_tests(BASE), "token": self.server.token,
                                   "python": sys.version.split()[0]})
            if url.path.startswith("/hudyat-timing/"):
                parts = url.path.split("/", 3)
                folder, config = camera_session(BASE, parts[2])
                path = local_path(unquote(parts[3]) if len(parts) > 3 and parts[3] else "index.html", folder)
                if not path.is_file():
                    return self.reply({"error": "Camera test file not found."}, 404)
                payload = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                return self.wfile.write(payload)
            if url.path.startswith('/paper-report/'):
                path = local_path(unquote(url.path[len('/paper-report/'):]))
                if (not path.is_file() or not (path.parent / 'report_manifest.json').is_file()
                        or path.name != 'index.html' and path.suffix not in {'.csv', '.png', '.svg', '.json'}):
                    return self.reply({'error': 'Report file not found. Build the tables and charts first.'}, 404)
                payload = path.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                return self.wfile.write(payload)
            if url.path == "/api/benchmarks/export":
                selected = runs()
                category = query.get("category", [""])[0]
                if category:
                    selected = [row for row in selected if row["category"] == category]
                if query.get("preferred", [""])[0] == "1":
                    records = read_records(BASE)
                    selected = [row for row in selected if records.get(row["id"], {}).get("decision") == "Preferred"]
                payload = benchmark_csv(selected).encode("utf-8-sig")
                return self.download(io.BytesIO(payload), "model-benchmarks.csv", "text/csv; charset=utf-8")
            if url.path == '/api/model/download':
                with native_archive(BASE, query.get('run', [''])[0], runs()) as (handle, filename):
                    return self.download(handle, filename)
            if url.path == '/api/model/export/download':
                job_id = query.get('id', [''])[0]
                job = next((j for j in self.server.jobs.snapshot()
                            if j['id'] == job_id and j['action'] == 'export' and j['status'] == 'completed'), None)
                if job is None:
                    raise ValueError('Complete a HUDYAT export before downloading its bundle.')
                path = local_path('exports/hudyat/' + job_id + '.zip')
                name = download_name(job['run'].replace('/', '-')) + '-hudyat.zip'
                with path.open('rb') as handle:
                    return self.download(handle, name)
            if url.path == "/api/model-record/export":
                exported = export_model(BASE, query.get("run", [""])[0], runs())
                name = exported["model"]["category"] + "-" + exported["model"]["name"]
                safe = "".join(c if c.isascii() and (c.isalnum() or c in "-_") else "-" for c in name)[:100]
                return self.reply(exported, filename=f"{safe}-record.json")
            if url.path == "/api/prediction-test/export":
                test = read_tests(BASE).get(query.get("id", [""])[0])
                if test is None:
                    raise ValueError("Prediction test not found.")
                format = query.get('format', ['json'])[0]
                if format in ('trials', 'metrics'):
                    if not test.get('automated'):
                        raise ValueError('CSV measurements require an automated test.')
                    payload = (trials_csv(test) if format == 'trials' else metrics_csv(test)).encode('utf-8-sig')
                    return self.download(io.BytesIO(payload), f"prediction-test-{test['number']}-{format}.csv",
                                         'text/csv; charset=utf-8')
                if format != 'json':
                    raise ValueError('Choose a JSON, trials CSV, or metrics CSV export.')
                return self.reply({"version": 1, "test": test}, filename=f"prediction-test-{test['number']}.json")
            if url.path == "/api/jobs":
                return self.reply(self.server.jobs.snapshot())
            if url.path == "/api/run":
                run_id = query.get("id", [""])[0]
                if run_id not in {r["id"] for r in runs()}:
                    raise ValueError("Run not found.")
                root = local_path(run_id)
                return self.reply({"metrics": read_json(root / "test_metrics.json", {}),
                                   "benchmark": next(row["benchmark"] for row in runs() if row["id"] == run_id),
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
            limit = 1_000_000 if self.path == "/api/live/frame" else 512000 if self.path == "/api/hudyat-timing" else 16384
            if not 0 < size <= limit:
                raise ValueError("Invalid request size.")
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object.")
            if self.path == "/api/hudyat-timing":
                return self.reply(save_camera_timing(BASE, data, runs()))
            if self.path == "/api/prediction-test":
                return self.reply(save_test(BASE, data, runs()))
            if self.path == "/api/prediction-test/trial":
                return self.reply(append_trial(BASE, data))
            if self.path == "/api/model-record":
                return self.reply({"model_records": save_record(BASE, data, runs())})
            if self.path == '/api/delete':
                return self.reply(self.server.jobs.delete(data))
            if self.path == "/api/live/start":
                run = local_path(data.get("run"))
                if BASE / "runs" not in run.parents or not (run / "best.keras").is_file():
                    raise ValueError("Select a run with a trained checkpoint.")
                if not read_json(run / "preprocessing.json", {}).get("extraction"):
                    raise ValueError("This run has no camera extraction settings; use normalized .npy input.")
                mode = data.get("mode", "motion")
                if mode not in ("pose", "motion", "gesture"):
                    raise ValueError("Choose Perform once, held pose, or moving gesture mode.")
                model = local_path(data["hand_model"]) if data.get("hand_model") else None
                if model and (not model.is_file() or model.suffix != ".task"):
                    raise ValueError("Choose an existing .task hand model.")
                return self.reply(self.server.live.start(run, model, mode))
            if self.path == "/api/live/frame":
                image = data.get("image")
                if not isinstance(image, str) or not image.startswith("data:image/jpeg;base64,") or len(image) > 900_000:
                    raise ValueError("Send a JPEG camera frame smaller than 900 KB.")
                timestamp = number(data, "timestamp", None, 0, 2**53)
                settings = {"timing": capture_timing(data["timing"])} if "timing" in data else {}
                return self.reply(self.server.live.frame(data.get("session"), image, timestamp, **settings))
            if self.path == "/api/live/stop":
                if not isinstance(data.get("session"), str) or not data["session"]:
                    raise ValueError("Specify the live session to stop.")
                self.server.live.stop(data["session"])
                return self.reply({"ok": True})
            if self.path == "/api/live/reset":
                if not isinstance(data.get("session"), str) or not data["session"]:
                    raise ValueError("Specify the live session to reset.")
                return self.reply(self.server.live.reset(data['session']))
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


class DashboardHTTPServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR permits a second dashboard to bind the same port.
    allow_reuse_address = sys.platform != "win32"

    def server_bind(self):
        if sys.platform == "win32":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8090, help="HTTP port for explicit --browser mode.")
    parser.add_argument("--browser", action="store_true",
                        help="Serve the browser UI without opening a desktop window.")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535 (0 selects a free port).")
    if not args.browser:
        from .desktop import DesktopWindow
        jobs = Jobs()
        try:
            DesktopWindow().run(jobs)
        except RuntimeError as error:
            parser.exit(1, f"FSL Studio: {error}\n")
        finally:
            jobs.shutdown()
        return
    try:
        server = DashboardHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as error:
        parser.exit(1, f"FSL Studio could not bind port {args.port}: {error}\n"
                    "Stop the other dashboard or use --port 8091.\n")
    try:
        server.token = secrets.token_hex(32)
        server.jobs = Jobs()
        server.live = LiveInference(server.jobs, BASE)
        url = f"http://localhost:{server.server_port}"
        print(f"FSL Studio: {url}", flush=True)
        print(f"Worker Python: {sys.executable}", flush=True)
        print("Open the URL in your browser. Press Ctrl+C to stop.", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except RuntimeError as error:
        parser.exit(1, f"FSL Studio: {error}\n")
    finally:
        try:
            if hasattr(server, "jobs"):
                server.jobs.shutdown()
        finally:
            try:
                if hasattr(server, "live"):
                    server.live.shutdown()
            finally:
                server.server_close()


if __name__ == "__main__":
    main()
