"""One persistent inference worker, with no ML imports in the dashboard server."""
from collections import deque
import json
from queue import Queue, Empty
import secrets
import subprocess
import sys
import threading
import time


class LiveInference:
    def __init__(self, jobs, root):
        self.jobs, self.root = jobs, root
        self.lock = threading.RLock()
        self.process = None
        self.session = None
        self.last_frame = 0
        self.closed = threading.Event()
        threading.Thread(target=self.expire, daemon=True).start()

    def expire(self):
        while not self.closed.wait(5):
            with self.lock:
                if self.session and time.monotonic() - self.last_frame > 60:
                    self.stop(self.session)

    def start(self, run, hand_model, mode):
        with self.lock:
            if self.closed.is_set():
                raise ValueError("The dashboard is closing.")
            if self.session:
                raise ValueError("Live prediction is already active. Stop it before starting again.")
            with self.jobs.lock:
                if self.jobs.closed:
                    raise ValueError("The dashboard is closing.")
                if self.jobs.process and self.jobs.process.poll() is None:
                    raise ValueError("A job is running. Wait or stop it before using the camera.")
                self.jobs.live_active = True
            self.errors = deque(maxlen=20)
            self.responses = Queue()
            command = [sys.executable, "-u", "-B", "-m", "fsl_trainer.dashboard.live_worker",
                       "--run", str(run), "--mode", mode]
            if hand_model:
                command += ["--hand-model", str(hand_model)]
            try:
                self.process = subprocess.Popen(command, cwd=self.root, stdin=subprocess.PIPE,
                                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                text=True, encoding="utf-8", errors="replace", bufsize=1,
                                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                process, responses, errors = self.process, self.responses, self.errors

                def collect():
                    try:
                        for line in process.stdout:
                            try:
                                responses.put(json.loads(line))
                            except ValueError:
                                errors.append(line.rstrip())
                    finally:
                        process.stdout.close()
                        responses.put({"error": "Live worker stopped. " + "\n".join(errors)})

                def diagnostics():
                    try:
                        for line in process.stderr:
                            errors.append(line.rstrip())
                    finally:
                        process.stderr.close()

                threading.Thread(target=diagnostics, daemon=True).start()
                threading.Thread(target=collect, daemon=True).start()
                if self.closed.is_set():
                    raise ValueError("The dashboard is closing.")
                self.receive(120)
                self.session = secrets.token_hex(16)
                self.last_frame = time.monotonic()
                return {"session": self.session, "mode": mode,
                        "window_seconds": None if mode == 'gesture' else 2}
            except Exception:
                self.stop()
                raise

    def receive(self, timeout):
        try:
            response = self.responses.get(timeout=timeout)
        except Empty:
            raise ValueError("Live inference timed out. Check the Python dependencies and try again.") from None
        if "error" in response:
            raise ValueError(response["error"])
        return response

    def frame(self, session, image, timestamp):
        return self.exchange(session, {"image": image, "timestamp": timestamp})

    def reset(self, session):
        return self.exchange(session, {"action": "reset"})

    def exchange(self, session, request):
        with self.lock:
            if not session or session != self.session:
                raise ValueError("Live session expired. Start the camera again.")
            try:
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
                result = self.receive(30)
                self.last_frame = time.monotonic()
                return result
            except (OSError, ValueError):
                self.stop(session)
                raise

    def stop(self, session=None):
        with self.lock:
            if session is not None and session != self.session:
                return
            process, self.process = self.process, None
            self.session = None
            if process:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                try:
                    process.stdin.close()
                except OSError:
                    pass
            with self.jobs.lock:
                self.jobs.live_active = False

    def shutdown(self):
        self.closed.set()
        # Interrupt a pending startup/frame response before waiting for its lock.
        process = self.process
        if process and process.poll() is None:
            process.terminate()
        self.stop()
