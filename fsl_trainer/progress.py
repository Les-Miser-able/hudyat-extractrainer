"""Periodic console status for long preparation stages (stderr, flushed)."""
import sys
import threading
import time


class Stage:
    def __init__(self, label, total=None, interval=10):
        self.label, self.total, self.interval = label, total, interval
        self.count = 0
        self.stop = threading.Event()

    def message(self, status="working"):
        elapsed = time.monotonic() - self.started
        detail = ""
        if self.total is not None:
            percent = 100 * self.count / self.total if self.total else 100
            detail = f" {self.count:,}/{self.total:,} ({percent:.1f}%)"
        print(f"[{self.label}] {status}{detail} | elapsed {elapsed:.0f}s",
              file=sys.stderr, flush=True)

    def heartbeat(self):
        while not self.stop.wait(self.interval):
            self.message()

    def __enter__(self):
        self.started = time.monotonic()
        self.message("starting")
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        self.thread.join()
        self.message("failed" if exc_type else "done")

    def advance(self):
        self.count += 1


def progress(items, label):
    with Stage(label, len(items)) as stage:
        for item in items:
            yield item
            stage.advance()
