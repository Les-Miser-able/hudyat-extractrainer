"""Render the actual native workspaces offscreen, without starting jobs/cameras.

Run with the project's Python: tools/preview_desktop.py --output outputs/native-desktop
"""
import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication

from fsl_trainer.dashboard.server import Jobs
from fsl_trainer.desktop.window import StudioWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/native-desktop")
    parser.add_argument("--sample", help="Optional source to preview; no extraction job is started")
    parser.add_argument("--overlay", action="store_true", help="Preview detected hands using the existing local detector")
    parser.add_argument("--frame", type=int, default=0, help="Video/sequence frame to show (zero based)")
    parser.add_argument("--camera-layout", action="store_true", help="Also show --sample in the camera layout and fullscreen; never opens a camera")
    args = parser.parse_args()
    app = QApplication([])
    app.setStyle("Fusion")
    # Windows' offscreen plugin does not enumerate installed system fonts.
    for filename in ("segoeui.ttf", "segoeuib.ttf", "consola.ttf"):
        font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / filename
        if font.is_file():
            QFontDatabase.addApplicationFont(str(font))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    window = StudioWindow(Jobs(), restore=False)
    # Layout previews must not change the user's saved mirror/window preferences.
    window.settings = QSettings(str(output.resolve() / "preview.ini"), QSettings.Format.IniFormat)
    errors = []
    window.error = lambda message: errors.append(message)
    window.show()
    started = time.monotonic()
    rendered = False

    def capture():
        nonlocal rendered
        if errors:
            print("Preview failed: " + "\n".join(errors), flush=True)
            timer.stop()
            window.shutdown()
            app.exit(1)
            return
        if window.scanning:
            return
        if not rendered:
            rendered = True
            if args.sample:
                window.overlay.setChecked(args.overlay)
                window.open_media(Path(args.sample).resolve())
                return
            else:
                save()
                return
        if args.sample and window.canvas.frame is not None:
            target = min(max(0, args.frame), window.timeline.maximum())
            if window.timeline.value() != target:
                window.seek_media(target)
                return
            save()

    def save():
        for index, name in enumerate(("extractor", "trainer", "models", "prediction")):
            if index == 1 and window.runs:
                window.run_table.selectRow(0)
                window.view_run_curves()
            window.tabs.setCurrentIndex(index)
            if index == 3 and args.camera_layout and window.canvas.frame is not None:
                window.camera_frame(window.canvas.frame)
                window.set_camera_status("Layout preview using a local sample; camera is off.")
                window.camera_mirror.setChecked(True)
            app.processEvents()
            if not window.grab().save(str(output / (name + ".png"))):
                raise RuntimeError("Could not save native preview.")
        if args.camera_layout:
            window.open_camera_preview()
            app.processEvents()
            window.camera_preview.grab().save(str(output / "camera-fullscreen.png"))
            window.camera_preview.close()
        print(f"Rendered 4 native workspaces; {len(window.categories)} categories, {len(window.runs)} saved runs; "
              f"{time.monotonic() - started:.1f}s", flush=True)
        timer.stop()
        window.shutdown()
        app.quit()

    timer = QTimer()
    timer.timeout.connect(capture)
    timer.start(500)
    try:
        code = app.exec()
    finally:
        window.shutdown()
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
