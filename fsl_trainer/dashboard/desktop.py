"""Native Qt launcher. Importing the browser backend never requires Qt."""
import sys


class DesktopWindow:
    def __init__(self):
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as error:
            raise RuntimeError(
                "Native desktop dependencies are missing. Install them in this Python environment:\n"
                f'  & "{sys.executable}" -m pip install -r requirements-desktop.txt\n'
                "The command-line tools and optional --browser mode remain available."
            ) from error
        self.application_class = QApplication

    def run(self, jobs):
        from ..desktop.window import StudioWindow

        app = self.application_class.instance() or self.application_class(["FSL Studio"])
        app.setApplicationName("FSL Studio")
        app.setOrganizationName("FSL Studio")
        app.setStyle("Fusion")
        self.window = StudioWindow(jobs)
        self.window.show()
        try:
            app.exec()
        finally:
            self.window.shutdown()
