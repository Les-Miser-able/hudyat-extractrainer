"""Optional Windows host for the existing local dashboard UI."""
import sys


class DesktopWindow:
    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("Desktop mode currently supports Windows. "
                               "Use python dashboard.py --browser on this platform.")
        try:
            import webview
        except ImportError as error:
            raise RuntimeError(
                "Desktop dependencies are missing. Install them in this Python environment:\n"
                f'  & "{sys.executable}" -m pip install -r requirements-desktop.txt\n'
                "Or use python dashboard.py --browser."
            ) from error
        self.webview = webview

    def run(self, url, jobs):
        initialization_errors = []
        try:
            self.webview.settings['ALLOW_DOWNLOADS'] = True
            window = self.webview.create_window(
                "FSL Studio", url, width=1280, height=850, min_size=(800, 600),
                resizable=True, background_color="#f6f7f9", text_select=True,
            )

            def on_closing():
                return jobs.request_close(lambda: window.create_confirmation_dialog(
                    "Exit FSL Studio",
                    "Stop the running job and exit? Partial outputs will be kept.",
                ))

            def on_initialized(renderer):
                # pywebview can fall back to IE even when Edge was requested.
                return renderer == "edgechromium"

            def before_show():
                def on_ready(sender, args):
                    if not args.IsSuccess:
                        initialization_errors.append(str(args.InitializationException))
                        window.destroy()

                window.native.webview.CoreWebView2InitializationCompleted += on_ready

            window.events.closing += on_closing
            window.events.initialized += on_initialized
            window.events.before_show += before_show
            self.webview.start(gui="edgechromium", debug=False)
            if initialization_errors:
                raise RuntimeError(initialization_errors[0])
            if self.webview.renderer != "edgechromium":
                raise RuntimeError("Microsoft Edge WebView2 was not available.")
        except Exception as error:
            raise RuntimeError(
                f"Could not open the desktop window: {error}\n"
                "Install or repair the Microsoft Edge WebView2 Runtime:\n"
                "  https://developer.microsoft.com/microsoft-edge/webview2/\n"
                "Check desktop dependencies with python -m pip install -r requirements-desktop.txt.\n"
                "You can use python dashboard.py --browser in the meantime."
            ) from error
