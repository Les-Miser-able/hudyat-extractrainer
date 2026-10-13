"""Desktop launch and shutdown checks without requiring a GUI or ML packages."""
import io
import subprocess
import sys
import threading
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import Mock, patch
from urllib.error import URLError
from urllib.request import urlopen

import dashboard
from fsl_trainer.dashboard.desktop import DesktopWindow


class Event(list):
    def __iadd__(self, callback):
        self.append(callback)
        return self


class DesktopWindowTests(unittest.TestCase):
    def setUp(self):
        self.webview = Mock(renderer="edgechromium")
        self.webview.create_window.return_value.events.closing = Event()
        self.webview.create_window.return_value.events.initialized = Event()
        self.webview.create_window.return_value.events.before_show = Event()
        self.webview.create_window.return_value.native.webview.CoreWebView2InitializationCompleted = Event()
        self.platform = patch('sys.platform', 'win32')
        self.modules = patch.dict(sys.modules, webview=self.webview)
        self.platform.start()
        self.modules.start()
        self.addCleanup(self.platform.stop)
        self.addCleanup(self.modules.stop)

    def test_window_uses_existing_http_url_and_main_thread(self):
        jobs = dashboard.Jobs()
        self.webview.start.side_effect = lambda **kw: self.assertIs(
            threading.current_thread(), threading.main_thread())
        DesktopWindow().run('http://localhost:8091', jobs)
        self.webview.create_window.assert_called_once_with(
            'FSL Studio', 'http://localhost:8091', width=1280, height=850,
            min_size=(800, 600), resizable=True, background_color='#f6f7f9', text_select=True)
        self.webview.start.assert_called_once_with(gui='edgechromium', debug=False)
        initialized = self.webview.create_window.return_value.events.initialized[0]
        self.assertTrue(initialized('edgechromium'))
        self.assertFalse(initialized('mshtml'))

    def test_close_with_running_job_can_be_cancelled_or_confirmed(self):
        jobs = dashboard.Jobs()
        jobs.process = Mock()
        jobs.process.poll.return_value = None
        DesktopWindow().run('http://localhost:8090', jobs)
        window = self.webview.create_window.return_value
        close = window.events.closing[0]
        window.create_confirmation_dialog.return_value = False
        self.assertFalse(close())
        self.assertFalse(jobs.closed)
        jobs.process.terminate.assert_not_called()
        window.create_confirmation_dialog.return_value = True
        self.assertTrue(close())
        self.assertTrue(jobs.closed)
        window.create_confirmation_dialog.assert_called_with(
            'Exit FSL Studio', 'Stop the running job and exit? Partial outputs will be kept.')
        with self.assertRaisesRegex(ValueError, 'closing'):
            jobs.start({'action': 'inspect'})

    def test_idle_close_needs_no_confirmation(self):
        jobs = dashboard.Jobs()
        DesktopWindow().run('http://localhost:8090', jobs)
        window = self.webview.create_window.return_value
        self.assertTrue(window.events.closing[0]())
        window.create_confirmation_dialog.assert_not_called()
        self.assertTrue(jobs.closed)

    def test_missing_dependencies_and_unsupported_platform_explain_recovery(self):
        with patch.dict(sys.modules, webview=None):
            with self.assertRaisesRegex(RuntimeError, 'requirements-desktop.txt'):
                DesktopWindow()
        with patch('sys.platform', 'linux'):
            with self.assertRaisesRegex(RuntimeError, '--browser'):
                DesktopWindow()

    def test_failed_startup_and_legacy_renderer_explain_recovery(self):
        self.webview.start.side_effect = OSError('runtime missing')
        with self.assertRaisesRegex(RuntimeError, 'WebView2 Runtime'):
            DesktopWindow().run('http://localhost:8090', dashboard.Jobs())
        self.webview.start.side_effect = None
        self.webview.renderer = 'mshtml'
        with self.assertRaisesRegex(RuntimeError, 'WebView2 was not available'):
            DesktopWindow().run('http://localhost:8090', dashboard.Jobs())

    def test_async_webview_initialization_failure_closes_the_window(self):
        window = self.webview.create_window.return_value

        def start(**kwargs):
            window.events.before_show[0]()
            ready = window.native.webview.CoreWebView2InitializationCompleted[0]
            ready(None, Mock(IsSuccess=False, InitializationException='runtime damaged'))

        self.webview.start.side_effect = start
        with self.assertRaisesRegex(RuntimeError, 'runtime damaged'):
            DesktopWindow().run('http://localhost:8090', dashboard.Jobs())
        window.destroy.assert_called_once()


class DashboardLaunchTests(unittest.TestCase):
    def test_desktop_has_a_serving_backend_and_cleans_up_on_exit_or_error(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                jobs = dashboard.Jobs()
                live = dashboard.LiveInference(jobs, dashboard.BASE)
                self.addCleanup(live.shutdown)
                desktop = Mock()
                addresses = []

                def run(url, actual_jobs):
                    addresses.append(url)
                    self.assertIs(actual_jobs, jobs)
                    with urlopen(url + '/', timeout=5) as response:
                        self.assertIn(b'FSL Studio', response.read())
                    if fail:
                        raise RuntimeError('window startup failed')

                desktop.run.side_effect = run
                with patch('fsl_trainer.dashboard.desktop.DesktopWindow', return_value=desktop), \
                        patch.object(dashboard, 'Jobs', return_value=jobs), \
                        patch.object(dashboard, 'LiveInference', return_value=live), \
                        redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    if fail:
                        with self.assertRaises(SystemExit) as error:
                            dashboard.main(['--port', '0'])
                        self.assertEqual(error.exception.code, 1)
                    else:
                        dashboard.main(['--port', '0'])
                self.assertTrue(jobs.closed)
                self.assertTrue(live.closed.is_set())
                self.assertFalse(any(t.name == 'dashboard-http' for t in threading.enumerate()))
                with self.assertRaises(URLError):
                    urlopen(addresses[0], timeout=1)

    def test_browser_mode_does_not_load_desktop_dependencies(self):
        with patch.dict(sys.modules, webview=None), \
                patch('fsl_trainer.dashboard.desktop.DesktopWindow') as desktop, \
                patch.object(dashboard.ThreadingHTTPServer, 'serve_forever', side_effect=KeyboardInterrupt), \
                redirect_stdout(io.StringIO()):
            dashboard.main(['--browser', '--port', '0'])
        desktop.assert_not_called()

    def test_missing_dependencies_do_not_start_a_server(self):
        with patch('sys.platform', 'win32'), patch.dict(sys.modules, webview=None), \
                patch.object(dashboard, 'DashboardHTTPServer') as server, \
                redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as error:
                dashboard.main([])
        self.assertEqual(error.exception.code, 1)
        self.assertIn('requirements-desktop.txt', errors.getvalue())
        server.assert_not_called()

    def test_occupied_port_and_invalid_port_explain_recovery(self):
        server = dashboard.DashboardHTTPServer(('127.0.0.1', 0), dashboard.Handler)
        self.addCleanup(server.server_close)
        with redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as error:
                dashboard.main(['--browser', '--port', str(server.server_port)])
        self.assertEqual(error.exception.code, 1)
        self.assertIn('--port 8091', errors.getvalue())
        for port in ('-1', '65536'):
            with self.subTest(port=port), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    dashboard.main(['--browser', '--port', port])
                self.assertEqual(error.exception.code, 2)


class JobShutdownTests(unittest.TestCase):
    def test_shutdown_waits_for_a_real_worker_and_preserves_partial_output(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp:
            partial = Path(temp) / 'partial.txt'
            code = ('import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text("partial"); '
                    'print("ready", flush=True); time.sleep(60)')
            process = subprocess.Popen([sys.executable, '-u', '-c', code, str(partial)],
                                       stdout=subprocess.PIPE, text=True,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            jobs = dashboard.Jobs()
            jobs.process = process
            try:
                self.assertEqual(process.stdout.readline().strip(), 'ready')
                jobs.shutdown()
                self.assertIsNotNone(process.poll())
                self.assertEqual(partial.read_text(), 'partial')
                with self.assertRaisesRegex(ValueError, 'closing'):
                    jobs.start({})
            finally:
                jobs.shutdown()
                process.stdout.close()

    def test_unresponsive_worker_is_killed_after_timeout(self):
        jobs = dashboard.Jobs()
        jobs.process = Mock()
        jobs.process.poll.return_value = None
        jobs.process.wait.side_effect = [subprocess.TimeoutExpired('worker', 5), 1]
        jobs.shutdown()
        jobs.process.terminate.assert_called_once()
        jobs.process.kill.assert_called_once()
        self.assertEqual(jobs.process.wait.call_count, 2)


if __name__ == '__main__':
    unittest.main()
