"""Native launch and shutdown contracts, independent of optional Qt/ML packages."""
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

import dashboard
from fsl_trainer.dashboard.desktop import DesktopWindow


class DashboardLaunchTests(unittest.TestCase):
    def test_native_launch_never_constructs_a_server_or_browser_inference_host(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                jobs, desktop = dashboard.Jobs(), Mock()
                if fail:
                    desktop.run.side_effect = RuntimeError("native startup failed")
                with patch('fsl_trainer.dashboard.desktop.DesktopWindow', return_value=desktop), \
                        patch.object(dashboard, 'Jobs', return_value=jobs), \
                        patch.object(dashboard, 'DashboardHTTPServer') as server, \
                        patch.object(dashboard, 'LiveInference') as live, redirect_stderr(io.StringIO()):
                    if fail:
                        with self.assertRaises(SystemExit) as error:
                            dashboard.main([])
                        self.assertEqual(error.exception.code, 1)
                    else:
                        dashboard.main([])
                desktop.run.assert_called_once_with(jobs)
                server.assert_not_called()
                live.assert_not_called()
                self.assertTrue(jobs.closed)

    def test_browser_mode_does_not_load_native_dependencies(self):
        with patch.dict(sys.modules, PySide6=None), \
                patch('fsl_trainer.dashboard.desktop.DesktopWindow') as desktop, \
                patch.object(dashboard.ThreadingHTTPServer, 'serve_forever', side_effect=KeyboardInterrupt), \
                redirect_stdout(io.StringIO()):
            dashboard.main(['--browser', '--port', '0'])
        desktop.assert_not_called()

    def test_missing_native_dependencies_explain_recovery_without_starting_a_server(self):
        with patch.dict(sys.modules, {'PySide6.QtWidgets': None}), \
                patch.object(dashboard, 'DashboardHTTPServer') as server, \
                redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as error:
                dashboard.main([])
        self.assertEqual(error.exception.code, 1)
        self.assertIn('requirements-desktop.txt', errors.getvalue())
        server.assert_not_called()

    def test_native_close_with_active_job_can_be_declined(self):
        jobs = dashboard.Jobs()
        jobs.process = Mock()
        jobs.process.poll.return_value = None
        self.assertFalse(jobs.request_close(lambda: False))
        self.assertFalse(jobs.closed)
        jobs.process.terminate.assert_not_called()
        self.assertTrue(jobs.request_close(lambda: True))
        with self.assertRaisesRegex(ValueError, 'closing'):
            jobs.start({'action': 'inspect'})

    def test_idle_close_needs_no_confirmation(self):
        jobs, confirm = dashboard.Jobs(), Mock()
        self.assertTrue(jobs.request_close(confirm))
        confirm.assert_not_called()

    def test_native_poll_can_skip_a_snapshot_while_job_validation_holds_the_lock(self):
        jobs = dashboard.Jobs()
        with jobs.lock:
            self.assertIsNone(jobs.snapshot(blocking=False))
        self.assertEqual(jobs.snapshot(), [])

    def test_occupied_port_only_applies_to_explicit_browser_mode(self):
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
    def test_shutdown_waits_for_real_worker_and_preserves_partial_output(self):
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
