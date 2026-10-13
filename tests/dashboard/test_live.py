"""Live worker lifetime and job exclusion; no ML dependencies."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import dashboard
from fsl_trainer.dashboard.live import LiveInference


class LiveControllerTests(unittest.TestCase):
    def setUp(self):
        self.jobs = dashboard.Jobs()
        self.live = LiveInference(self.jobs, Path.cwd())
        self.addCleanup(self.live.shutdown)

    def worker(self, script):
        popen = subprocess.Popen
        return patch('fsl_trainer.dashboard.live.subprocess.Popen',
                     side_effect=lambda *a, **kw: popen([sys.executable, '-u', '-c', script], **kw))

    def test_frames_reuse_worker_and_stop_releases_job_slot(self):
        script = "import json,sys; print(json.dumps({'ready':True}),flush=True)\nfor line in sys.stdin: print(json.dumps({'status':'no_hand'}),flush=True)"
        with self.worker(script) as starts:
            ready = self.live.start('fixture', None, 'motion')
            session = ready['session']
            self.assertEqual(self.live.frame(session, 'fixture', 1)['status'], 'no_hand')
            self.assertEqual(self.live.frame(session, 'fixture', 2)['status'], 'no_hand')
            self.assertEqual(starts.call_count, 1)
            with self.assertRaisesRegex(ValueError, 'Stop live'):
                self.jobs.start({'action': 'inspect'})
            with self.assertRaisesRegex(ValueError, 'already active'):
                self.live.start('fixture', None, 'pose')
            self.live.stop('stale-session')
            self.assertTrue(self.jobs.live_active)
            self.live.stop(session)
            self.assertFalse(self.jobs.live_active)
            with self.assertRaisesRegex(ValueError, 'expired'):
                self.live.frame(session, 'fixture', 3)

    def test_startup_and_inference_failures_release_worker(self):
        for script, fail_during_frame in [
            ("import json; print(json.dumps({'error':'bad asset'}),flush=True)", False),
            ("import json,sys; print(json.dumps({'ready':True}),flush=True); sys.stdin.readline(); print(json.dumps({'error':'bad frame'}),flush=True)", True),
        ]:
            with self.subTest(frame=fail_during_frame), self.worker(script):
                with self.assertRaisesRegex(ValueError, 'bad'):
                    ready = self.live.start('fixture', None, 'pose')
                    if fail_during_frame:
                        self.live.frame(ready['session'], 'fixture', 1)
                self.assertFalse(self.jobs.live_active)
                self.assertIsNone(self.live.process)

    def test_next_gesture_reuses_session_and_model_worker(self):
        script = "import json,sys; print(json.dumps({'ready':True}),flush=True)\nfor line in sys.stdin:\n request=json.loads(line); print(json.dumps({'status':'waiting' if request.get('action')=='reset' else 'held'}),flush=True)"
        with self.worker(script) as starts:
            ready = self.live.start('fixture', None, 'gesture')
            self.assertIsNone(ready['window_seconds'])
            self.assertEqual(self.live.frame(ready['session'], 'fixture', 1)['status'], 'held')
            self.assertEqual(self.live.reset(ready['session'])['status'], 'waiting')
            self.assertEqual(starts.call_count, 1)
            self.assertTrue(self.jobs.live_active)
