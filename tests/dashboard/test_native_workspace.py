"""Offscreen interaction checks for the native application; optional Qt dependency."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fsl_trainer.desktop.artifacts import import_category, read_history, stage_progress


class ArtifactMonitorTests(unittest.TestCase):
    def test_epoch_reader_ignores_partial_writes_and_prefers_live_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'history.json').write_text('{"accuracy":[0.1],"loss":[2]}')
            self.assertEqual(read_history(root)['loss'], [2.0])
            (root / 'history.csv').write_text('epoch,accuracy,loss,val_accuracy,val_loss\n0,0.8,0.4,0.7,0.5\n1,0.9,')
            result = read_history(root)
            self.assertEqual(result['accuracy'], [.8])
            self.assertEqual(result['val_loss'], [.5])
            (root / 'history.csv').write_text('epoch,accuracy,loss,val_accuracy,val_loss\n0,nan,0.4,0.7,0.5\n')
            self.assertEqual(read_history(root)['loss'], [2.0])

    def test_progress_comes_from_actual_stage_output(self):
        self.assertEqual(stage_progress(['[Extract] working 1,200/2,000 (60.0%) | elapsed 20s', 'other output']),
                         ('Extract', 1200, 2000))
        self.assertIsNone(stage_progress(['Loading TensorFlow…']))

    def test_import_copies_originals_and_rejects_overwrite_or_recursive_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'external'
            (source / 'A').mkdir(parents=True)
            (source / 'A' / 'sample.mp4').write_bytes(b'original')
            target = root / 'project' / 'data' / 'new_category'
            import_category(source, target)
            self.assertEqual((target / 'A' / 'sample.mp4').read_bytes(), b'original')
            self.assertTrue((source / 'A' / 'sample.mp4').is_file())
            with self.assertRaisesRegex(ValueError, 'already exists'):
                import_category(source, target)
            with self.assertRaisesRegex(ValueError, 'non-nested'):
                import_category(source, source / 'copy')


HAS_QT = importlib.util.find_spec('PySide6') is not None


@unittest.skipUnless(HAS_QT, 'Install requirements-desktop.txt for native widget checks')
class NativeWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyle('Fusion')

    def setUp(self):
        from fsl_trainer.dashboard import server
        from fsl_trainer.desktop.window import StudioWindow
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)
        for gesture in ('A', 'B'):
            directory = self.root / 'data' / 'alphabet' / gesture
            directory.mkdir(parents=True)
            for index in range(3):
                (directory / f'{index}.jpg').write_bytes(b'source fixture')
        self.dataset = self.root / 'datasets_augmented' / 'alphabet' / 'prepared_v1'
        self.dataset.mkdir(parents=True)
        (self.dataset / 'manifest.json').write_text('[]')
        self.run = self.root / 'runs' / 'alphabet' / 'first'
        self.run.mkdir(parents=True)
        for name, value in {'classes.json':['A','B'], 'test_metrics.json':{'accuracy':.9,'macro_f1':.88},
                            'preprocessing.json':{}, 'training_config.json':{'epochs':5},
                            'history.json':{'accuracy':[.6,.8], 'val_accuracy':[.5,.7], 'loss':[.8,.4], 'val_loss':[.9,.5]}}.items():
            (self.run / name).write_text(json.dumps(value))
        (self.run / 'best.keras').write_bytes(b'checkpoint fixture')
        base = patch.object(server, 'BASE', self.root)
        base.start()
        self.addCleanup(base.stop)
        self.jobs = server.Jobs()
        self.window = StudioWindow(self.jobs, restore=False)
        self.addCleanup(self.window.shutdown)
        self.wait_for(lambda: not self.window.scanning)

    def wait_for(self, predicate, timeout=8):
        started = time.monotonic()
        while not predicate():
            self.app.processEvents()
            if time.monotonic() - started > timeout:
                self.fail('Native UI did not complete its background operation')
            time.sleep(.01)
        self.app.processEvents()

    def test_native_explorer_selects_prepared_dataset_and_uses_existing_job_validation(self):
        from fsl_trainer.dashboard import server
        window = self.window
        self.assertEqual(window.tabs.count(), 4)
        self.assertEqual(window.prepare_fields['category'].currentData(), 'alphabet')
        self.assertEqual(window.train_fields['dataset'].currentData(), 'datasets_augmented/alphabet/prepared_v1')
        prepared = window.tree.topLevelItem(1).child(0)
        window.activate_item(prepared)
        self.assertEqual(window.tabs.currentIndex(), 1)
        with patch.object(self.jobs, 'start', return_value='native-job') as start:
            window.train_fields['run'].setText('native_run')
            window.train_fields['epochs'].setValue(7)
            window.train()
            self.wait_for(lambda: start.called and not window.tasks)
            request = start.call_args.args[0]
        self.assertEqual(request['epochs'], 7)
        command = server.build_command(request)
        self.assertIn(str(self.dataset), command)
        self.assertIn(str(self.root / 'runs' / 'alphabet' / 'native_run'), command)
        self.assertIn('--epochs', command)

    def test_saved_run_curves_and_review_preserve_checkpoint(self):
        window = self.window
        before = (self.run / 'best.keras').read_bytes()
        window.run_table.selectRow(0)
        window.view_run_curves()
        self.assertEqual(window.accuracy_plot.history['accuracy'], [.6,.8])
        self.assertIn('70.0%', window.training_metrics.text())
        window.notes.setPlainText('Native review')
        window.decision.setCurrentIndex(window.decision.findData('Preferred'))
        window.save_review()
        self.wait_for(lambda: not window.tasks)
        saved = json.loads((self.root / 'runs' / 'model_records.json').read_text())
        self.assertEqual(saved['records']['runs/alphabet/first']['notes'], 'Native review')
        self.assertEqual((self.run / 'best.keras').read_bytes(), before)

    def test_all_workspaces_render_as_native_widgets(self):
        from PySide6.QtGui import QFontDatabase
        font = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / 'segoeui.ttf'
        if font.is_file():
            QFontDatabase.addApplicationFont(str(font))
        self.window.show()
        self.window.view_run_curves()
        for index in range(self.window.tabs.count()):
            self.window.tabs.setCurrentIndex(index)
            self.app.processEvents()
            screenshot = self.window.grab()
            self.assertFalse(screenshot.isNull())
        self.window.hide()

    def test_sample_result_can_be_saved_with_the_model_it_came_from(self):
        window = self.window
        window.sample_result({'run':'runs/alphabet/first', 'sample':'data/alphabet/A/0.jpg',
                              'logs':['initialization output', '{"A":0.9,"B":0.1}']})
        self.assertIn('A', window.prediction_title.text())
        window.save_prediction()
        self.wait_for(lambda: not window.tasks)
        stored = json.loads((self.root / 'runs' / 'prediction_tests.json').read_text())
        test = next(iter(stored['tests'].values()))
        self.assertEqual(test['run'], 'runs/alphabet/first')
        self.assertEqual(test['prediction']['probabilities'], {'A':.9,'B':.1})

    def test_refresh_keeps_recognition_and_unsaved_review_edits(self):
        window = self.window
        window.run_table.selectRow(0)
        window.notes.setPlainText('An unfinished review')
        window.show_prediction({'A':.9,'B':.1}, 'runs/alphabet/first', 'sample')
        window.refresh()
        self.wait_for(lambda: not window.scanning)
        self.assertEqual(window.notes.toPlainText(), 'An unfinished review')
        self.assertEqual(window.last_prediction['prediction']['probabilities'], {'A':.9,'B':.1})
        self.assertTrue(window.save_prediction_button.isEnabled())

    def test_camera_preview_is_larger_and_mirroring_changes_display_without_mutating_frames(self):
        from PySide6.QtGui import QColor, QImage, QPainter
        window = self.window
        window.tabs.setCurrentIndex(3)
        window.show()
        frame = QImage(640, 480, QImage.Format.Format_RGB888)
        frame.fill(QColor('red'))
        painter = QPainter(frame)
        painter.fillRect(320, 0, 320, 480, QColor('blue'))
        painter.end()
        window.camera_frame(frame)
        self.app.processEvents()
        canvas = window.camera_canvas
        self.assertGreaterEqual(canvas.height(), 300)
        self.assertLessEqual(window.prediction_splitter.widget(1).height(), 160)
        original = canvas.grab().toImage()
        left, right, middle = canvas.width() * 2 // 5, canvas.width() * 3 // 5, canvas.height() // 2
        self.assertEqual(original.pixelColor(left, middle).name(), '#ff0000')
        self.assertEqual(original.pixelColor(right, middle).name(), '#0000ff')
        window.camera_mirror.setChecked(True)
        mirrored = canvas.grab().toImage()
        self.assertEqual(mirrored.pixelColor(left, middle).name(), '#0000ff')
        self.assertEqual(mirrored.pixelColor(right, middle).name(), '#ff0000')
        self.assertEqual(frame.pixelColor(0, 0).name(), '#ff0000')
        self.assertEqual(frame.pixelColor(639, 0).name(), '#0000ff')
        self.assertTrue(window.settings.value('mirror_preview', False, type=bool))
        window.settings.sync()
        window.hide()

    def test_fullscreen_camera_streams_frames_and_keeps_mirror_controls_in_sync(self):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QImage
        from PySide6.QtTest import QTest
        window = self.window
        window.camera_mirror.setChecked(True)
        frame = QImage(640, 480, QImage.Format.Format_RGB888)
        frame.fill(QColor('green'))
        window.camera_frame(frame)
        window.open_camera_preview()
        self.app.processEvents()
        self.assertTrue(window.camera_preview.isFullScreen())
        self.assertTrue(window.preview_canvas.mirrored)
        self.assertTrue(window.preview_mirror.isChecked())
        window.preview_mirror.setChecked(False)
        self.assertFalse(window.camera_mirror.isChecked())
        self.assertFalse(window.camera_canvas.mirrored)
        next_frame = frame.copy()
        next_frame.fill(QColor('blue'))
        window.camera_frame(next_frame)
        self.assertEqual(window.preview_canvas.frame.pixelColor(0, 0).name(), '#0000ff')
        QTest.keyClick(window.camera_preview, Qt.Key.Key_Escape)
        self.app.processEvents()
        self.assertFalse(window.camera_preview.isVisible())
        window.settings.sync()

    def test_running_training_csv_updates_curves_and_cancel_uses_real_job_identity(self):
        from collections import deque
        from unittest.mock import Mock
        window = self.window
        self.jobs.process = Mock()
        self.jobs.process.poll.return_value = None
        self.jobs.items.appendleft({'id':'training', 'action':'train', 'category':'alphabet', 'status':'running',
                                   'started':'now', 'logs':deque(['training']), 'exit_code':None})
        window.training_path, window.training_epochs = self.run, 5
        (self.run / 'history.csv').write_text('epoch,accuracy,loss,val_accuracy,val_loss\n0,0.9,0.3,0.8,0.4\n')
        window.poll_jobs()
        self.assertEqual(window.training_progress.value(), 1)
        self.assertEqual(window.accuracy_plot.history['val_accuracy'], [.8])
        self.assertFalse(window.train_button.isEnabled())
        window.stop_job()
        self.jobs.process.terminate.assert_called_once()
        self.assertEqual(self.jobs.items[0]['status'], 'cancelled')
        self.jobs.process.poll.return_value = 1

    @unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('cv2'), 'Media preview uses numpy and OpenCV')
    def test_landmark_preview_seek_and_shutdown_without_starting_a_job(self):
        import numpy as np
        path = self.root / 'data' / 'alphabet' / 'A' / 'sequence.npy'
        features = np.zeros((4, 128), dtype=np.float32)
        features[:, 126] = 1
        np.save(path, features)
        self.window.open_media(path)
        self.wait_for(lambda: self.window.canvas.frame is not None)
        self.assertEqual(self.window.timeline.maximum(), 3)
        self.window.seek_media(3)
        self.wait_for(lambda: self.window.timeline.value() == 3)
        self.assertEqual(self.jobs.snapshot(), [])
        self.window.shutdown()
        self.assertFalse(self.window.media.isRunning())

    @unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('cv2'), 'Camera transport uses numpy and OpenCV')
    def test_camera_worker_uses_existing_local_inference_protocol_and_releases_device(self):
        import numpy as np
        from unittest.mock import Mock
        from fsl_trainer.dashboard.live import LiveInference
        from fsl_trainer.desktop.workers import CameraThread
        live, capture = LiveInference(self.jobs, self.root), Mock()
        self.addCleanup(live.shutdown)
        live.start = Mock(return_value={'session':'test-session'})
        capture.read.return_value = (True, np.zeros((480,640,3), dtype=np.uint8))
        timing = {'settle_ms':600, 'missing_ms':400, 'frame_interval_ms':50}
        worker = CameraThread(live, self.run, 'gesture', 0, timing)
        self.assertEqual(worker.frame_interval_ms, 50)
        self.assertEqual(timing, {'settle_ms':600, 'missing_ms':400, 'frame_interval_ms':50})

        errors = []
        worker.failed.connect(errors.append)

        def inference(*args):
            worker.stop()
            return {'status':'prediction', 'probabilities':{'A':.9,'B':.1}}

        live.exchange = Mock(side_effect=inference)
        with patch('cv2.VideoCapture', return_value=capture):
            worker.start()
            self.wait_for(lambda: worker.isFinished())
            worker.wait()
        live.start.assert_called_once_with(self.run, None, 'gesture')
        self.assertEqual(errors, [], 'The native camera settings must pass the real LiveInference.frame validator')
        request = live.exchange.call_args.args
        self.assertEqual(request[0], 'test-session')
        self.assertTrue(request[1]['image'].startswith('data:image/jpeg;base64,'))
        self.assertEqual(request[1]['timing'], {'settle_ms':600, 'missing_ms':400})
        capture.release.assert_called_once()
        self.assertFalse(self.jobs.live_active)


if __name__ == '__main__':
    unittest.main()
