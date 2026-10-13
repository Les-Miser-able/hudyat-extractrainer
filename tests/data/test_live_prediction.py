"""Streaming inference preprocessing uses the same 32x128 layout as training."""
import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from fsl_trainer.dashboard.live_worker import LivePredictor, decode_frame, load_predictor


class LivePredictionTests(unittest.TestCase):
    def predictor(self, mode):
        samples = []
        extractor = NS(swap_hands=False, image=lambda rgb: rgb, close=lambda: None)
        extractor.image_detector = NS(detect=lambda image: image)
        extractor.detector = lambda video: NS(detect_for_video=lambda image, timestamp: image, close=lambda: None)

        def model(sequence, training):
            self.assertFalse(training)
            samples.append(sequence.copy())
            return np.array([[.2, .8]])

        predictor = LivePredictor(model, ['A', 'B'], extractor, mode)
        self.addCleanup(predictor.close)
        return predictor, samples

    def feature(self, value):
        x = np.zeros(128, dtype=np.float32)
        x[:63], x[126] = value, 1
        return x

    def test_pose_repeats_current_frame_and_clears_missing_hand(self):
        predictor, samples = self.predictor('pose')
        with patch('fsl_trainer.dashboard.live_worker.pack_result', side_effect=lambda result, swap: result):
            result = predictor.frame(self.feature(.3), 1)
            self.assertEqual(result['probabilities'], {'A': .2, 'B': .8})
            np.testing.assert_array_equal(samples[0][0], np.repeat(self.feature(.3)[None], 32, axis=0))
            result = predictor.frame(np.zeros(128), 2)
            self.assertEqual(result['status'], 'no_hand')
            self.assertEqual(len(samples), 1)

    def test_motion_waits_full_window_and_uses_capture_timestamps(self):
        predictor, samples = self.predictor('motion')
        with patch('fsl_trainer.dashboard.live_worker.pack_result', side_effect=lambda result, swap: result):
            for timestamp in (0, 300, 700, 1200, 1800):
                self.assertEqual(predictor.frame(self.feature(timestamp / 2000), timestamp)['status'], 'buffering')
            self.assertEqual(predictor.frame(self.feature(1), 2000)['status'], 'prediction')
            np.testing.assert_allclose(samples[0][0, :, 0], np.linspace(0, 1, 32), atol=1e-7)
            self.assertEqual(samples[0].shape, (1, 32, 128))
            predictor.frame(np.zeros(128), 2100)
            self.assertEqual(predictor.frame(self.feature(.5), 2200)['status'], 'buffering')
            for timestamp in (2200, -1, float('nan'), True):
                with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                    predictor.frame(self.feature(.5), timestamp)

    def test_one_gesture_preserves_opening_motion_and_holds_one_prediction(self):
        predictor, samples = self.predictor('gesture')
        with patch('fsl_trainer.dashboard.live_worker.pack_result', side_effect=lambda result, swap: result):
            for timestamp in range(0, 3200, 200):
                result = predictor.frame(self.feature(.1 + timestamp / 10000), timestamp)
                self.assertNotEqual(result['status'], 'prediction')
            for timestamp in (3200, 3400, 3600, 3800):
                result = predictor.frame(self.feature(.4), timestamp)
            self.assertEqual(result['status'], 'prediction')
            self.assertTrue(result['complete'])
            self.assertEqual(len(samples), 1)
            self.assertAlmostEqual(float(samples[0][0, 0, 0]), .1)
            self.assertAlmostEqual(float(samples[0][0, -1, 0]), .4)
            self.assertEqual(predictor.frame(self.feature(.8), 4000)['status'], 'held')
            self.assertEqual(predictor.frame(np.zeros(128), 4200)['status'], 'held')
            self.assertEqual(len(samples), 1)
            self.assertEqual(predictor.reset()['status'], 'waiting')
            for timestamp in range(4400, 5600, 200):
                predictor.frame(self.feature(.2 + (timestamp - 4400) / 10000), timestamp)
            for timestamp in (5600, 5800, 6000, 6200):
                result = predictor.frame(self.feature(.3), timestamp)
            self.assertEqual(result['status'], 'prediction')
            self.assertEqual(len(samples), 2)
            self.assertAlmostEqual(float(samples[1][0, 0, 0]), .2)

    def test_stationary_pose_and_small_jitter_do_not_start_gesture_capture(self):
        predictor, samples = self.predictor('gesture')
        with patch('fsl_trainer.dashboard.live_worker.pack_result', side_effect=lambda result, swap: result):
            for timestamp in range(0, 4000, 200):
                result = predictor.frame(self.feature(.2 + (.0001 if timestamp % 400 else 0)), timestamp)
                self.assertEqual(result['status'], 'waiting')
        self.assertEqual(samples, [])

    def test_capture_keeps_the_short_lead_in_before_hands_enter_the_frame(self):
        predictor, samples = self.predictor('gesture')
        with patch('fsl_trainer.dashboard.live_worker.pack_result', side_effect=lambda result, swap: result):
            for timestamp in (0, 200, 400):
                predictor.frame(np.zeros(128), timestamp)
            for timestamp in (600, 800, 1000, 1200):
                predictor.frame(self.feature(.1 + (timestamp - 600) / 10000), timestamp)
            for timestamp in (1400, 1600, 1800, 2000):
                result = predictor.frame(self.feature(.16), timestamp)
            self.assertEqual(result['status'], 'prediction')
            self.assertEqual(samples[0][0, 0, 126], 0)
            self.assertEqual(samples[0][0, -1, 126], 1)

    def test_brief_missing_hands_preserve_capture_and_lowered_hands_finish_it(self):
        predictor, samples = self.predictor('gesture')
        with patch('fsl_trainer.dashboard.live_worker.pack_result', side_effect=lambda result, swap: result):
            for timestamp in range(0, 1200, 200):
                predictor.frame(self.feature(.1 + timestamp / 10000), timestamp)
            self.assertEqual(predictor.frame(np.zeros(128), 1200)['status'], 'capturing')
            self.assertEqual(predictor.frame(self.feature(.22), 1400)['status'], 'capturing')
            self.assertEqual(predictor.frame(self.feature(.25), 1600)['status'], 'capturing')
            for timestamp in (1800, 2000, 2200, 2400):
                result = predictor.frame(np.zeros(128), timestamp)
            self.assertEqual(result['status'], 'prediction')
            self.assertEqual(len(samples), 1)
            self.assertAlmostEqual(float(samples[0][0, 0, 0]), .1)
            self.assertTrue(np.any(samples[0][0, :, 126] == 0))
            absent = samples[0][0, :, 126] == 0
            np.testing.assert_array_equal(samples[0][0, absent, :63], 0)

    def test_long_or_interrupted_capture_does_not_classify_a_truncated_gesture(self):
        predictor, samples = self.predictor('gesture')
        with patch('fsl_trainer.dashboard.live_worker.pack_result', side_effect=lambda result, swap: result):
            for timestamp in range(0, 6400, 200):
                result = predictor.frame(self.feature(.1 + timestamp / 10000), timestamp)
            self.assertEqual(result['status'], 'incomplete')
            self.assertEqual(samples, [])
            predictor.reset()
            for timestamp in (6600, 6800, 7000):
                predictor.frame(self.feature((timestamp - 6600) / 10000), timestamp)
            result = predictor.frame(self.feature(.4), 9000)
            self.assertEqual(result['status'], 'waiting')
            self.assertEqual(samples, [])

    def test_jpeg_decode_and_rejected_images(self):
        from PIL import Image
        stream = io.BytesIO()
        Image.new('RGB', (4, 3), 'red').save(stream, format='JPEG')
        encoded = 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode()
        self.assertEqual(decode_frame(encoded).shape, (3, 4, 3))
        for value in ('invalid', 'data:image/jpeg;base64,%%%', encoded + 'x' * 900000):
            with self.subTest(value=value[:30]), self.assertRaises(ValueError):
                decode_frame(value)

    def test_loading_rejects_missing_metadata_and_different_hand_asset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = root / 'preprocessing.json'
            metadata.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'no camera extraction'):
                load_predictor(root, None, 'pose')
            asset = root / 'hand.task'
            asset.write_bytes(b'fixture')
            metadata.write_text(json.dumps({'extraction': {'detection_confidence': .7,
                                                          'swap_hands': True, 'model_sha256': 'wrong'}}))
            with patch('fsl_trainer.dashboard.live_worker.Extractor') as factory:
                factory.return_value.model = asset
                with self.assertRaisesRegex(ValueError, 'differs from training'):
                    load_predictor(root, asset, 'pose')
                factory.assert_called_once_with(model=asset, confidence=.7, swap_hands=True)
                factory.return_value.close.assert_called_once()
