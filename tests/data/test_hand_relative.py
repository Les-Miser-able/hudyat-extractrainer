"""Hand scale invariance, preserved motion, and legacy category isolation."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from fsl_trainer.data.preprocessing import (
    IMAGE, HAND_RELATIVE, normalize_sequence, augment_pose, preparation_normalization,
)
from fsl_trainer.data.sequences import resample
from fsl_trainer.dashboard.live_worker import LivePredictor
from fsl_trainer.training import trainer


def pose():
    frame = np.zeros((1, 128), np.float32)
    points = frame[0, :63].reshape(21, 3)
    points[:, 0] = np.linspace(.4, .6, 21)
    points[:, 1] = np.linspace(.8, .3, 21)
    points[:, 2] = np.linspace(0., -.05, 21)
    points[0] = [.4, .8, 0.]
    points[9] = [.4, .6, -.01]
    frame[0, 126] = 1
    return frame


class HandRelativeTests(unittest.TestCase):
    def test_pose_is_relative_to_wrist_and_pixel_palm_length(self):
        value = normalize_sequence(pose(), 640, 480)
        np.testing.assert_allclose(value[0, :3], 0, atol=1e-6)
        np.testing.assert_allclose(value[0, 27:30], [0, -1, -.01 * 640 / 96], atol=1e-6)
        np.testing.assert_array_equal(value[0, 63:126], 0)
        np.testing.assert_array_equal(value[0, 126:], [1, 0])

    def test_position_scale_and_aspect_ratio_do_not_change_features(self):
        source = pose()
        expected = normalize_sequence(source, 640, 480)
        for width, height, scale in [(640, 480, .5), (1280, 720, .8), (480, 640, 1.5)]:
            changed = source.copy()
            points = changed[0, :63].reshape(21, 3)
            pixels = points.astype(np.float64) * [640, 480, 640]
            pixels = (pixels - pixels[0]) * scale + [width * .25, height * .7, 0]
            points[:] = pixels / [width, height, width]
            np.testing.assert_allclose(normalize_sequence(changed, width, height), expected, atol=2e-6)

    def test_motion_is_preserved_instead_of_centering_every_frame(self):
        source = np.repeat(pose(), 4, axis=0)
        points = source[:, :63].reshape(4, 21, 3)
        points[:, :, 0] += np.linspace(0, .15, 4)[:, None]
        normalized = normalize_sequence(source, 640, 480)
        np.testing.assert_allclose(normalized[:, 0], np.linspace(0, 1, 4), atol=2e-6)
        self.assertGreater(float(np.ptp(normalized[:, 0])), .99)
        for frame in normalized[1:]:
            np.testing.assert_allclose(frame[3:63], normalized[0, 3:63], atol=2e-6)
        translated = source.copy()
        translated[:, :63].reshape(4, 21, 3)[:, :, :2] += [.1, -.2]
        np.testing.assert_allclose(normalize_sequence(translated, 640, 480), normalized, atol=2e-6)

    def test_changing_hand_scale_keeps_finger_shape_stable(self):
        source = np.repeat(pose(), 4, axis=0)
        for i, scale in enumerate([.5, .75, 1., 1.5]):
            points = source[i, :63].reshape(21, 3)
            points[:] = points[0] + scale * (points - points[0])
        normalized = normalize_sequence(source, 640, 480)
        for frame in normalized[1:]:
            np.testing.assert_allclose(frame[3:63], normalized[0, 3:63], atol=2e-6)
        np.testing.assert_allclose(normalized[:, :3], 0, atol=1e-6)

    def test_missing_and_degenerate_hands_stay_zero(self):
        source = np.concatenate([np.zeros((1, 128), np.float32), pose(), pose()])
        source[2, :63] = .3
        normalized = normalize_sequence(source, 640, 480)
        np.testing.assert_array_equal(normalized[[0, 2]], 0)
        self.assertEqual(normalized[1, 126], 1)
        for size in [(0, 480), (640, float('nan'))]:
            with self.assertRaises(ValueError):
                normalize_sequence(source, *size)

    def test_augmentation_is_deterministic_and_preserves_missing_slots(self):
        frame = normalize_sequence(pose(), 640, 480)
        first = augment_pose(frame, np.random.default_rng(9))
        np.testing.assert_array_equal(first, augment_pose(frame, np.random.default_rng(9)))
        self.assertTrue(np.isin(first[:, 126:], [0, 1]).all())
        for hand in range(2):
            if not first[0, 126 + hand]:
                np.testing.assert_array_equal(first[:, hand * 63:(hand + 1) * 63], 0)

    def test_automatic_normalization_is_limited_to_raw_alphabet(self):
        self.assertEqual(preparation_normalization('auto', 'alphabet', True), HAND_RELATIVE)
        for category, media in [('alphabet', False), ('colors', True), ('greetings', True), ('numbers', True), ('family', True), ('shapes', True)]:
            self.assertEqual(preparation_normalization('auto', category, media), IMAGE)
        self.assertEqual(preparation_normalization(IMAGE, 'alphabet', True), IMAGE)
        with self.assertRaises(ValueError):
            preparation_normalization('unknown', 'alphabet', True)

    def test_active_legacy_preparation_matches_existing_byte_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in ('A', 'J'):
                folder = root / 'raw' / label
                folder.mkdir(parents=True)
                for index in range(5):
                    length = 1 if label == 'A' else 9
                    value = np.zeros((length, 128), dtype=np.float32)
                    value[:, :63] = np.linspace(.1, .8, 63) + index * .02
                    value[:, 126] = 1
                    if label == 'J':
                        value[:, 0] = np.linspace(0, .7 + index * .02, 9)
                    np.save(folder / f'{index}.npy', value)
            args = argparse.Namespace(source=str(root / 'raw'), output=str(root / 'dataset'),
                                      groups=None, validation=.15, test=.15, seed=42,
                                      static_variants=8, jitter=.0075)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                trainer.prepare(args)
            actual = {p.relative_to(root / 'dataset').as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (root / 'dataset').rglob('*')
                      if p.is_file() and p.relative_to(root / 'dataset').parts[0] != 'paper_report'}
            expected = json.loads((Path(__file__).parents[1] / 'fixtures/preparation_hashes.json').read_text())
            self.assertEqual(actual, expected)
            self.assertTrue((root / 'dataset/paper_report/index.html').is_file())

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for browser feature parity')
    def test_browser_and_python_preprocessing_agree(self):
        root = Path(__file__).resolve().parents[2]
        source = np.repeat(pose(), 7, axis=0)
        source[:, :63].reshape(7, 21, 3)[:, :, 0] += np.linspace(0, .1, 7)[:, None]
        source[2] = 0
        source[5, :63] = .3  # a degenerate detected hand must be removed
        cases = []
        for value in [pose(), source, np.zeros((2, 128), np.float32)]:
            for width, height in [(640, 480), (1280, 720), (480, 640)]:
                cases.append({'frames': value.tolist(), 'width': width, 'height': height})
        script = ('import {readFileSync} from "node:fs"; import {preprocessFrames} from '
                  + json.dumps((root / 'pwa-sample/dist/core.js').as_uri()) + ';'
                  'const cases=JSON.parse(readFileSync(0,"utf8"));'
                  'console.log(JSON.stringify(cases.map(c=>Array.from(preprocessFrames(c.frames,c.width,c.height,'
                  '{normalization:"hand-relative-sequence-v1"})))));')
        result = subprocess.run(['node', '--input-type=module', '-e', script], input=json.dumps(cases),
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        for case, actual in zip(cases, json.loads(result.stdout)):
            expected = resample(normalize_sequence(np.array(case['frames']), case['width'], case['height']))
            np.testing.assert_allclose(np.array(actual).reshape(32, 128), expected, atol=2e-6)

    def test_live_pose_uses_run_normalization_and_legacy_input_is_unchanged(self):
        rgb = np.zeros((480, 640, 3), np.uint8)
        for normalization in [IMAGE, HAND_RELATIVE]:
            samples = []
            def model(value, training):
                samples.append(value.copy())
                return np.array([[.8, .2]])
            extractor = NS(swap_hands=False, image=lambda image: image, close=lambda: None,
                           image_detector=NS(detect=lambda image: image))
            predictor = LivePredictor(model, ['A', 'B'], extractor, 'pose', normalization)
            with patch('fsl_trainer.dashboard.live_worker.pack_result', return_value=pose()[0]):
                response = predictor.frame(rgb, 100)
            self.assertEqual(response['status'], 'prediction')
            expected = normalize_sequence(pose(), 640, 480) if normalization == HAND_RELATIVE else pose()
            np.testing.assert_array_equal(samples[0][0], resample(expected))
            predictor.close()

    def test_raw_image_extraction_and_live_pose_produce_the_same_features(self):
        from PIL import Image, ImageOps
        from fsl_trainer.data.extraction import Extractor
        raw = pose()[0]
        points = [NS(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                  for p in raw[:63].reshape(21, 3)]
        result = NS(hand_landmarks=[points], handedness=[[NS(category_name='Left', score=.9)]])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'pose.png'
            Image.new('RGB', (640, 480), 'white').save(path)
            extractor = Extractor.__new__(Extractor)
            extractor.Image, extractor.ImageOps = Image, ImageOps
            extractor.image = lambda rgb: rgb
            extractor.image_detector = NS(detect=lambda image: result, close=lambda: None)
            extractor.swap_hands, extractor.normalization = False, HAND_RELATIVE
            sequence, details = extractor.extract(path)
            self.assertEqual(details['kind'], 'image')
            captured = []
            def model(value, training):
                captured.append(value[0].copy())
                return np.array([[1., 0.]])
            predictor = LivePredictor(model, ['A', 'B'], extractor, 'pose', HAND_RELATIVE)
            self.assertEqual(predictor.frame(np.zeros((480, 640, 3), np.uint8), 100)['status'], 'prediction')
            np.testing.assert_array_equal(captured[0], resample(sequence))
            predictor.close()

    def test_live_loader_selects_feature_contract_and_rejects_unknown_versions(self):
        from fsl_trainer.dashboard import live_worker
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / 'hand.task'
            asset.write_bytes(b'fixture')
            metadata = {'normalization': HAND_RELATIVE,
                        'extraction': {'detection_confidence': .5, 'swap_hands': False,
                                       'model_sha256': hashlib.sha256(asset.read_bytes()).hexdigest()}}
            (root / 'preprocessing.json').write_text(json.dumps(metadata))
            (root / 'classes.json').write_text('["A", "B"]')
            model = NS(input_shape=(None, 32, 128), output_shape=(None, 2))
            with patch.object(live_worker, 'Extractor') as factory, patch.object(live_worker, 'tensorflow') as tensorflow:
                factory.return_value.model = asset
                tensorflow.return_value.keras.models.load_model.return_value = model
                predictor = live_worker.load_predictor(root, asset, 'pose')
                self.assertEqual(predictor.normalization, HAND_RELATIVE)
                factory.assert_called_once_with(model=asset, confidence=.5, swap_hands=False)
                predictor.close()
            metadata['normalization'] = 'future-unknown-version'
            (root / 'preprocessing.json').write_text(json.dumps(metadata))
            with patch.object(live_worker, 'Extractor') as factory, self.assertRaisesRegex(ValueError, 'normalization'):
                live_worker.load_predictor(root, asset, 'pose')
            factory.assert_not_called()

    def test_live_motion_preserves_capture_path_and_normalizes_after_capture(self):
        from fsl_trainer.dashboard.gesture_capture import GestureCapture
        captured = []
        def model(value, training):
            captured.append(value[0].copy())
            return np.array([[.8, .2]])
        extractor = NS(swap_hands=False, image=lambda image: image, close=lambda: None,
                       detector=lambda video: NS(detect_for_video=lambda image, timestamp: image,
                                                 close=lambda: None))
        predictor = LivePredictor(model, ['J', 'Z'], extractor, 'motion', HAND_RELATIVE)
        reference = GestureCapture()
        complete = None
        with patch('fsl_trainer.dashboard.live_worker.pack_result') as pack:
            for timestamp in range(0, 2400, 100):
                frame = pose()[0].copy()
                frame[:63].reshape(21, 3)[:, 0] += min(timestamp, 1000) / 5000
                pack.return_value = frame
                expected = reference.push(frame, timestamp)
                if expected['status'] == 'complete':
                    complete = expected['frames']
                predictor.frame(np.zeros((480, 640, 3), np.uint8), timestamp)
        self.assertEqual(len(captured), 1)
        times = np.array([item[0] for item in complete])
        positions = np.interp(np.linspace(times[0], times[-1], 32), times, np.arange(len(times)))
        expected = resample(normalize_sequence(np.stack([item[1] for item in complete]), 640, 480), positions)
        np.testing.assert_array_equal(captured[0], expected)
        self.assertGreater(float(np.ptp(captured[0][:, 0])), .5)
        predictor.close()

    def test_new_preparation_keeps_heldout_originals_and_augments_training_only(self):
        from fsl_trainer.data import extraction
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / 'raw'
            for label in ('A', 'J'):
                folder = raw / label
                folder.mkdir(parents=True)
                for index in range(5):
                    (folder / f'{index}.jpg' if label == 'A' else folder / f'{index}.mp4').write_text(str(index))
            asset = root / 'hand.task'
            asset.write_bytes(b'fixture')
            class FakeExtractor:
                def __init__(self, **options):
                    self.model, self.mp = asset, NS(__version__='fixture')
                    self.confidence, self.swap_hands = .5, False
                    self.normalization = options['normalization']
                def extract(self, path):
                    value = pose()
                    value[0, 3] += int(path.stem) * .01
                    kind = 'image'
                    if path.suffix == '.mp4':
                        kind = 'video'
                        value = np.repeat(value, 9, axis=0)
                        value[:, :63].reshape(9, 21, 3)[:, :, 0] += np.linspace(0, .2, 9)[:, None]
                    value = normalize_sequence(value, 640, 480)
                    return value, {'kind': kind, 'frames': len(value), 'detected_frames': len(value)}
                def close(self):
                    pass
            args = argparse.Namespace(source=str(raw), output=str(root / 'dataset'), groups=None,
                                      validation=.15, test=.15, seed=42, static_variants=8, jitter=None,
                                      normalization=HAND_RELATIVE, video_extra_samples=2, video_time_warp=.05)
            with patch.object(extraction, 'Extractor', FakeExtractor), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                trainer.prepare(args)
                rows, _, _ = trainer.inspect_dataset(root / 'dataset')
            self.assertEqual(args.jitter, .03)
            metadata = json.loads((root / 'dataset/preprocessing.json').read_text())
            self.assertEqual(metadata['normalization'], HAND_RELATIVE)
            reverse = {v: k for k, v in json.loads((root / 'dataset/extracted_sources.json').read_text()).items()}
            for source in {r['source'] for r in rows}:
                siblings = [r for r in rows if r['source'] == source]
                split = siblings[0]['split']
                self.assertEqual(len({r['split'] for r in siblings}), 1)
                self.assertFalse(siblings[0]['synthetic'])
                expected = resample(np.load(root / 'dataset/extracted' / reverse[source]))
                np.testing.assert_array_equal(np.load(root / 'dataset' / siblings[0]['path']), expected)
                self.assertEqual(len(siblings), (8 if source.startswith('A/') else 3) if split == 'train' else 1)
                self.assertTrue(all(r['synthetic'] for r in siblings[1:]))

    def test_hand_relative_rejects_unknown_dimension_arrays_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in ['A', 'J']:
                folder = root / 'raw' / label
                folder.mkdir(parents=True)
                np.save(folder / '0.npy', pose())
            args = argparse.Namespace(source=str(root / 'raw'), output=str(root / 'out'), groups=None,
                                      validation=.15, test=.15, seed=42, static_variants=8,
                                      jitter=None, normalization=HAND_RELATIVE)
            with self.assertRaisesRegex(ValueError, 'raw images/videos'):
                trainer.prepare(args)
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
