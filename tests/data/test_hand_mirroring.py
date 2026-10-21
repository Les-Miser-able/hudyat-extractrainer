"""Opposite-hand augmentation through the transform and preparation interfaces."""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from fsl_trainer.data.augmentation import augment_sequence, configuration
from fsl_trainer.data.preprocessing import IMAGE, HAND_RELATIVE, mirror_sequence
from fsl_trainer.paths import PROJECT_ROOT
from fsl_trainer.training import trainer


def right_hand(length=9, offset=0):
    value = np.zeros((length, 128), np.float32)
    points = value[:, 63:126].reshape(length, 21, 3)
    points[:, :, 0] = np.linspace(.1, .7, length)[:, None] + np.arange(21)[None, :] * .002 + offset
    points[:, :, 1] = np.linspace(.2, .4, 21)
    points[:, :, 2] = -.05
    value[:, 127] = 1
    return value


class HandMirrorTests(unittest.TestCase):
    def test_whole_sequence_reflects_x_and_swaps_slots_flags_and_missing_frames(self):
        source = right_hand()
        source[2] = 0
        source[4, :63] = source[4, 63:126] * .5
        source[4, 126] = 1
        original = source.copy()
        for normalization in (IMAGE, HAND_RELATIVE):
            with self.subTest(normalization=normalization):
                result = mirror_sequence(source, normalization)
                self.assertEqual(result.dtype, np.float32)
                np.testing.assert_array_equal(source, original)
                np.testing.assert_array_equal(result[:, 126:], source[:, [127, 126]])
                for hand in range(2):
                    before = source[:, (1 - hand) * 63:(2 - hand) * 63].reshape(-1, 21, 3)
                    after = result[:, hand * 63:(hand + 1) * 63].reshape(-1, 21, 3)
                    present = result[:, 126 + hand] == 1
                    expected_x = 1 - before[present, :, 0] if normalization == IMAGE else -before[present, :, 0]
                    np.testing.assert_array_equal(after[present, :, 0], expected_x)
                    np.testing.assert_array_equal(after[present, :, 1:], before[present, :, 1:])
                    self.assertTrue((after[~present] == 0).all())
                np.testing.assert_allclose(mirror_sequence(result, normalization), source, atol=1e-7)
                self.assertGreater(source[-1, 63], source[0, 63])
                self.assertLess(result[-1, 0], result[0, 0])

    def test_mirror_pairs_include_original_and_every_timing_variant_without_changing_timing_stream(self):
        source = right_hand(90)
        for normalization in (IMAGE, HAND_RELATIVE):
            timing, _ = augment_sequence(source, 'A/right.mp4', 42, 3, .05)
            samples, detail = augment_sequence(source, 'A/right.mp4', 42, 3, .05,
                                               mirror_hands=True, normalization=normalization)
            self.assertEqual(len(samples), 7)
            self.assertEqual(detail['requested_extra_samples'], 7)
            self.assertEqual(detail['generated_timing_samples'], 3)
            self.assertEqual(detail['generated_mirrored_samples'], 4)
            self.assertEqual(detail['unfilled_extra_samples'], 0)
            for original, retained in zip(timing, samples[:3]):
                np.testing.assert_array_equal(original[0], retained[0])
                self.assertEqual(original[1], retained[1])
            parents = [trainer.resample(source), *[sample for sample, _ in timing]]
            for index, (sample, transform) in enumerate(samples[3:]):
                self.assertEqual(transform['kind'], 'hand_mirror')
                self.assertEqual(transform['mirror_of_variant'], index)
                np.testing.assert_array_equal(sample, mirror_sequence(parents[index], normalization))

    def test_mirroring_works_without_timing_and_symmetric_duplicates_are_reported(self):
        source = right_hand()
        settings = configuration(0, .05, 42, mirror_hands=True)
        self.assertEqual(settings['algorithm'], 'hand_mirroring')
        samples, detail = augment_sequence(source, 'right', 42, 0, .05, mirror_hands=True)
        self.assertEqual(len(samples), 1)
        self.assertEqual(detail['generated_mirrored_samples'], 1)
        symmetric = right_hand()
        symmetric[:, 63:126:3] = .5
        symmetric[:, :63] = symmetric[:, 63:126]
        symmetric[:, 126] = 1
        samples, detail = augment_sequence(symmetric, 'both', 42, 0, .05, mirror_hands=True)
        self.assertEqual(samples, [])
        self.assertEqual(detail['mirrored_duplicate_candidates'], 1)
        self.assertEqual(detail['unfilled_mirrored_samples'], 1)
        self.assertEqual(detail['unfilled_extra_samples'], 1)

    def test_invalid_controls_and_invalid_features_fail_explicitly(self):
        for value in (1, 0, 'true', None):
            with self.assertRaises(ValueError):
                configuration(0, .05, 42, mirror_hands=value)
        for value in (right_hand()[:, :127], np.full((3, 128), np.nan), right_hand().astype(complex)):
            with self.assertRaises(ValueError):
                mirror_sequence(value)
        bad = right_hand()
        bad[:, 126] = .5
        with self.assertRaises(ValueError):
            mirror_sequence(bad)
        bad = right_hand()
        bad[0, 0] = .2
        with self.assertRaises(ValueError):
            mirror_sequence(bad)


class HandMirrorPreparationTests(unittest.TestCase):
    def make_sources(self, root):
        source = root / 'raw'
        for label_index, label in enumerate(('A', 'B')):
            (source / label).mkdir(parents=True)
            for index in range(3):
                np.save(source / label / f'{index}.npy', right_hand(offset=label_index * .01 + index * .001))
        return source

    def prepare(self, source, output, *, extras=0, mirror=True, report=False):
        args = argparse.Namespace(source=str(source), output=str(output), groups=None, validation=.15,
                                  test=.15, seed=42, static_variants=8, jitter=.0075, normalization=IMAGE,
                                  video_extra_samples=extras, video_time_warp=.05, video_mirror_hands=mirror)
        from fsl_trainer.reporting.paper import save_dataset_report
        def reporting(*args, **kwargs):
            return save_dataset_report(*args, **kwargs) if report else None
        with patch('fsl_trainer.reporting.paper.save_dataset_report', side_effect=reporting), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            trainer.prepare(args)
            return trainer.inspect_dataset(output)[0]

    def test_preparation_pairs_training_only_preserves_sources_and_records_paper_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_sources(root)
            baseline = self.prepare(source, root / 'baseline', mirror=False)
            mirrored = self.prepare(source, root / 'mirrored', extras=3, report=True)
            self.assertEqual(len(baseline), 6)
            self.assertEqual(len(mirrored), 20)
            for base in baseline:
                siblings = [row for row in mirrored if row['source'] == base['source']]
                self.assertEqual(siblings[0], base)
                self.assertEqual(len(siblings), 8 if base['split'] == 'train' else 1)
                self.assertEqual({row['split'] for row in siblings}, {base['split']})
                self.assertEqual({row['group'] for row in siblings}, {base['group']})
                np.testing.assert_array_equal(np.load(root / 'baseline' / base['path']),
                                              np.load(root / 'mirrored' / base['path']))
                for row in siblings:
                    self.assertEqual(row['label'], base['label'])
                    transform = row.get('augmentation', {})
                    if transform.get('kind') == 'hand_mirror':
                        parent = next(r for r in siblings if r['variant'] == transform['mirror_of_variant'])
                        np.testing.assert_array_equal(np.load(root / 'mirrored' / row['path']),
                                                      mirror_sequence(np.load(root / 'mirrored' / parent['path'])))
            report = json.loads((root / 'mirrored/video_augmentation_report.json').read_text())
            summary = json.loads((root / 'mirrored/augmentation_summary.json').read_text())
            self.assertEqual(report['generated_timing_samples'], 6)
            self.assertEqual(report['generated_mirrored_samples'], 8)
            self.assertEqual(report['generated_extra_samples'], 14)
            self.assertEqual(summary['generated_mirrored_samples'], 8)
            for split in ('validation', 'test'):
                self.assertEqual(summary['by_split'][split]['generated_mirrored_samples'], 0)
            self.assertTrue((root / 'mirrored/paper_report/hand_mirroring.csv').is_file())
            self.assertTrue((root / 'mirrored/paper_report/hand_mirroring.png').is_file())
            with (root / 'mirrored/paper_report/hand_mirroring.csv').open(encoding='utf-8-sig', newline='') as handle:
                counts = list(csv.DictReader(handle))
            self.assertEqual([row['Mirrored clean sequences'] for row in counts], ['1', '1'])
            self.assertEqual([row['Mirrored timing sequences'] for row in counts], ['3', '3'])
            from fsl_trainer.training.cache import open_cache, batches
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                packed, _, _, _ = open_cache(root / 'mirrored')
                training = np.concatenate([x for x, _ in batches(packed, 'train', 3)])
            self.assertEqual(len(training), 16)
            self.assertEqual(sum(sample[:, 126].all() for sample in training), 8)
            self.assertEqual(sum(sample[:, 127].all() for sample in training), 8)

    def test_cli_and_category_cli_enable_mirroring_without_timing_extras(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_sources(root / 'data')
            commands = [
                ([sys.executable, '-B', str(PROJECT_ROOT / 'trainer.py'), 'prepare', '--source', str(source),
                  '--output', str(root / 'direct'), '--video-mirror-hands'], root / 'direct'),
                ([sys.executable, '-B', str(PROJECT_ROOT / 'categories.py'), 'prepare', '--data', str(root / 'data'),
                  '--category', 'raw', '--datasets', str(root / 'datasets'), '--dataset-name', 'mirror',
                  '--video-mirror-hands'], root / 'datasets/raw/mirror')]
            for command, output in commands:
                result = subprocess.run(command, capture_output=True, text=True, timeout=45)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(json.loads((output / 'manifest.json').read_text())), 8)
                self.assertTrue(json.loads((output / 'preprocessing.json').read_text())['video_augmentation']['mirror_hands'])

    def test_raw_video_provenance_and_hand_relative_trajectories_are_preserved(self):
        from fsl_trainer.data import extraction
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / 'raw'
            for label_index, label in enumerate(('A', 'B')):
                (raw / label).mkdir(parents=True)
                for index in range(3):
                    (raw / label / f'{index}.mp4').write_text(str(label_index * 3 + index))
            asset = root / 'hand.task'
            asset.write_bytes(b'fixture')
            class FakeExtractor:
                def __init__(self, **options):
                    self.model, self.mp = asset, SimpleNamespace(__version__='fixture')
                    self.confidence, self.swap_hands = .5, False
                def extract(self, path):
                    value = right_hand(offset=int(path.read_text()) * .001)
                    return value, {'kind': 'video', 'frames': len(value), 'detected_frames': len(value), 'fps': 30}
                def close(self):
                    pass
            args = argparse.Namespace(source=str(raw), output=str(root / 'dataset'), groups=None,
                                      validation=.15, test=.15, seed=42, static_variants=8, jitter=.03,
                                      normalization=HAND_RELATIVE, video_extra_samples=0, video_time_warp=.05,
                                      video_mirror_hands=True)
            with patch.object(extraction, 'Extractor', FakeExtractor), \
                    patch('fsl_trainer.reporting.paper.save_dataset_report'), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                trainer.prepare(args)
                rows = trainer.inspect_dataset(root / 'dataset')[0]
            self.assertEqual(len(rows), 8)
            for row in rows:
                self.assertTrue(row['source'].endswith('.mp4'))
                transform = row.get('augmentation', {})
                if transform.get('kind') == 'hand_mirror':
                    parent = next(r for r in rows if r['source'] == row['source'] and r['variant'] == 0)
                    before = np.load(root / 'dataset' / parent['path'])
                    after = np.load(root / 'dataset' / row['path'])
                    np.testing.assert_array_equal(after[:, :63:3], -before[:, 63:126:3])
                    self.assertEqual(transform['normalization'], HAND_RELATIVE)
            report = json.loads((root / 'dataset/video_augmentation_report.json').read_text())
            self.assertTrue(all(item['source'].endswith('.mp4') for item in report['sources']))


if __name__ == '__main__':
    unittest.main()
