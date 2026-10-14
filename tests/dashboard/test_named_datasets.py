"""Named dataset discovery and command routing, without ML dependencies."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dashboard


class NamedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.patch = patch.object(dashboard, 'BASE', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        for label in ('blue', 'red'):
            folder = self.root / 'data' / 'colors' / label / 'S1'
            folder.mkdir(parents=True)
            for i in range(3):
                (folder / f'{i}.mp4').touch()

    def prepared(self, category, version=None, count=10):
        path = self.root / 'datasets_augmented' / category
        if version is not None:
            path /= version
        path.mkdir(parents=True, exist_ok=True)
        (path / 'manifest.json').write_text('[]')
        (path / 'augmentation_summary.json').write_text(json.dumps({'total_output_sequences': count}))
        return path

    def test_prepare_targets_named_folder_and_keeps_preprocessing_options(self):
        request = {'action': 'prepare', 'category': 'colors', 'dataset_name': 'videos v2',
                   'source_folders': ['S1'], 'videos_only': True, 'video_extra_samples': 3,
                   'video_time_warp': .05, 'validation': .2, 'test': .1, 'variants': 4,
                   'jitter': .006, 'confidence': .6, 'seed': 10, 'swap_hands': True}
        command = dashboard.build_command(request)
        expected = self.root / 'datasets_augmented' / 'colors' / 'videos v2'
        self.assertEqual(command[command.index('--output') + 1], str(expected))
        self.assertEqual(command[command.index('--source') + 1], str(self.root / 'data' / 'colors'))
        for flag, value in [('--validation', '.2'), ('--test', '.1'), ('--static-variants', '4'),
                            ('--jitter', '.006'), ('--detection-confidence', '.6'), ('--seed', '10'),
                            ('--source-folder', 'S1'), ('--video-extra-samples', '3'), ('--video-time-warp', '.05')]:
            actual = command[command.index(flag) + 1]
            if flag == '--source-folder':
                self.assertEqual(actual, value)
            else:
                self.assertEqual(float(actual), float(value))
        self.assertIn('--videos-only', command)
        self.assertIn('--swap-hands', command)
        self.assertFalse(expected.exists())

    def test_inventory_lists_named_and_original_datasets_with_separate_summaries(self):
        original = self.prepared('colors', count=4)
        first = self.prepared('colors', 'v1', 10)
        second = self.prepared('colors', 'v2', 20)
        (original / 'unfinished').mkdir()
        (original / 'unfinished' / 'keep.txt').write_text('partial')
        row = next(c for c in dashboard.inventory() if c['name'] == 'colors')
        self.assertTrue(row['prepared'])
        self.assertEqual([d['path'] for d in row['datasets']],
                         [p.relative_to(self.root).as_posix() for p in (original, first, second)])
        self.assertEqual([d['summary']['total_output_sequences'] for d in row['datasets']], [4, 10, 20])
        self.assertTrue(row['datasets'][0]['legacy'])
        self.assertFalse(row['datasets'][1]['legacy'])
        self.assertEqual(row['summary'], {})

    def test_train_and_validate_use_exact_selected_dataset(self):
        first = self.prepared('colors', 'v1')
        second = self.prepared('colors', 'v2')
        for action in ('train', 'inspect'):
            for path in (first, second):
                with self.subTest(action=action, path=path):
                    command = dashboard.build_command({'action': action, 'category': 'colors', 'run': 'new',
                                                       'dataset': path.relative_to(self.root).as_posix()})
                    self.assertEqual(command[command.index('--dataset') + 1], str(path))
                    if action == 'train':
                        self.assertEqual(command[command.index('--output') + 1], str(self.root / 'runs' / 'colors' / 'new'))
        with self.assertRaisesRegex(ValueError, 'multiple prepared datasets'):
            dashboard.build_command({'action': 'train', 'category': 'colors', 'run': 'new'})

    def test_selection_cannot_escape_category_or_selected_root(self):
        own = self.prepared('colors', 'v1')
        other = self.prepared('alphabet', 'v1')
        for selection in ('', None, '../outside', own.parent.relative_to(self.root).as_posix(),
                          other.relative_to(self.root).as_posix(), 'datasets_augmented/colors/missing'):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                dashboard.build_command({'action': 'inspect', 'category': 'colors', 'dataset': selection})
        with self.assertRaises(ValueError):
            dashboard.build_command({'action': 'inspect', 'category': 'colors', 'datasets': 'other_datasets',
                                     'dataset': own.relative_to(self.root).as_posix()})

    def test_invalid_names_and_partial_outputs_are_rejected_but_siblings_are_allowed(self):
        for name in (None, '', '.', '..', '../v1', 'a/b', 'a\\b', 'name.', ' name', 'CON', 'nul.txt', False):
            with self.subTest(name=name), self.assertRaises(ValueError):
                dashboard.build_command({'action': 'prepare', 'category': 'colors', 'dataset_name': name})
        partial = self.root / 'datasets_augmented' / 'colors' / 'partial'
        partial.mkdir(parents=True)
        (partial / 'keep.txt').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            dashboard.build_command({'action': 'prepare', 'category': 'colors', 'dataset_name': 'partial'})
        command = dashboard.build_command({'action': 'prepare', 'category': 'colors', 'dataset_name': 'fresh'})
        self.assertIn(str(partial.parent / 'fresh'), command)
        self.assertEqual((partial / 'keep.txt').read_text(), 'keep')

    def test_single_original_dataset_still_works_without_a_selector(self):
        original = self.prepared('colors')
        command = dashboard.build_command({'action': 'inspect', 'category': 'colors'})
        self.assertIn(str(original), command)


if __name__ == '__main__':
    unittest.main()
