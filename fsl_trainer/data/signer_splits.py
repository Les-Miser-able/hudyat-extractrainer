"""Signer holdouts from gesture/signer/recording paths, shared by all entry points."""
from pathlib import PurePosixPath
import json

STRATEGIES = ('grouped', 'loso')


def options(strategy='grouped', test_signer=None, validation_signer=None):
    if strategy not in STRATEGIES:
        raise ValueError('Choose grouped or loso as the split strategy.')
    values = []
    for value in (test_signer, validation_signer):
        if value in (None, ''):
            values.append(None)
        elif (not isinstance(value, str) or value != value.strip() or value.startswith('.')
              or any(c in value for c in '/\\:*?"<>|') or any(ord(c) < 32 for c in value)):
            raise ValueError('Signer IDs must be individual folder names.')
        else:
            values.append(value)
    test_signer, validation_signer = values
    if strategy == 'loso' and not test_signer:
        raise ValueError('Choose a held-out test signer for leave-one-signer-out preparation.')
    if strategy == 'grouped' and any(values):
        raise ValueError('Signer holdouts require the loso split strategy.')
    if test_signer and test_signer == validation_signer:
        raise ValueError('Test and validation signers must be different.')
    return strategy, test_signer, validation_signer


def signer_from_source(source):
    parts = PurePosixPath(source.replace('\\', '/')).parts
    if len(parts) < 3 or any(part in ('.', '..', '/') or part.startswith('.') for part in parts):
        raise ValueError(f'{source}: LOSO needs gesture/signer/recording paths, for example blue/S1/clip.mp4.')
    options('loso', parts[1])
    return parts[1]


def split_signers(records, test_signer, validation_signer=None):
    """Hold out one test signer and a separate validation signer across every sign."""
    options('loso', test_signer, validation_signer)
    signers = sorted({row['signer'] for row in records})
    if len(signers) < 3:
        raise ValueError('LOSO needs at least three signers: training, validation, and held-out test.')
    if len({s.casefold() for s in signers}) != len(signers):
        raise ValueError('Signer folder names must use consistent capitalization across gestures.')
    if test_signer not in signers:
        raise ValueError(f'Test signer {test_signer} not found. Available: {", ".join(signers)}')
    # Rotate deterministically, independent of seed or augmentation settings.
    validation_signer = validation_signer or signers[(signers.index(test_signer) + 1) % len(signers)]
    if validation_signer not in signers:
        raise ValueError(f'Validation signer {validation_signer} not found.')
    protocol = {'strategy': 'leave_one_signer_out', 'signer_identity': 'gesture/signer/recording subfolder',
                'test_signer': test_signer, 'validation_signer': validation_signer,
                'train_signers': [s for s in signers if s not in (test_signer, validation_signer)],
                'all_signers': signers, 'evaluation': 'one clean sequence per source; training-only augmentation'}
    assignment = {}
    coverage = {split: set() for split in ('train', 'validation', 'test')}
    for row in records:
        split = ('test' if row['signer'] == test_signer else
                 'validation' if row['signer'] == validation_signer else 'train')
        if row['group'] in assignment and assignment[row['group']] != split:
            raise ValueError('A source group crosses signer holdouts. Use signer-specific group IDs.')
        assignment[row['group']] = split
        coverage[split].add(row['label'])
    labels = {row['label'] for row in records}
    for split, present in coverage.items():
        if labels != present:
            raise ValueError(f'LOSO {split} is missing gestures: {", ".join(sorted(labels - present))}. '
                             'Record every gesture for the held-out and validation signers, and the training signers.')
    return assignment, protocol


def validate_protocol(manifest, protocol):
    """Audit on both file and cached training paths before loading TensorFlow."""
    if not protocol:
        if any('signer' in row for row in manifest):
            raise ValueError('Signer-tagged data is missing its evaluation protocol.')
        return
    if not isinstance(protocol, dict) or protocol.get('strategy') != 'leave_one_signer_out':
        raise ValueError('Unsupported evaluation protocol.')
    test, validation = protocol.get('test_signer'), protocol.get('validation_signer')
    options('loso', test, validation)
    if not validation:
        raise ValueError('LOSO protocol needs a recorded validation signer.')
    seen, group_splits, classes = {}, {}, {row['label'] for row in manifest}
    coverage = {split: set() for split in ('train', 'validation', 'test')}
    sources = set()
    for row in manifest:
        signer = row.get('signer')
        if signer != signer_from_source(row['source']):
            raise ValueError('Signer identity differs from the source subfolder.')
        expected = 'test' if signer == test else 'validation' if signer == validation else 'train'
        if row['split'] != expected:
            raise ValueError(f'Signer leakage: {signer} must belong to {expected}.')
        for key in ('group', 'source', 'variant_of'):
            identity = (key, row[key])
            if identity in group_splits and group_splits[identity] != expected:
                raise ValueError(f'LOSO leakage: {key} spans splits.')
            group_splits[identity] = expected
        if expected != 'train':
            if row.get('synthetic') is not False or row.get('variant') != 0 or row['source'] in sources:
                raise ValueError('LOSO validation/test must contain exactly one clean sequence per source.')
            sources.add(row['source'])
        seen[signer] = expected
        coverage[expected].add(row['label'])
    if (sorted(seen) != protocol.get('all_signers') or
            sorted(s for s in seen if seen[s] == 'train') != protocol.get('train_signers')):
        raise ValueError('LOSO signer membership differs from its recorded protocol.')
    for split, present in coverage.items():
        if not present or present != classes:
            raise ValueError(f'LOSO {split} must contain every gesture.')


def add_arguments(parser):
    parser.add_argument('--split-strategy', choices=STRATEGIES, default='grouped',
                        help='Source-group split (default) or leave-one-signer-out (loso)')
    parser.add_argument('--test-signer', help='LOSO test signer folder, for example S1')
    parser.add_argument('--validation-signer', help='Separate LOSO validation signer; default is next signer alphabetically')


def command_options(args):
    strategy, test, validation = options(getattr(args, 'split_strategy', 'grouped'),
                                         getattr(args, 'test_signer', None),
                                         getattr(args, 'validation_signer', None))
    if strategy == 'grouped':
        return []
    return ['--split-strategy', strategy, '--test-signer', test] + (
        ['--validation-signer', validation] if validation else [])


def fold_datasets(category_root, name):
    """Select a complete, consistent fold series; never mix independently prepared studies."""
    from .dataset_paths import prepared_datasets, dataset_name
    prefix = dataset_name(name) + '-loso-'
    paths = [p for p in prepared_datasets(category_root) if p.name.startswith(prefix)]
    if not paths:
        raise ValueError(f'No LOSO folds found for {name}. Prepare with --loso-all first.')
    reference, expected, seen = None, None, set()
    for path in paths:
        metadata = json.loads((path / 'preprocessing.json').read_text(encoding='utf-8'))
        manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
        protocol = metadata.get('evaluation_protocol')
        if not protocol:
            raise ValueError(f'{path.name}: not a LOSO dataset.')
        validate_protocol(manifest, protocol)
        signer = protocol['test_signer']
        if path.name != prefix + signer or signer in seen:
            raise ValueError('LOSO fold names must uniquely match their held-out test signers.')
        seen.add(signer)
        # Class/source identities, fingerprints, groups, and preparation settings
        # must match; only the held-out/validation signer may vary between folds.
        originals = sorted({(r['source'], r['label'], r['group'], r['sha256'], r['static']) for r in manifest})
        settings = {k: v for k, v in metadata.items() if k != 'evaluation_protocol'}
        basis = (originals, settings, protocol['all_signers'])
        if reference is None:
            reference, expected = basis, set(protocol['all_signers'])
        elif reference != basis:
            raise ValueError('LOSO folds have different sources or preprocessing settings. Prepare a consistent fold series.')
    if seen != expected:
        raise ValueError('LOSO fold series is incomplete. Missing test signers: ' + ', '.join(sorted(expected - seen)))
    return paths
