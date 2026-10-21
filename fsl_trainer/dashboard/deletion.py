"""Remove saved artifacts from the workspace by moving them into local trash."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets

from ..data.dataset_paths import prepared_datasets

# A legacy category folder can also contain named datasets. Move only its own
# generated artifacts so those sibling datasets remain usable.
LEGACY_ARTIFACTS = ('manifest.json', 'preprocessing.json', 'augmentation_summary.json',
                    'video_augmentation_report.json', 'extraction_report.json', 'extracted_sources.json',
                    'train', 'validation', 'test', 'extracted', '_training_cache', 'paper_report')


def checked_path(base, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Choose an existing saved model or prepared dataset.')
    base = Path(base).resolve()
    path = base / value
    resolved = path.resolve()
    if base not in resolved.parents:
        raise ValueError('Deletion must stay inside the project folder.')
    # Reject linked paths even when they resolve to another in-project artifact.
    current = path.absolute()
    while current != base:
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise ValueError('Linked deletion targets are unsupported.')
        if current.parent == current:
            raise ValueError('Invalid deletion target.')
        current = current.parent
    return resolved


def delete_artifact(base, data):
    """Validate an exact saved target, preserve sources, and return its recovery path."""
    base = Path(base).resolve()
    kind = data.get('kind')
    target = checked_path(base, data.get('path'))
    if kind == 'model':
        relative = target.relative_to(base)
        if (len(relative.parts) != 3 or relative.parts[0] != 'runs'
                or not (target / 'classes.json').is_file()):
            raise ValueError('Choose an existing saved model run.')
        items = [target]
    elif kind == 'dataset':
        root = checked_path(base, data.get('datasets', 'datasets_augmented'))
        sources = checked_path(base, data.get('data', 'data'))
        if (root == sources or sources in root.parents or root in sources.parents
                or root.parts[len(base.parts)] in {'data', 'runs', 'exports', 'reports'}
                or root.parts[len(base.parts)].startswith('.')):
            raise ValueError('Choose a separate prepared-dataset root; source folders cannot be deleted here.')
        relative = target.relative_to(root) if root in target.parents else None
        if relative is None or len(relative.parts) not in (1, 2):
            raise ValueError('Choose a prepared dataset from the selected dataset root.')
        category = root / relative.parts[0]
        prepared = {p.resolve() for p in prepared_datasets(category)}
        if target not in prepared:
            raise ValueError('Choose an existing prepared dataset.')
        items = ([target / name for name in LEGACY_ARTIFACTS
                  if (target / name).exists() and target / name not in prepared]
                 if target == category else [target])
    else:
        raise ValueError('Choose model or dataset for deletion.')
    for item in items:
        checked_path(base, item.relative_to(base).as_posix())
    # Never recursively erase files. A receipt makes manual recovery unambiguous.
    trash_root = checked_path(base, '.trash')
    trash = trash_root / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + secrets.token_hex(6))
    payload = trash / 'files'
    payload.mkdir(parents=True)
    receipt = {'kind': kind, 'original_path': target.relative_to(base).as_posix(),
               'deleted_at': datetime.now(timezone.utc).isoformat(), 'items': []}
    moved = []
    try:
        for item in items:
            destination = payload / item.name
            os.replace(item, destination)
            moved.append((item, destination))
            receipt['items'].append({'original': item.relative_to(base).as_posix(), 'stored': 'files/' + item.name})
        (trash / 'deletion.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        if kind == 'model':
            from .model_records import remove_record
            # Frozen prediction tests retain their model metadata as paper evidence.
            remove_record(base, receipt['original_path'], trash / 'model_record.json')
    except Exception:
        for original, stored in reversed(moved):
            os.replace(stored, original)
        receipt['status'] = 'failed_rolled_back'
        try:
            (trash / 'deletion.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        except OSError:
            pass
        raise
    return {'ok': True, 'trash': trash.relative_to(base).as_posix(), 'path': receipt['original_path']}
