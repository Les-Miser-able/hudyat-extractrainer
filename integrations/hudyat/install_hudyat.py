"""Import an extracted HUDYAT model bundle, preserving other category releases."""
import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .runtime_support import inside, runtime_changes, apply_changes
else:
    from runtime_support import inside, runtime_changes, apply_changes


def import_bundle(bundle, project):
    server_package = inside(project, 'server/package.json')
    if 'models:register' not in json.loads(server_package.read_text(encoding='utf-8')).get('scripts', {}):
        raise ValueError('Choose the HUDYAT project folder containing client/ and server/.')
    source = bundle / 'model-assets'
    catalog = json.loads((source / 'catalog.json').read_text(encoding='utf-8'))
    entries = [e for e in catalog['categories'] if e.get('available')]
    if len(entries) != 1:
        raise ValueError('This bundle must contain exactly one model release.')
    entry = entries[0]
    category, version = entry['id'], entry['version']
    if (not isinstance(category, str) or not category
            or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in category)
            or not isinstance(version, str) or not version or any(c not in '0123456789abcdef' for c in version)):
        raise ValueError('Invalid category or model version.')
    assets = inside(project, 'server/model-assets')
    changes = runtime_changes(project)
    required_paths = {
        'modelUrl': f'models/{category}/{version}/model.json',
        'weightsUrl': f'models/{category}/{version}/weights.bin',
        'handUrl': 'shared/' + entry['handSha256'] + '.task',
    }
    for url, digest in (('modelUrl', 'modelSha256'), ('weightsUrl', 'weightsSha256'), ('handUrl', 'handSha256')):
        relative = entry[url].removeprefix('./')
        if relative != required_paths[url]:
            raise ValueError('Unexpected model artifact path.')
        path = inside(source, relative)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry[digest]:
            raise ValueError('Model artifact checksum mismatch: ' + relative)
        target = inside(assets, relative)
        if target.exists() and target.read_bytes() != payload:
            raise ValueError('A different artifact already uses this version: ' + relative)
        changes[target] = payload
    metadata = json.loads(changes[inside(assets, required_paths['modelUrl'])])
    if (metadata.get('format') != 'fsl-conv1d-bilstm-v1' or metadata.get('inputShape') != [32, 128]
            or metadata.get('category') != category or metadata.get('version') != version
            or metadata.get('weightSha256') != entry['weightsSha256']
            or metadata.get('preprocessing', {}).get('extraction', {}).get('model_sha256') != entry['handSha256']):
        raise ValueError('The model metadata does not match its release.')
    normalization = metadata.get('preprocessing', {}).get('normalization', 'image')
    if normalization not in ('image', 'hand-relative-sequence-v1'):
        raise ValueError('Unsupported model preprocessing.')
    for name in ('parity.json', 'camera-preprocessing-parity.json'):
        relative = f'models/{category}/{version}/{name}'
        path = inside(source, relative)
        if path.is_file():
            changes[inside(assets, relative)] = path.read_bytes()
    catalog_path = inside(assets, 'catalog.json')
    previous = json.loads(catalog_path.read_text(encoding='utf-8')) if catalog_path.exists() else {'categories': []}
    merged = {e['id']: e for e in previous['categories']}
    merged[category] = entry
    previous['categories'] = sorted(merged.values(), key=lambda e: e['name'])
    changes[catalog_path] = (json.dumps(previous, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    # All source, integrity, and compatibility checks finish before any target is written.
    apply_changes(changes)
    print('Imported ' + entry['name'] + ' ' + version)
    print('Activate in HUDYAT/server with: npm run models:register')
    print('Then rebuild/restart HUDYAT and re-enter the category online.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--runtime-only', action='store_true', help='Update camera preprocessing without importing models')
    args = parser.parse_args()
    try:
        if args.runtime_only:
            apply_changes(runtime_changes(args.project.resolve()))
        else:
            import_bundle(Path(__file__).resolve().parent, args.project.resolve())
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
