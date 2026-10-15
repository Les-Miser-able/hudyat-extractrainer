"""Prepare an isolated HUDYAT camera benchmark; never write to the HUDYAT checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fsl_trainer.dashboard.model_exports import matching_hand_model, _convert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--project', default=str(ROOT.parent / 'HUDYAT'))
    parser.add_argument('--session', required=True)
    args = parser.parse_args()
    if len(args.session) != 16 or any(c not in '0123456789abcdef' for c in args.session):
        parser.error('Invalid camera session identifier.')
    run = (ROOT / args.run).resolve()
    if (ROOT / 'runs').resolve() not in run.parents:
        parser.error('Choose a saved run inside this trainer.')
    project = Path(args.project).resolve()
    if not (project / 'client/src/features/fsl-recognition/category-recognition.tsx').is_file():
        parser.error('Choose the HUDYAT project folder containing client/src/features/fsl-recognition.')
    output = ROOT / 'exports/hudyat-timing' / args.session
    if output.exists():
        parser.error('Camera session output already exists.')
    detector = matching_hand_model(ROOT, run)
    output.mkdir(parents=True)
    config = {'session': args.session, 'run': run.relative_to(ROOT).as_posix(),
              'category': run.parent.name, 'name': run.name,
              'classes': json.loads((run / 'classes.json').read_text(encoding='utf-8')),
              'checkpoint_sha256': hashlib.sha256((run / 'best.keras').read_bytes()).hexdigest()}
    (output / 'session.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    _convert(ROOT / 'pwa-sample/tools/export_model.py', ['--category', run.parent.name, '--run', str(run),
             '--hand-model', str(detector), '--site', str(output / 'model-assets')])
    subprocess.run(['node', str(ROOT / 'tools/build_hudyat_timing.mjs'), str(project), str(output), args.session],
                   check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    print('Open the camera test from Activity. Perform the signs on the target browser/device, then save results.')


if __name__ == '__main__':
    main()
