"""Create a HUDYAT model ZIP without changing HUDYAT's files or database."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fsl_trainer.dashboard.model_exports import write_hudyat_bundle
from fsl_trainer.dashboard.server import runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--hand-model')
    args = parser.parse_args()
    try:
        output = write_hudyat_bundle(ROOT, args.run, runs(), args.output, args.hand_model)
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + '\n')
    print('HUDYAT bundle ready: ' + str(output), flush=True)
    print('Extract the ZIP and follow HUDYAT_IMPORT.md to import and register the release.', flush=True)


if __name__ == '__main__':
    main()
