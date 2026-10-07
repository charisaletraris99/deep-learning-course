"""Run the digits rate search, then create three accuracy plots with its chosen rates."""
from pathlib import Path
from datetime import datetime
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 62)))
    args = parser.parse_args()
    output = ROOT / 'results_digits' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    subprocess.run([sys.executable, str(ROOT / 'run_experiment.py'), '--dataset', 'digits',
                    '--epochs', str(args.epochs), '--seeds', *map(str, args.seeds),
                    '--output', str(output)], check=True)
    selection = output / 'best_final_rates.json'
    if not selection.exists():
        raise RuntimeError('A complete rate selection was not available. Inspect the search runs.csv.')
    subprocess.run([sys.executable, str(ROOT / 'run_best_rates.py'),
                    '--selection', str(selection)], check=True)
