"""Train one previously selected rate per optimizer and plot epoch accuracies."""
import argparse
import json
from datetime import datetime
from pathlib import Path

# Reuse the original dataset, architecture and training loop.
from run_experiment import train, load_data, tf, np, pd, plt

ROOT = Path(__file__).resolve().parent

# Best tested rates by MEAN VALIDATION ACCURACY at epoch 40 across seeds 42-61.
# Source: results/20261006-093705-631888/history.csv (test accuracy was not used).
# These values directly control training; edit this dictionary to change rates.
BEST_BASE_RATES = {
    "sgd": 1.0,
    "adam": 0.01,
    "mag": 10.0,
    "mag_floor": 1.0,
    "inverse_mag": 0.01,
    "adagrad_norm": 1.0,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, help='best_final_rates.json from a dataset-specific search')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 62)))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Epochs and batch size must be positive; seeds must be unique.')
    if any(not np.isfinite(rate) or rate <= 0 for rate in BEST_BASE_RATES.values()):
        parser.error('Base learning rates must be finite and positive.')
    selection = {
        "source_results": "results/20261006-093705-631888",
        "selection_metric": "Highest mean validation accuracy at epoch 40 across all 20 seeds; ties use smaller rate; test results not used",
        "rate_source": "BEST_BASE_RATES in run_best_rates.py",
        "selected": [{"method": method, "base_lr": rate}
                     for method, rate in BEST_BASE_RATES.items()],
    }
    if args.selection:
        selection = json.loads(args.selection.read_text())
        args.epochs = selection['epochs']
        args.batch_size = selection['batch_size']
        args.seeds = selection['seeds']
    dataset = selection.get('dataset_key', 'iris')
    if dataset not in {'iris', 'digits'}:
        parser.error('Unsupported dataset in selection file')
    output = args.output or ROOT / ('results_best_rates' if dataset == 'iris' else 'results_digits_best_rates') / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True, exist_ok=False)
    tf.config.experimental.enable_op_determinism()
    x, y, train_idx, val, test, scaler = load_data(dataset)
    np.savez(output / 'split_indices.npz', train=train_idx, validation=val, test=test)
    if scaler is not None:
        np.savez(output / 'scaler.npz', mean=scaler.mean_, scale=scaler.scale_)
    config = dict(selection, epochs=args.epochs, batch_size=args.batch_size, seeds=args.seeds,
                  dataset='sklearn Iris' if dataset == 'iris' else 'sklearn digits', dataset_key=dataset, split_counts=[len(train_idx), len(val), len(test)],
                  architecture=[x.shape[1], 16, y.shape[1]], tensorflow=tf.__version__, numpy=np.__version__,
                  reporting='End-of-epoch accuracy on each full partition; arithmetic mean across all requested seeds')
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    histories, runs = [], []
    for candidate in selection['selected']:
        method, rate = candidate['method'], candidate['base_lr']
        for seed in args.seeds:
            result, history, _ = train(method, rate, seed, args,
                (x[train_idx], x[val], y[train_idx], y[val]), test_data=(x[test], y[test]))
            runs.append(result)
            histories.extend(history)
            pd.DataFrame(runs).to_csv(output / 'runs.csv', index=False)
            pd.DataFrame(histories).to_csv(output / 'history.csv', index=False)
            print(f'{method:12s} rate={rate:g} seed={seed}: {result["status"]}', flush=True)
            if result['status'] != 'ok' or len(history) != args.epochs:
                raise RuntimeError(f'{method}, seed {seed} failed. Details saved in {output}. '
                                   'Stopped rather than plotting an incomplete mean.')
    frame = pd.DataFrame(histories)
    metrics = ['train_accuracy', 'val_accuracy', 'test_accuracy']
    means = frame.groupby(['method', 'base_lr', 'epoch'], sort=False)[metrics].mean().reset_index()
    means.to_csv(output / 'mean_accuracy_by_epoch.csv', index=False)
    means[means.epoch == args.epochs].to_csv(output / 'final_accuracy.csv', index=False)
    plots = [('train_accuracy', 'Training', 'training_accuracy.png'),
             ('val_accuracy', 'Validation', 'validation_accuracy.png'),
             ('test_accuracy', 'Test', 'test_accuracy.png')]
    for metric, title, filename in plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        for index, candidate in enumerate(selection['selected']):
            method, rate = candidate['method'], candidate['base_lr']
            curve = means[means.method == method].sort_values('epoch')
            ax.plot(curve.epoch, curve[metric], label=f'{method} (base rate={rate:g})',
                    linestyle=['-', '--', '-.', ':', '--', '-.'][index % 6],
                    marker=['o', 's', '^', 'D', 'v', 'x'][index % 6],
                    markevery=max(1, args.epochs // 8), markersize=4)
        ax.set(xlabel='Epoch', ylabel=f'Mean {title.lower()} accuracy', ylim=(0, 1.05),
               title=f'{title} accuracy: mean of {len(args.seeds)} runs')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)
    print(means[means.epoch == args.epochs].to_string(index=False))
    print(f'Three plots saved in: {output.resolve()}')


if __name__ == '__main__':
    main()
