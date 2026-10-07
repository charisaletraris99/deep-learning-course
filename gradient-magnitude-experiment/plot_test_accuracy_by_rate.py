"""One test-accuracy plot per optimizer, one mean curve per tested base rate."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from run_experiment import load_data, train, tf, np, pd, plt

ROOT = Path(__file__).resolve().parent


def plot_curves(frame, config, output):
    """Require every configured seed at every epoch; never average partial runs."""
    keys = ['method', 'base_lr', 'seed', 'epoch']
    expected = pd.MultiIndex.from_tuples(
        [(m, float(r), s, e) for m in config['methods'] for r in config['grids'][m]
         for s in config['seeds'] for e in range(1, config['epochs'] + 1)], names=keys)
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate method/rate/seed/epoch records.')
    values = frame.set_index(keys)['test_accuracy'].reindex(expected)
    if values.isna().any() or not values.between(0, 1).all():
        raise ValueError('Missing or invalid test accuracy: all seeds and epochs are required. Inspect runs.csv.')
    means = values.groupby(level=['method', 'base_lr', 'epoch']).agg(['mean', 'std', 'count']).reset_index()
    means.to_csv(output / 'mean_test_accuracy_by_epoch.csv', index=False)
    for method in config['methods']:
        fig, ax = plt.subplots(figsize=(10, 6))
        for index, rate in enumerate(config['grids'][method]):
            curve = means[(means.method == method) & (means.base_lr == rate)].sort_values('epoch')
            ax.plot(curve.epoch, curve['mean'], label=f'Base rate = {rate:g}',
                    linestyle=['-', '--', '-.', ':'][index % 4])
        ax.set(xlabel='Epoch', ylabel='Mean test accuracy', ylim=(0, 1.05),
               title=f"{method} | {config['dataset']} | mean of {len(config['seeds'])} runs")
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f'{method}_test_accuracy.png', dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new output folder for plots and CSV files")
    parser.add_argument("--results", type=Path, required=True, help="Search results folder containing config.json and history.csv")
    args = parser.parse_args()
    source = args.results
    config = json.loads((source / 'config.json').read_text())
    if 'grids' not in config:
        parser.error('Use a run_experiment search folder, not a run_best_rates folder.')
    frame = pd.read_csv(source / 'history.csv')
    output = args.output or ROOT / 'results_rate_plots' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True, exist_ok=False)
    config = dict(config, source_results=str(source.resolve()))
    print(f'Source: {source.resolve()}', flush=True)
    if 'test_accuracy' not in frame.columns:
        dataset = config.get('dataset_key', 'iris' if config['dataset'] == 'sklearn Iris' else 'digits')
        x, y, tr, va, te, _ = load_data(dataset)
        if config['architecture'] != [x.shape[1], 16, y.shape[1]]:
            raise ValueError('Source architecture differs from current model; cannot reproduce these curves.')
        saved = np.load(source / 'split_indices.npz')
        for name, indices in [('train', tr), ('validation', va), ('test', te)]:
            np.testing.assert_array_equal(saved[name], indices, err_msg='Source split does not match current loader')
        config['test_curve_origin'] = 'Recomputed with current code using source settings; not recovered from old checkpoints'
        config['rerun_tensorflow'] = tf.__version__
        (output / 'config.json').write_text(json.dumps(config, indent=2))
        print('Per-epoch test accuracy was not saved. Repeating all configured rates and seeds.', flush=True)
        tf.config.experimental.enable_op_determinism()
        histories, runs = [], []
        options = SimpleNamespace(epochs=config['epochs'], batch_size=config['batch_size'])
        for method in config['methods']:
            for rate in config['grids'][method]:
                for seed in config['seeds']:
                    result, history, _ = train(method, rate, seed, options,
                        (x[tr], x[va], y[tr], y[va]), test_data=(x[te], y[te]))
                    runs.append(result)
                    histories.extend(history)
                    pd.DataFrame(runs).to_csv(output / 'runs.csv', index=False)
                    pd.DataFrame(histories).to_csv(output / 'history.csv', index=False)
                    print(f'{method} rate={rate:g} seed={seed}: {result["status"]}', flush=True)
        frame = pd.DataFrame(histories)
    else:
        config['test_curve_origin'] = 'Saved per-epoch test measurements from source history.csv'
        frame.to_csv(output / 'history.csv', index=False)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    plot_curves(frame, config, output)
    print(f'Plots saved in: {output.resolve()}')


if __name__ == '__main__':
    main()
