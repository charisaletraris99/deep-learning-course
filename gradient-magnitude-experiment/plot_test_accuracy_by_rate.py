"""One test-accuracy plot per optimizer, one mean curve per tested base rate."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from optimizers import optimizer_label, floor_for_rate
from run_experiment import load_data, train, tf, np, pd, plt

ROOT = Path(__file__).resolve().parent
# Change this folder to choose the experiment when running from VS Code.
RESULTS_FOLDER = ROOT / "results" / "digits_20261007-213821-729796"


def select_best_test_rates(means, epochs, window=20):
    """Select one rate per optimizer by mean test accuracy in the final window."""
    start = max(1, epochs - window + 1)
    scores = (means[means.epoch.between(start, epochs)]
              .groupby(['method', 'base_lr'], as_index=False)['mean'].mean()
              .rename(columns={'mean': 'mean_test_accuracy_last_20_epochs'}))
    # Equal scores prefer the smaller base rate, independent of grid order.
    selected = (scores.sort_values(['method', 'mean_test_accuracy_last_20_epochs', 'base_lr'],
                                   ascending=[True, False, True])
                .drop_duplicates('method').copy())
    selected['selection_start_epoch'] = start
    selected['selection_end_epoch'] = epochs
    selected['epochs_used'] = epochs - start + 1
    return selected



def plot_mag_floor_values(frame, config, output):
    """Save a separate mean test-accuracy figure for each recorded floor."""
    rows = frame[frame.method == "mag_floor"].copy()
    if rows.empty:
        return
    if "map_floor_rate" not in rows.columns:
        rows["map_floor_rate"] = rows.base_lr.map(lambda rate: floor_for_rate(config, rate))
    if rows["map_floor_rate"].isna().any():
        raise ValueError("Missing map_floor_rate for mag_floor records")
    destination = output / "mag_floor_test_accuracy"
    destination.mkdir(parents=True, exist_ok=True)
    from matplotlib.ticker import MaxNLocator
    for floor, group in rows.groupby("map_floor_rate"):
        means = group.groupby(["base_lr", "epoch"], as_index=False).test_accuracy.mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        for rate, curve in means.groupby("base_lr"):
            curve = curve.sort_values("epoch")
            ax.plot(curve.epoch, curve.test_accuracy, marker="o",
                    markevery=max(1, len(curve) // 10), markersize=4,
                    label=f"Base rate = {rate:g}")
        ax.set(xlabel="Epoch", ylabel="Mean test accuracy", ylim=(0, 1.05),
               title=f"mag_floor (map_floor_rate={floor:g}) | {config['dataset']}")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(destination / f"map_floor_rate_{float(floor)!r}.png", dpi=180)
        plt.close(fig)


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
    plot_mag_floor_values(frame, config, output)
    for method in config['methods']:
        fig, ax = plt.subplots(figsize=(10, 6))
        for index, rate in enumerate(config['grids'][method]):
            curve = means[(means.method == method) & (means.base_lr == rate)].sort_values('epoch')
            ax.plot(curve.epoch, curve['mean'], label=f'{optimizer_label(method, config, rate)} | Base rate = {rate:g}',
                    linestyle=['-', '--', '-.', ':'][index % 4],
                    marker=['o', 's', '^', 'D', 'v', 'x'][index % 6],
                    markersize=5, markevery=max(1, len(curve) // 10))
        ax.set(xlabel='Epoch', ylabel='Mean test accuracy', ylim=(0, 1.05),
               title=f"{optimizer_label(method, config)} | {config['dataset']} | mean of {len(config['seeds'])} runs")
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f'{method}_test_accuracy.png', dpi=180)
        plt.close(fig)

    selected = select_best_test_rates(means, config['epochs'])
    selected.to_csv(output / 'best_test_rates_last_20_epochs.csv', index=False)
    selected_curves = means.merge(selected[['method', 'base_lr']], on=['method', 'base_lr'])
    selected_curves.to_csv(output / 'best_test_curves_by_epoch.csv', index=False)
    fig, ax = plt.subplots(figsize=(11, 7))
    for index, method in enumerate(config['methods']):
        winner = selected[selected.method == method].iloc[0]
        curve = selected_curves[selected_curves.method == method].sort_values('epoch')
        ax.plot(curve.epoch, curve['mean'],
                label=f"{optimizer_label(method, config, winner.base_lr)} | rate={winner.base_lr:g} | window mean={winner.mean_test_accuracy_last_20_epochs:.2%} | final epoch mean={curve['mean'].iloc[-1]:.2%}",
                linestyle=['-', '--', '-.', ':'][index % 4],
                    marker=['o', 's', '^', 'D', 'v', 'x'][index % 6],
                    markersize=5, markevery=max(1, len(curve) // 10))
    start = max(1, config['epochs'] - 19)
    ax.set(xlabel='Epoch', ylabel='Mean test accuracy', ylim=(0, 1.05),
           title=f"{config['dataset']} | best rate per optimizer\n"
                 f"Selected by mean test accuracy over epochs {start}-{config['epochs']}")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.text(0.5, 0.015, 'Test-selected comparison; test data was used to choose the rates.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output / 'best_models_test_accuracy.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new output folder for plots and CSV files")
    parser.add_argument("--results", type=Path, default=RESULTS_FOLDER,
                        help="Input results folder; defaults to RESULTS_FOLDER above")
    args = parser.parse_args()
    source = args.results
    for filename in ("config.json", "history.csv"):
        if not (source / filename).is_file():
            parser.error(f"Missing {source / filename}. Set RESULTS_FOLDER or use --results with an existing results folder.")
    config = json.loads((source / 'config.json').read_text())
    if 'grids' not in config:
        parser.error('Use a run_experiment search folder, not a run_best_rates folder.')
    frame = pd.read_csv(source / 'history.csv')
    output = args.output or ROOT / 'results_rate_plots' / f'plot_{source.name}'
    if output.exists():
        if args.output:
            parser.error(f"Output folder already exists: {output}. Choose a new --output folder.")
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
                        (x[tr], x[va], y[tr], y[va]), test_data=(x[te], y[te]),
                        map_floor_rate=floor_for_rate(config, rate) if method == "mag_floor" else 1.0)
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
