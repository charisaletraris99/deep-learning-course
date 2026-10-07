"""Reproducible optimizer comparison on five public classification datasets."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import argparse
import json
import platform
from datetime import datetime
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from optimizers import GradientStep
from experiment_datasets import DATASETS, load_additional

# Choose: "iris", "digits", "letter", "mnist", or "fashion_mnist".
DATASET = "digits"

GRIDS = {"sgd": [0.01, 0.1, 1.0, 10.0], "adam": [0.0001, 0.001, 0.01, 0.1],
         "mag": [0.01, 0.1, 1.0, 10.0], "mag_floor": [0.01, 0.1, 1.0,10.0], "inverse_mag": [0.00001, 0.0001, 0.001, 0.01],
         "adagrad_norm": [0.01, 0.1, 1.0, 10.0], "rmsprop": [0.0001, 0.001, 0.01, 0.1],
         "adagrad": [0.01, 0.1, 1.0, 10.0]}


def load_assignment_data():
    """Match exercise1a.py's row order, scaler and Keras validation_split=0.1.

    The original fits scaling on all 120 non-test rows before holding out the
    last 12 for validation. Preserve that behavior for assignment comparability.
    """
    iris = load_iris()
    indices = np.arange(len(iris.target))
    trainval, test = train_test_split(indices, test_size=0.2, random_state=42)
    split_at = int(len(trainval) * 0.9)
    train_idx, val = trainval[:split_at], trainval[split_at:]
    scaler = StandardScaler().fit(iris.data[trainval])
    x = scaler.transform(iris.data).astype("float32")
    y = np.eye(3, dtype="float32")[iris.target]
    return x, y, train_idx, val, test, scaler


def load_data(dataset="iris"):
    if dataset == "iris":
        return load_assignment_data()
    if dataset in {"letter", "mnist", "fashion_mnist"}:
        return load_additional(dataset)
    if dataset != "digits":
        raise ValueError(f"Choose one of {list(DATASETS)}")
    data = load_digits()
    indices = np.arange(len(data.target))
    trainval, test = train_test_split(indices, test_size=0.2, stratify=data.target, random_state=42)
    train_idx, val = train_test_split(trainval, test_size=0.1, stratify=data.target[trainval], random_state=42)
    return ((data.data / 16.0).astype("float32"), np.eye(10, dtype="float32")[data.target],
            train_idx, val, test, None)


def model_for(seed, input_dim=4, classes=3):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    return tf.keras.Sequential([tf.keras.Input((input_dim,)),
                                tf.keras.layers.Dense(16, activation="tanh"),
                                tf.keras.layers.Dense(classes, activation="softmax")])


def evaluate(model, x, y):
    logits = model(x, training=False)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits, from_logits=False))
    accuracy = np.mean(np.argmax(logits.numpy(), axis=1) == np.argmax(y, axis=1))
    return float(loss), float(accuracy)


def train(method, rate, seed, args, data, test_data=None):
    x_train, x_val, y_train, y_val = data
    model = model_for(seed, x_train.shape[1], y_train.shape[1])
    optimizer = GradientStep(method, rate, model.trainable_variables)
    rng = np.random.default_rng(seed)

    @tf.function(reduce_retracing=True)
    def step(x, y):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
                y, model(x, training=True), from_logits=False))
        tf.debugging.assert_all_finite(loss, "Non-finite training loss")
        return optimizer.apply(tape.gradient(loss, model.trainable_variables), model.trainable_variables)

    history, best_weights, best_epoch = [], None, None
    best_loss = float("inf")
    status, error = "ok", ""
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        values = []
        try:
            order = rng.permutation(len(x_train))
            for start in range(0, len(order), args.batch_size):
                indices = order[start:start + args.batch_size]
                values.append([float(v) for v in step(x_train[indices], y_train[indices])])
            train_loss, train_accuracy = evaluate(model, x_train, y_train)
            val_loss, val_accuracy = evaluate(model, x_val, y_val)
            if not np.isfinite([train_loss, val_loss]).all():
                raise FloatingPointError("Non-finite evaluation loss")
        except (tf.errors.InvalidArgumentError, FloatingPointError) as exc:
            status, error = "diverged", str(exc)[:1000]
            break
        mag, scalar, update = np.mean(values, axis=0)
        history.append(dict(method=method, base_lr=rate, seed=seed, epoch=epoch,
                            train_loss=train_loss, train_accuracy=train_accuracy,
                            val_loss=val_loss, val_accuracy=val_accuracy,
                            mean_abs_gradient=mag, scalar_lr=scalar, update_l2=update))
        if test_data is not None:
            test_loss, test_accuracy = evaluate(model, *test_data)
            if not np.isfinite(test_loss):
                status, error = "diverged", "Non-finite test loss"
                history.pop()
                break
            history[-1].update(test_loss=test_loss, test_accuracy=test_accuracy)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_weights = model.get_weights()
    result = dict(method=method, base_lr=rate, seed=seed, status=status,
                  best_val_loss=best_loss, best_epoch=best_epoch,
                  seconds=time.perf_counter() - started, error=error)
    return result, history, best_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), default=DATASET)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 62)))
    parser.add_argument("--methods", nargs="+", choices=list(GRIDS), default=["sgd", "adam", "mag", "mag_floor", "inverse_mag", "adagrad_norm"])
    parser.add_argument("--grid", type=Path, help="JSON mapping methods to positive base-rate lists")
    parser.add_argument("--record-test", action="store_true", help="Save test accuracy at each epoch for all candidates")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.dataset not in DATASETS:
        parser.error(f"DATASET must be one of {list(DATASETS)}")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    grids = GRIDS.copy()
    if args.grid:
        grids.update(json.loads(args.grid.read_text()))
    for method in args.methods:
        if not grids[method] or any(not np.isfinite(v) or v <= 0 for v in grids[method]):
            parser.error(f"Invalid grid for {method}")
    # Prefix the date with the dataset selected at the top of this file.
    results_directory = "results" if args.dataset == "iris" else f"results_{args.dataset}"
    run_folder = f"{args.dataset}_{datetime.now():%Y%m%d-%H%M%S-%f}"
    output = args.output or Path(__file__).parent / results_directory / run_folder
    output.mkdir(parents=True, exist_ok=False)
    tf.config.experimental.enable_op_determinism()
    x, y, train_idx, val, test, scaler = load_data(args.dataset)
    np.savez(output / "split_indices.npz", train=train_idx, validation=val, test=test)
    if scaler is not None:
        np.savez(output / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
    config = dict(record_test=args.record_test, epochs=args.epochs, batch_size=args.batch_size, seeds=args.seeds,
                  methods=args.methods, grids={m: grids[m] for m in args.methods},
                  dataset=DATASETS[args.dataset][0], dataset_key=args.dataset, split_seed=42, epsilon=1e-8, b0=1.0,
                  split_counts={"train": len(train_idx), "validation": len(val), "test": len(test)},
                  preprocessing=DATASETS[args.dataset][1],
                  architecture=[x.shape[1], 16, y.shape[1]], hidden_activation="tanh", output_activation="softmax", tensorflow=tf.__version__,
                  numpy=np.__version__, sklearn=sklearn.__version__, python=platform.python_version())
    (output / "config.json").write_text(json.dumps(config, indent=2))
    runs, histories, selected = [], [], []
    for seed in args.seeds:
        for method in args.methods:
            candidates = []
            for rate in grids[method]:
                result, history, weights = train(method, rate, seed, args,
                    (x[train_idx], x[val], y[train_idx], y[val]),
                    test_data=(x[test], y[test]) if args.record_test else None)
                runs.append(result)
                histories.extend(history)
                if result["status"] == "ok" and weights is not None:
                    candidates.append((result, weights))
                print(f"seed={seed} {method:12s} rate={rate:g} {result['status']} validation={result['best_val_loss']:.4f}", flush=True)
            if candidates:
                winner, weights = min(candidates, key=lambda item: item[0]["best_val_loss"])
                model = model_for(seed, x.shape[1], y.shape[1])
                model.set_weights(weights)
                test_loss, test_accuracy = evaluate(model, x[test], y[test])
                model.save_weights(output / f"{method}-seed{seed}.weights.h5")
                selected.append(dict(winner, test_loss=test_loss, test_accuracy=test_accuracy))
            pd.DataFrame(runs).to_csv(output / "runs.csv", index=False)
            pd.DataFrame(histories).to_csv(output / "history.csv", index=False)
    choices = []
    for method in args.methods:
        candidates = []
        for rate in grids[method]:
            completed = [r for r in runs if r['method'] == method and r['base_lr'] == rate]
            last = [h for h in histories if h['method'] == method and h['base_lr'] == rate and h['epoch'] == args.epochs]
            if len(last) == len(args.seeds) and all(r['status'] == 'ok' for r in completed):
                candidates.append((float(np.mean([h['val_accuracy'] for h in last])), rate))
        if candidates:
            accuracy, rate = min(candidates, key=lambda item: (-item[0], item[1]))
            choices.append(dict(method=method, base_lr=rate, mean_final_val_accuracy=accuracy))
    if len(choices) == len(args.methods):
        export = dict(config, source_results=str(output.resolve()), selected=choices,
                      selection_metric=f'Highest mean validation accuracy at epoch {args.epochs}; ties use smaller rate; test not used')
        (output / 'best_final_rates.json').write_text(json.dumps(export, indent=2))
    if not selected:
        print(f"All runs diverged. Inspect {output / 'runs.csv'}")
        return
    selected_frame = pd.DataFrame(selected)
    selected_frame.to_csv(output / "selected.csv", index=False)
    summary = selected_frame.groupby("method").agg(
        successful_seeds=("seed", "count"), test_accuracy_mean=("test_accuracy", "mean"),
        test_accuracy_std=("test_accuracy", "std"), test_loss_mean=("test_loss", "mean"))
    summary["requested_seeds"] = len(args.seeds)
    summary.to_csv(output / "summary.csv")
    history_frame = pd.DataFrame(histories)
    chosen = history_frame.merge(selected_frame[["method", "base_lr", "seed"]], on=["method", "base_lr", "seed"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metrics = ["train_loss", "val_loss", "val_accuracy", "mean_abs_gradient", "scalar_lr", "update_l2"]
    for axis, metric in zip(axes.flat, metrics):
        for method, group in chosen.groupby("method"):
            mean = group.groupby("epoch")[metric].mean()
            axis.plot(mean.index, mean.values, label=method)
        axis.set(xlabel="Epoch", ylabel=metric)
        if metric != "val_accuracy":
            axis.set_yscale("symlog", linthresh=1e-8)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Validation-selected base rates; curves averaged across successful seeds")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    print(summary.to_string())
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
