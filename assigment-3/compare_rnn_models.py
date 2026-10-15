"""Compare SimpleRNN, GRU and LSTM with 1, 2 and 3 recurrent layers."""

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

EPOCHS = 15
BATCH_SIZE = 128
SEED = 42
LAYER_TYPES = {"SimpleRNN": keras.layers.SimpleRNN, "GRU": keras.layers.GRU, "LSTM": keras.layers.LSTM}


def load_data():
    (x1, y1), (x2, y2) = keras.datasets.imdb.load_data(num_words=10000)
    x, y = np.concatenate([x1, x2]), np.concatenate([y1, y2])
    # Split indices once so every model uses exactly the same reviews.
    train, test = train_test_split(np.arange(len(y)), test_size=0.2,
                                   random_state=SEED, stratify=y)
    train, val = train_test_split(train, test_size=0.125,
                                 random_state=SEED, stratify=y[train])
    x_train, x_val = [keras.utils.pad_sequences(x[i], maxlen=500, padding="post",
                                               truncating="pre") for i in (train, val)]
    return x_train, y[train], x_val, y[val], {"train": train, "validation": val, "test": test}


def build_model(name, depth):
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    model = keras.Sequential([
        keras.Input(shape=(500,), dtype="int32"),
        keras.layers.Embedding(10000, 32, mask_zero=True),
    ])
    for layer in range(depth):
        # Intermediate layers pass the full sequence to the next recurrent layer.
        model.add(LAYER_TYPES[name](32, return_sequences=(layer < depth - 1)))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def plot_family(name, runs, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for depth, color in zip((1, 2, 3), ("tab:blue", "tab:orange", "tab:green")):
        if str(depth) not in runs:
            continue
        history = runs[str(depth)]["history"]
        epochs = np.arange(1, len(history["val_loss"]) + 1)
        if not len(epochs):
            continue
        for ax, metric in zip(axes, ("accuracy", "loss")):
            ax.plot(epochs, history[f"train_{metric}"], color=color, linestyle="--", marker=".",
                    label=f"{depth} layer(s) train")
            ax.plot(epochs, history[f"val_{metric}"], color=color, marker=".",
                    label=f"{depth} layer(s) validation")
    for ax, label in zip(axes, ("Accuracy", "Binary cross-entropy")):
        ax.set(xlabel="Epoch", ylabel=label, xlim=(1, max(2, EPOCHS)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylim(0, 1)
    complete = sum(run["completed"] for run in runs.values())
    fig.suptitle(f"{name}: 1, 2 and 3 recurrent layers ({complete}/3 experiments complete)")
    fig.tight_layout()
    fig.savefig(output / f"{name.lower()}_comparison.png", dpi=160)
    plt.close(fig)


def save_results(results, output):
    (output / "histories.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = []
    for name, runs in results.items():
        for depth, run in runs.items():
            h = run["history"]
            if not h["val_loss"]:
                continue
            best = int(np.argmin(h["val_loss"]))
            rows.append(dict(model=name, layers=int(depth), parameters=run["parameters"],
                             epochs_completed=len(h["val_loss"]), complete=run["completed"],
                             best_epoch=best + 1, best_val_loss=h["val_loss"][best],
                             val_accuracy_at_best_loss=h["val_accuracy"][best],
                             final_train_accuracy=h["train_accuracy"][-1],
                             final_val_accuracy=h["val_accuracy"][-1]))
    if rows:
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    # Separate run folders preserve earlier experiments when this script is rerun.
    output = Path(__file__).parent / "results_recurrent_comparison" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    print(f"Results: {output}", flush=True)
    x_train, y_train, x_val, y_val, indices = load_data()
    np.savez(output / "split_indices.npz", **indices)
    config = dict(epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED, vocabulary=10000,
                  max_length=500, embedding_size=32, units_per_layer=32,
                  learning_rate=0.001, clipnorm=1.0, train_size=len(y_train),
                  validation_size=len(y_val), test_size=len(indices["test"]),
                  test_used=False, selection_metric="validation binary cross-entropy")
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Training: {len(y_train)}; validation: {len(y_val)}; reserved test: {len(indices['test'])}", flush=True)
    results = {name: {} for name in LAYER_TYPES}
    for name in LAYER_TYPES:
        for depth in (1, 2, 3):
            model = build_model(name, depth)
            run_dir = output / f"{name.lower()}_{depth}_layers"
            run_dir.mkdir()
            history = {key: [] for key in ("train_loss", "train_accuracy", "val_loss", "val_accuracy")}
            run = dict(parameters=model.count_params(), completed=False, history=history)
            results[name][str(depth)] = run
            best_loss = float("inf")
            for epoch in range(EPOCHS):
                print(f"\n{name}, {depth} layer(s), epoch {epoch + 1}/{EPOCHS}", flush=True)
                model.fit(x_train, y_train, epochs=1, batch_size=BATCH_SIZE, shuffle=True, verbose=2)
                # Both curves use the same end-of-epoch weights, as in the original experiment.
                for split, x, y in (("train", x_train, y_train), ("val", x_val, y_val)):
                    scores = model.evaluate(x, y, batch_size=BATCH_SIZE, verbose=0, return_dict=True)
                    for metric in ("loss", "accuracy"):
                        history[f"{split}_{metric}"].append(float(scores[metric]))
                    print(f"{split}: loss={scores['loss']:.4f}, accuracy={scores['accuracy']:.4f}", flush=True)
                if history["val_loss"][-1] < best_loss:
                    best_loss = history["val_loss"][-1]
                    model.save(run_dir / "best.keras")
                save_results(results, output)
                plot_family(name, results[name], output)
            model.save(run_dir / "final.keras")
            run["completed"] = True
            save_results(results, output)
            plot_family(name, results[name], output)
    candidates = [(min(run["history"]["val_loss"]), name, depth)
                  for name, runs in results.items() for depth, run in runs.items()]
    loss, name, depth = min(candidates)
    print(f"\nAll nine experiments finished. Lowest validation loss: {name}, {depth} layer(s), {loss:.4f}.", flush=True)
    print(f"Graphs and results: {output}", flush=True)


if __name__ == "__main__":
    main()
