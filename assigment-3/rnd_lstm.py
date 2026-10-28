"""Exercise 2: Reuters topic classification with 1, 2 and 3 LSTM layers."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

EPOCHS = 15
BATCH_SIZE = 64
SEED = 42


def load_data():
    # Keep the original Keras test set; reserve 12.5% of the other 80% for validation.
    # Load the dataset
    # Only considering the top 10,000 most frequently occurring words
    (x, y), (x_test, y_test) = keras.datasets.reuters.load_data(num_words=10000)
    #split the data to training and validation sets
    train, validation = train_test_split(
        np.arange(len(y)), test_size=0.125, random_state=SEED, stratify=y
    )
    datasets = {}
    for name, sequences, labels in (
        ("train", x[train], y[train]),
        ("validation", x[validation], y[validation]),
        ("test", x_test, y_test),
    ):
        # News articles usually introduce the topic early, so retain the first 20 tokens.
        padded = keras.utils.pad_sequences(
            sequences, maxlen=20, padding="post", truncating="post"
        )
        datasets[name] = (padded, labels)
    return datasets, train, validation


def build_model(depth):
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    model = keras.Sequential([
        keras.Input(shape=(20,), dtype="int32"),
        keras.layers.Embedding(10000, 64, mask_zero=True),
    ])
    for layer in range(depth):
        model.add(keras.layers.LSTM(64, return_sequences=layer < depth - 1))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(46, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"],
    )
    return model


def plot_results(histories, output):
    # Colour identifies depth; line style identifies the data split (nine curves).
    for metric, ylabel in (("accuracy", "Accuracy"), ("loss", "Cross-entropy loss")):
        fig, ax = plt.subplots(figsize=(11, 6))
        for depth, color in zip((1, 2, 3), ("tab:blue", "tab:orange", "tab:green")):
            if str(depth) not in histories:
                continue
            history = histories[str(depth)]
            for split, style in (("train", "--"), ("validation", "-"), ("test", ":")):
                values = history[f"{split}_{metric}"]
                ax.plot(range(1, len(values) + 1), values, color=color,
                        linestyle=style, marker=".", label=f"{depth} layer(s) - {split}")
        ax.set(xlabel="Epoch", ylabel=ylabel, xlim=(1, EPOCHS),
               title=f"Reuters LSTM with 1, 2 and 3 layers - {ylabel}")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if metric == "accuracy":
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
        fig.tight_layout()
        fig.savefig(output / f"{metric}.png", dpi=160)
        plt.close(fig)


def main():
    # A new folder preserves previous results whenever the script is rerun.
    output = Path(__file__).parent / "results_reuters_lstm" / "20tokens"
    output.mkdir(parents=True)
    datasets, train, validation = load_data()
    np.savez(output / "split_indices.npz", train=train, validation=validation)
    config = dict(epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED,
                  dataset_seed=113, vocabulary=10000, sequence_length=20,
                  embedding_size=64, units_per_layer=64, layers=[1, 2, 3],
                  dropout=0.2, learning_rate=0.001, clipnorm=1.0,
                  split_sizes={s: len(y) for s, (_, y) in datasets.items()},
                  selection_metric="validation_loss")
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Results: {output}\nSplit sizes: {config['split_sizes']}", flush=True)
    histories, summary = {}, {}
    for depth in (1, 2, 3):
        model = build_model(depth)
        history = {f"{s}_{m}": [] for s in datasets for m in ("loss", "accuracy")}
        histories[str(depth)] = history
        best_loss = float("inf")
        for epoch in range(1, EPOCHS + 1):
            print(f"\nLSTM {depth} layer(s) - epoch {epoch}/{EPOCHS}", flush=True)
            model.fit(*datasets["train"], batch_size=BATCH_SIZE, epochs=1,
                      shuffle=True, verbose=2)
            # End-of-epoch evaluation disables dropout for all three comparable curves.
            # Test curves are descriptive only: never use them to select layers/epochs.
            for split, (x, y) in datasets.items():
                scores = model.evaluate(x, y, batch_size=BATCH_SIZE,
                                        verbose=0, return_dict=True)
                for metric in ("loss", "accuracy"):
                    history[f"{split}_{metric}"].append(float(scores[metric]))
                print(f"{split}: loss={scores['loss']:.4f}, accuracy={scores['accuracy']:.4f}", flush=True)
            if history["validation_loss"][-1] < best_loss:
                best_loss = history["validation_loss"][-1]
                model.save(output / f"lstm_{depth}_layers_best.keras")
                summary[str(depth)] = dict(best_epoch=epoch, validation_loss=best_loss,
                                          validation_accuracy=history["validation_accuracy"][-1])
            (output / "history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
            (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            plot_results(histories, output)

    best_depth = min(summary, key=lambda d: summary[d]["validation_loss"])
    model = keras.models.load_model(output / f"lstm_{best_depth}_layers_best.keras")
    x_test, y_test = datasets["test"]
    scores = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0, return_dict=True)
    predictions = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0).argmax(axis=1)
    selected = dict(layers=int(best_depth), **summary[best_depth],
                    test_accuracy=float(scores["accuracy"]), test_loss=float(scores["loss"]),
                    test_macro_f1=float(f1_score(y_test, predictions, labels=np.arange(46),
                                                 average="macro", zero_division=0)))
    (output / "selected_model.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"\nSelected using validation loss only:\n{json.dumps(selected, indent=2)}")
    print(f"Saved accuracy.png and loss.png to {output}", flush=True)


if __name__ == "__main__":
    main()
