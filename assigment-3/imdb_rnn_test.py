"""Exercise 1, Question 4: full run of the selected three-layer GRU.

The filename is retained for convenience; the selected architecture is now GRU.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save graphs without opening blocking windows.
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from tensorflow import keras


def build_model():
    model = keras.Sequential([
        keras.Input(shape=(500,), dtype="int32"),
        keras.layers.Embedding(10000, 32, mask_zero=True),
        keras.layers.GRU(32, return_sequences=True),
        keras.layers.GRU(32, return_sequences=True),
        keras.layers.GRU(32),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    num_epochs = 35
    keras.utils.set_random_seed(42)
    output = Path(__file__).parent / "results_gru_3_layers"
    output.mkdir(exist_ok=True)

    # Combine all 50,000 reviews, then create balanced, separate 70/10/20 sets.
    (x_original, y_original), (x_original_test, y_original_test) = keras.datasets.imdb.load_data(num_words=10000)
    x = np.concatenate([x_original, x_original_test])
    y = np.concatenate([y_original, y_original_test])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    # 12.5% of the remaining 80% is 10% of the full dataset.
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.125, random_state=42, stratify=y_train
    )

    # Keep the last 500 tokens, like the assignment; append zeros and mask them.
    x_train, x_val, x_test = [
        keras.utils.pad_sequences(x, maxlen=500, padding="post", truncating="pre")
        for x in (x_train, x_val, x_test)
    ]
    print(f"Train: {x_train.shape}; validation: {x_val.shape}; test: {x_test.shape}", flush=True)

    model = build_model()
    model.summary()
    config = dict(model="GRU", recurrent_layers=3, units_per_layer=32,
                  embedding_size=32, vocabulary=10000, max_length=500,
                  epochs=num_epochs, batch_size=128, learning_rate=0.001,
                  clipnorm=1.0, seed=42, split=[35000, 5000, 10000],
                  selection_metric="validation_loss",
                  comparison_run="20261015_150230_305523")
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Evaluate all sets with the same end-of-epoch weights for comparable curves.
    # Test scores are recorded for the requested graph, never used to tune training.
    datasets = {"train": (x_train, y_train), "validation": (x_val, y_val), "test": (x_test, y_test)}
    history = {f"{split}_{metric}": [] for split in datasets for metric in ("loss", "accuracy")}
    lowest_val_loss = float("inf")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}", flush=True)
        model.fit(x_train, y_train, epochs=1, batch_size=128, shuffle=True, verbose=2)
        for split, (x, y) in datasets.items():
            scores = model.evaluate(x, y, batch_size=128, verbose=0, return_dict=True)
            for metric in ("loss", "accuracy"):
                history[f"{split}_{metric}"].append(float(scores[metric]))
            print(f"{split}: loss={scores['loss']:.4f}, accuracy={scores['accuracy']:.4f}", flush=True)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if history["validation_loss"][-1] < lowest_val_loss:
            lowest_val_loss = history["validation_loss"][-1]
            model.save(output / "model.keras")  # Best validation-loss checkpoint.

    # The requested two learning curves, plus a final test confusion matrix.
    best_epoch = int(np.argmin(history["validation_loss"])) + 1
    print(f"Lowest validation loss at epoch {best_epoch}. Compare later loss trends for overfitting.")
    epochs = range(1, num_epochs + 1)
    for metric, label in (("accuracy", "Accuracy"), ("loss", "Binary cross-entropy loss")):
        fig, ax = plt.subplots(figsize=(7, 4))
        for split in datasets:
            ax.plot(epochs, history[f"{split}_{metric}"], marker="o", label=split.capitalize())
        ax.set(xlabel="Epoch", ylabel=label, title=f"IMDB GRU (3 layers) - {label}")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ax.axvline(best_epoch, color="gray", linestyle="--", label=f"Lowest validation loss: epoch {best_epoch}")
        if metric == "accuracy":
            ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / f"{metric}.png", dpi=160)
        plt.close(fig)

    model.save(output / "final.keras")  # Preserve epoch 35 separately.
    model = keras.models.load_model(output / "model.keras")
    selected_scores = {split: model.evaluate(x, y, batch_size=128, verbose=0, return_dict=True)
                       for split, (x, y) in datasets.items()}
    (output / "selected_metrics.json").write_text(
        json.dumps(dict(selected_epoch=best_epoch, metrics=selected_scores), indent=2), encoding="utf-8")
    probabilities = model.predict(x_test, batch_size=128, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=[0, 1],
                                           display_labels=["Negative", "Positive"], cmap="Blues",
                                           values_format="d")
    plt.title(f"IMDB GRU (3 layers) - test predictions at selected epoch {best_epoch}")
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close()
    report = classification_report(y_test, predictions, labels=[0, 1],
                                   target_names=["Negative", "Positive"], zero_division=0)
    (output / "classification_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Saved graphs, metrics and model to {output}", flush=True)


if __name__ == "__main__":
    main()
