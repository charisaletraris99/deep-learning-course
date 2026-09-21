"""Exercise 2: a smaller model based on the best configuration from 1(e)."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras


# Keep the best batch-size experiment's optimizer settings, but use
# four hidden neurons and ReLU to reduce computation per example.
HIDDEN_NEURONS = 4
ACTIVATION = "relu"
LEARNING_RATE = 0.02
MOMENTUM = 0.9
BATCH_SIZE = 8
EPOCHS = 40
RUNS = 20
OUTPUT_DIR = Path(__file__).resolve().parent


class TestAccuracyCallback(keras.callbacks.Callback):
    """Evaluate the test set after each epoch without updating weights."""

    def __init__(self, X_test, y_test):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.test_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.model.evaluate(
            self.X_test, self.y_test, verbose=0, return_dict=True
        )
        self.test_accuracies.append(metrics["accuracy"])


def build_model(seed):
    """Start each run with fresh weights and a fresh optimizer."""
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.Input(shape=(4,)),
        keras.layers.Dense(HIDDEN_NEURONS, activation=ACTIVATION),
        keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=LEARNING_RATE, momentum=MOMENTUM
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    # Use the same data preparation and split as Exercise 1.
    iris = load_iris()
    X = iris.data
    encoder = OneHotEncoder()
    y = encoder.fit_transform(iris.target.reshape(-1, 1)).toarray()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(
        f"Hidden neurons={HIDDEN_NEURONS}, activation={ACTIVATION}, "
        f"learning rate={LEARNING_RATE}, momentum={MOMENTUM}, "
        f"batch size={BATCH_SIZE}, epochs={EPOCHS}, runs={RUNS}"
    )
    results = {"training": [], "validation": [], "test": []}
    for run in range(RUNS):
        model = build_model(seed=42 + run)
        if run == 0:
            model.summary()
        callback = TestAccuracyCallback(X_test, y_test)
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            callbacks=[callback],
            verbose=0,
        )
        results["training"].append(history.history["accuracy"])
        results["validation"].append(history.history["val_accuracy"])
        results["test"].append(callback.test_accuracies)
        print(
            f"Run {run + 1}/{RUNS} | "
            f"Training: {results['training'][-1][-1]:.2%} | "
            f"Validation: {results['validation'][-1][-1]:.2%} | "
            f"Test: {results['test'][-1][-1]:.2%}"
        )

    results = {metric: np.array(values) for metric, values in results.items()}
    np.savez_compressed(OUTPUT_DIR / "exercise2_results.npz", **results)
    with (OUTPUT_DIR / "exercise2_summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Metric", "Mean final accuracy (%)",
            "Standard deviation (percentage points)",
        ])
        print(f"\nMean accuracies at epoch {EPOCHS}:")
        for metric, values in results.items():
            final_values = values[:, -1]
            mean = 100 * final_values.mean()
            std = 100 * final_values.std(ddof=1) if RUNS > 1 else 0.0
            writer.writerow([metric, mean, std])
            print(f"{metric.title()}: {mean:.2f}% (standard deviation: {std:.2f} percentage points)")

    # Plot all metrics together, then save each as a separate figure.
    epochs = np.arange(1, EPOCHS + 1)
    for selected in [list(results), ["training"], ["validation"], ["test"]]:
        plt.figure(figsize=(9, 6))
        for metric in selected:
            plt.plot(epochs, results[metric].mean(axis=0), label=metric.title())
        suffix = "" if len(selected) > 1 else f"_{selected[0]}"
        plt.xlabel("Epoch")
        plt.ylabel("Average accuracy")
        plt.title(f"Exercise 2: average of {RUNS} runs")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"exercise2{suffix}_accuracy.png", dpi=300)
    plt.show()
    return results


if __name__ == "__main__":
    main()
