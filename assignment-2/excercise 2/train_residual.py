"""Exercise 2: add residual shortcuts to the Exercise 1 Fashion-MNIST CNN."""

from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf

import prepare_data as data

EPOCHS = 40
# Larger batches reduce updates per epoch for the larger Fashion MNIST dataset.
BATCH_SIZE = 128

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "exercise_2_residual_cnn_gelu"


@tf.keras.utils.register_keras_serializable(package="Assignment2")
class ReplicatePadding2D(tf.keras.layers.Layer):
    """Extend the nearest edge value by one pixel on every spatial side."""

    def call(self, inputs):
        x = tf.concat([inputs[:, :1, :, :], inputs, inputs[:, -1:, :, :]], axis=1)
        return tf.concat([x[:, :, :1, :], x, x[:, :, -1:, :]], axis=2)

    def compute_output_shape(self, input_shape):
        batch, height, width, channels = input_shape
        return (
            batch,
            None if height is None else height + 2,
            None if width is None else width + 2,
            channels,
        )


@tf.keras.utils.register_keras_serializable(package="Assignment2")
class ZeroPadChannels(tf.keras.layers.Layer):
    """Append zero channels to the shortcut without adding learned weights."""

    def __init__(self, extra_channels, **kwargs):
        super().__init__(**kwargs)
        self.extra_channels = extra_channels

    def call(self, inputs):
        # Concatenate zero maps to avoid a TensorFlow CPU Pad/Conv fusion issue.
        zeros = tf.repeat(tf.zeros_like(inputs[..., :1]), self.extra_channels, axis=-1)
        return tf.concat([inputs, zeros], axis=-1)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], input_shape[-1] + self.extra_channels)

    def get_config(self):
        return {**super().get_config(), "extra_channels": self.extra_channels}


def residual_block(x, filters, number):
    """Add a zero-padded shortcut around one Exercise 1 convolution."""
    shortcut = x
    extra_channels = filters - int(shortcut.shape[-1])
    if extra_channels:
        shortcut = ZeroPadChannels(
            extra_channels, name=f"shortcut_{number}_zero_channels"
        )(shortcut)

    # These layers match Exercise 1: replicate padding, a 3 x 3 convolution,
    # stride 1, and GELU. The GELU is applied after the shortcut is added.
    y = ReplicatePadding2D(name=f"replicate_pad_{number}")(x)
    y = tf.keras.layers.Conv2D(
        filters, 3, strides=1, padding="valid", name=f"conv_{number}"
    )(y)
    y = tf.keras.layers.Add(name=f"add_{number}")([y, shortcut])
    return tf.keras.layers.Activation("gelu", name=f"gelu_{number}")(y)


def build_model():
    """Keep the Exercise 1 CNN and add one residual shortcut per convolution."""
    inputs = tf.keras.Input(shape=(28, 28), name="image")
    x = tf.keras.layers.Reshape((28, 28, 1), name="grayscale_channel")(inputs)
    x = residual_block(x, 32, 1)
    x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid", name="pool_1")(x)
    x = residual_block(x, 64, 2)
    x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid", name="pool_2")(x)
    x = residual_block(x, 64, 3)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(128, activation="gelu", name="dense")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="class_probabilities")(x)
    return tf.keras.Model(inputs, outputs, name="fashion_mnist_residual_three_conv_gelu")


class TestMetrics(tf.keras.callbacks.Callback):
    """Record test metrics and keep the best validation weights in memory."""

    def on_train_begin(self, logs=None):
        self.best_loss = float("inf")
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.model.evaluate(
            data.test_images, data.test_labels,
            batch_size=BATCH_SIZE, verbose=0, return_dict=True,
        )
        logs["test_loss"] = metrics["loss"]
        logs["test_accuracy"] = metrics["accuracy"]
        if logs["val_loss"] < self.best_loss:
            self.best_loss = logs["val_loss"]
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)


def plot_results(history, true_labels, predicted_labels, output_dir):
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), layout="constrained")
    for ax, metric, title in zip(axes, ["loss", "accuracy"], ["Cross-entropy loss", "Accuracy"]):
        ax.plot(epochs, history[metric], "o-", label="Training")
        ax.plot(epochs, history["val_" + metric], "o-", label="Validation")
        ax.plot(epochs, history["test_" + metric], "o-", label="Test")
        ax.set(xlabel="Epoch", ylabel=title, title=title)
        ax.set_xticks(epochs)
        ax.tick_params(axis="x", labelsize=8, labelrotation=45)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(fig)

    matrix = confusion_matrix(true_labels, predicted_labels, labels=np.arange(10))
    fig, ax = plt.subplots(figsize=(10, 9), layout="constrained")
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45
    )
    ax.set_title("Test set confusion matrix: rows = true, columns = predicted")
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(42)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy", metrics=["accuracy"],
    )
    model.summary()
    start = time.perf_counter()
    history = model.fit(
        data.train_images, data.train_labels,
        validation_data=(data.validation_images, data.validation_labels),
        epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2,
        callbacks=[TestMetrics()],
    ).history
    training_seconds = time.perf_counter() - start

    # The callback restored the best validation weights without saving a model file.
    test_metrics = model.evaluate(
        data.test_images, data.test_labels, batch_size=BATCH_SIZE,
        verbose=0, return_dict=True,
    )
    probabilities = model.predict(data.test_images, batch_size=BATCH_SIZE, verbose=0)
    true_labels = data.test_labels.argmax(axis=1)
    predictions = probabilities.argmax(axis=1)
    plot_results(history, true_labels, predictions, output_dir)
    best_index = int(np.argmin(history["val_loss"]))
    print(f"Best validation epoch: {best_index + 1}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Training and per-epoch evaluation time: {training_seconds:.1f} seconds")
    print(f"Saved confusion_matrix.png and learning_curves.png to {output_dir}")



if __name__ == "__main__":
    main()
