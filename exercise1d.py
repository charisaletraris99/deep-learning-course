from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import csv

# Record test accuracy at the end of every epoch.
class TestAccuracyCallback(keras.callbacks.Callback):
    def __init__(self, X_test, y_test):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.test_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_accuracy = self.model.evaluate(
            self.X_test,
            self.y_test,
            verbose=0
        )
        self.test_accuracies.append(test_accuracy)

def build_model(activation_function, seed):
    # Clear Keras's model-building state between runs.
    keras.backend.clear_session()

    # Choose reproducible random starting conditions for this run.
    keras.utils.set_random_seed(seed)

    # Create a fresh network with new weights.
    model = keras.Sequential([
        keras.Input(shape=(4,)),
        keras.layers.Dense(16, activation=activation_function),
        keras.layers.Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=0.02,
            momentum=0.9
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# One-hot encode the labels
encoder = OneHotEncoder()
y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

activation_functions = ["tanh", "sigmoid", "relu"]
runs_numbers = 20
epochs_numbers = 40
results = {}

for activation_function in activation_functions:
    training_accuracies = []
    validation_accuracies = []
    test_accuracies = []

    for run in range(runs_numbers):
        # Use different seeds across runs, but the same sequence
        # of seeds for every activation function.
        seed = 42 + run

        # Start each run with a fresh model and optimizer.
        model = build_model(
            activation_function=activation_function,
            seed=seed
        )

        # Start with an empty test-accuracy recorder.
        test_callback = TestAccuracyCallback(X_test, y_test)

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs_numbers,
            batch_size=32,
            validation_split=0.1,
            callbacks=[test_callback],
            verbose=0
        )

        # Save the complete accuracy curve from each run.
        training_accuracies.append(history.history["accuracy"])
        validation_accuracies.append(history.history["val_accuracy"])
        test_accuracies.append(test_callback.test_accuracies)

        # Get the accuracy from the final epoch of this run.
        training_accuracy = history.history["accuracy"][-1]
        validation_accuracy = history.history["val_accuracy"][-1]
        test_accuracy = test_callback.test_accuracies[-1]

        print(
            f"Activation={activation_function}, "
            f"run {run + 1}/{runs_numbers} | "
            f"Training: {training_accuracy:.2%} | "
            f"Validation: {validation_accuracy:.2%} | "
            f"Test: {test_accuracy:.2%}"
        )

    results[activation_function] = {
        "training": np.array(training_accuracies),
        "validation": np.array(validation_accuracies),
        "test": np.array(test_accuracies)
    }
epochs = np.arange(1, epochs_numbers + 1)

plot_types = [
    ("training", "Training"),
    ("validation", "Validation"),
    ("test", "Test")
]

for metric, title in plot_types:
    # Create a separate figure for this accuracy type.
    plt.figure(figsize=(9, 6))

    for activation_function in activation_functions:
        # Average the 20 runs at each epoch.
        mean_accuracy = results[activation_function][metric].mean(axis=0)

        plt.plot(
            epochs,
            mean_accuracy,
            label=f"Activation = {activation_function}"
        )

    plt.xlabel("Epoch")
    plt.ylabel(f"Average {title.lower()} accuracy")
    plt.title(f"{title} accuracy: average of {runs_numbers} runs")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        f"exercise1d_{metric}_accuracy.png",
        dpi=300
    )


# Print the mean final accuracy for all three datasets.
print(f"\nMean accuracies at epoch {epochs_numbers}:")

for activation_function in activation_functions:
    training_mean = results[activation_function]["training"][:, -1].mean()
    validation_mean = results[activation_function]["validation"][:, -1].mean()
    test_mean = results[activation_function]["test"][:, -1].mean()

    print(
        f"Activation={activation_function} | "
        f"Training: {training_mean:.2%} | "
        f"Validation: {validation_mean:.2%} | "
        f"Test: {test_mean:.2%}"
    )

# Save exact final averages for the report (values are percentages).
with open("exercise1d_summary.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Activation", "Training (%)", "Validation (%)", "Test (%)"])
    for activation_function in activation_functions:
        writer.writerow([activation_function] + [
            100 * results[activation_function][metric][:, -1].mean()
            for metric in ["training", "validation", "test"]
        ])

# Save every run and epoch, so results can be inspected without retraining.
np.savez_compressed("exercise1d_results.npz", **{
    f"activation_{activation_function}_{metric}": values
    for activation_function, metrics in results.items()
    for metric, values in metrics.items()
})

# Display the figures after printing the final results.
plt.show()
