from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

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

def build_model(learning_rate, seed):
    # Clear Keras's model-building state between runs.
    keras.backend.clear_session()

    # Choose reproducible random starting conditions for this run.
    keras.utils.set_random_seed(seed)

    # Create a fresh network with new weights.
    model = keras.Sequential([
        keras.Input(shape=(4,)),
        keras.layers.Dense(16, activation="tanh"),
        keras.layers.Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.0
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

learning_rates = [0.003, 0.03, 0.06, 0.2, 0.5]
runs_numbers = 20
epochs_numbers = 40
results = {}

for learning_rate in learning_rates:
    training_accuracies = []
    validation_accuracies = []
    test_accuracies = []

    for run in range(runs_numbers):
        # Use different seeds across runs, but the same sequence
        # of seeds for every learning rate.
        seed = 42 + run

        # Start each run with a fresh model and optimizer.
        model = build_model(
            learning_rate=learning_rate,
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

        # Save every epoch so the curves can be averaged across runs.
        training_accuracies.append(history.history["accuracy"])
        validation_accuracies.append(history.history["val_accuracy"])
        test_accuracies.append(test_callback.test_accuracies)

        # Get the accuracy from the final epoch of this run.
        training_accuracy = history.history["accuracy"][-1]
        validation_accuracy = history.history["val_accuracy"][-1]
        test_accuracy = test_callback.test_accuracies[-1]

        print(
            f"Learning rate={learning_rate}, "
            f"run {run + 1}/{runs_numbers} | "
            f"Training: {training_accuracy:.2%} | "
            f"Validation: {validation_accuracy:.2%} | "
            f"Test: {test_accuracy:.2%}"
        )

    results[learning_rate] = {
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

    for learning_rate in learning_rates:
        # Average the 20 runs at each epoch.
        mean_accuracy = results[learning_rate][metric].mean(axis=0)

        plt.plot(
            epochs,
            mean_accuracy,
            label=f"Learning rate = {learning_rate}"
        )

    plt.xlabel("Epoch")
    plt.ylabel(f"Average {title.lower()} accuracy")
    plt.title(f"{title} accuracy: average of {runs_numbers} runs")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        f"exercise1a_{metric}_accuracy.png",
        dpi=300
    )


# Print the mean final accuracy for all three datasets.
print(f"\nMean accuracies at epoch {epochs_numbers}:")

for learning_rate in learning_rates:
    training_mean = results[learning_rate]["training"][:, -1].mean()
    validation_mean = results[learning_rate]["validation"][:, -1].mean()
    test_mean = results[learning_rate]["test"][:, -1].mean()

    print(
        f"Learning rate={learning_rate} | "
        f"Training: {training_mean:.2%} | "
        f"Validation: {validation_mean:.2%} | "
        f"Test: {test_mean:.2%}"
    )

# Display all three figures after printing the results.
plt.show()
