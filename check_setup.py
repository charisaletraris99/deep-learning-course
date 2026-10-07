"""Run this file to verify the course environment and train a tiny model."""
import sys
import numpy as np
import pandas as pd
import matplotlib
import sklearn
import tensorflow as tf

print("Python:", sys.version)
print("Interpreter:", sys.executable)
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)
print("scikit-learn:", sklearn.__version__)
print("TensorFlow:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())

tf.keras.utils.set_random_seed(42)
x = np.linspace(-1, 1, 128, dtype=np.float32).reshape(-1, 1)
y = 2 * x + 1
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss="mse")
history = model.fit(x, y, epochs=15, batch_size=32, verbose=0)
first, last = history.history["loss"][0], history.history["loss"][-1]
assert np.isfinite(last) and last < first, "Training did not reduce the loss"
print(f"Training loss: {first:.6f} -> {last:.6f}")
print("Setup verified: model training works.")
