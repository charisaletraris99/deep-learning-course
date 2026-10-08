"""Dense TensorFlow updates; MAG is global across every trainable scalar."""
import math
import tensorflow as tf


class GradientStep:
    def __init__(self, method, learning_rate, variables, epsilon=1e-8, b0=1.0, *, map_floor_rate=0.1):
        if learning_rate <= 0 or epsilon <= 0 or b0 <= 0:
            raise ValueError("learning_rate, epsilon and b0 must be positive")
        if not math.isfinite(map_floor_rate) or map_floor_rate < 0:
            raise ValueError("map_floor_rate must be finite and non-negative")
        self.map_floor_rate = float(map_floor_rate)
        self.method = method
        self.rate = float(learning_rate)
        self.epsilon = epsilon
        self.b_squared = tf.Variable(b0 ** 2, trainable=False, dtype=tf.float32)
        factories = {"sgd": tf.keras.optimizers.SGD,
                     "adam": tf.keras.optimizers.Adam,
                     "rmsprop": tf.keras.optimizers.RMSprop,
                     "adagrad": tf.keras.optimizers.Adagrad}
        self.builtin = factories[method](learning_rate=self.rate) if method in factories else None
        if self.builtin is not None:
            self.builtin.build(variables)
        elif method not in {"mag", "mag_floor", "inverse_mag", "adagrad_norm"}:
            raise ValueError(f"Unknown method: {method}")

    def apply(self, gradients, variables):
        if any(g is None or isinstance(g, tf.IndexedSlices) for g in gradients):
            raise ValueError("This experiment requires connected, dense gradients")
        count = tf.add_n([tf.cast(tf.size(g), tf.float32) for g in gradients])
        magnitude = tf.add_n([tf.reduce_sum(tf.abs(g)) for g in gradients]) / count
        squared_norm = tf.add_n([tf.reduce_sum(tf.square(g)) for g in gradients])
        for g in gradients:
            tf.debugging.assert_all_finite(g, "Non-finite gradient")
        if self.method == "mag":
            rate = self.rate * magnitude
        elif self.method == "mag_floor":
            rate = self.rate * tf.maximum(magnitude, self.map_floor_rate)
        elif self.method == "inverse_mag":
            rate = self.rate / (magnitude + self.epsilon)
        elif self.method == "adagrad_norm":
            self.b_squared.assign_add(squared_norm)
            rate = self.rate / tf.sqrt(self.b_squared)
        else:
            rate = tf.constant(self.rate, tf.float32)
        before = [tf.identity(v) for v in variables]
        if self.builtin is not None:
            self.builtin.apply_gradients(zip(gradients, variables))
        else:
            for g, v in zip(gradients, variables):
                v.assign_sub(rate * g)
        update_norm = tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(v - old))
                                      for v, old in zip(variables, before)]))
        for v in variables:
            tf.debugging.assert_all_finite(v, "Non-finite parameter after update")
        return magnitude, rate, update_norm


def floor_for_rate(config, rate):
    """Read a rate-specific floor, preserving legacy fixed-floor experiments."""
    if "map_floor_rates" in config:
        return float(config["map_floor_rates"][str(float(rate))])
    return float(config.get("map_floor_rate", 1.0))


def calculate_floor_rates(grids):
    """Pair grid positions, or broadcast a single SGD reference rate."""
    mag_rates = grids["mag_floor"]
    sgd_rates = grids.get("sgd", [])
    if not sgd_rates or any(not math.isfinite(r) or r <= 0 for r in sgd_rates):
        raise ValueError("SGD reference rates must be finite and positive")
    if len(sgd_rates) not in (1, len(mag_rates)):
        raise ValueError("Use one SGD reference rate or equal-length SGD and mag_floor grids")
    if len(set(mag_rates)) != len(mag_rates):
        raise ValueError("mag_floor base rates must be unique")
    result = {}
    for i, rate in enumerate(mag_rates):
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("mag_floor base rates must be finite and positive")
        floor = sgd_rates[0 if len(sgd_rates) == 1 else i] / rate
        if not math.isfinite(floor):
            raise ValueError("Calculated map_floor_rate must be finite")
        result[str(float(rate))] = floor
    return result


def optimizer_label(method, config, rate=None):
    if method == "mag_floor":
        if rate is not None:
            values = [floor_for_rate(config, rate)]
        elif "map_floor_rates" in config:
            values = sorted(set(config["map_floor_rates"].values()))
        else:
            values = [config.get("map_floor_rate", 1.0)]
        return "mag_floor (map_floor_rate=" + ", ".join(f"{v:g}" for v in values) + ")"
    return method
