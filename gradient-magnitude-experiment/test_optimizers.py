import unittest
import numpy as np
import tensorflow as tf
from optimizers import GradientStep, calculate_floor_rates, floor_for_rate


class GradientRuleTests(unittest.TestCase):
    def check_rule(self, method, expected_rate):
        variables = [tf.Variable([1.0, 2.0]), tf.Variable([3.0])]
        gradients = [tf.constant([2.0, -4.0]), tf.constant([0.0])]
        optimizer = GradientStep(method, 0.1, variables)
        magnitude, rate, norm = optimizer.apply(gradients, variables)
        self.assertAlmostEqual(float(magnitude), 2.0)
        self.assertAlmostEqual(float(rate), expected_rate, places=6)
        np.testing.assert_allclose(variables[0].numpy(), [1 - 2 * expected_rate, 2 + 4 * expected_rate], rtol=1e-6)
        self.assertAlmostEqual(float(norm), expected_rate * np.sqrt(20), places=6)
        return optimizer, variables, gradients

    def test_exact_numerator(self):
        self.check_rule('mag', 0.2)

    def test_floor_above_one(self):
        self.check_rule('mag_floor', 0.2)

    def test_floor_below_and_at_one(self):
        for value in [0.0, 0.25, 1.0]:
            with self.subTest(magnitude=value):
                variable = tf.Variable([1.0, 2.0])
                optimizer = GradientStep('mag_floor', 0.1, [variable], map_floor_rate=1.0)
                magnitude, rate, _ = optimizer.apply(
                    [tf.constant([value, -value])], [variable])
                self.assertAlmostEqual(float(magnitude), value)
                self.assertAlmostEqual(float(rate), 0.1)
                np.testing.assert_allclose(variable.numpy(),
                                           [1 - 0.1 * value, 2 + 0.1 * value], rtol=1e-6)

    def test_floor_ratios(self):
        self.assertEqual(calculate_floor_rates({'sgd': [0.1], 'mag_floor': [10., 20.]}),
                         {'10.0': 0.01, '20.0': 0.005})
        self.assertEqual(calculate_floor_rates({'sgd': [0.1, 1.], 'mag_floor': [10., 30.]}),
                         {'10.0': 0.01, '30.0': 1 / 30})
        with self.assertRaises(ValueError):
            calculate_floor_rates({'sgd': [0.1, 1.], 'mag_floor': [10.]})
        self.assertEqual(floor_for_rate({}, 10.), 1.)
        self.assertEqual(floor_for_rate({'map_floor_rate': 0.03}, 10.), 0.03)
        self.assertEqual(floor_for_rate({'map_floor_rates': {'10.0': 0.01}}, 10.), 0.01)

    def test_custom_floor(self):
        for magnitude, expected_rate in [(0.005, 0.5), (0.01, 0.5), (0.05, 0.5), (0.1, 1.0)]:
            with self.subTest(magnitude=magnitude):
                variable = tf.Variable([1.0])
                optimizer = GradientStep('mag_floor', 10.0, [variable], map_floor_rate=0.01)
                _, rate, _ = optimizer.apply([tf.constant([magnitude])], [variable])
                self.assertAlmostEqual(float(rate), expected_rate, places=6)
                self.assertAlmostEqual(float(variable[0]), 1 - expected_rate * magnitude, places=6)
                self.assertEqual(optimizer.epsilon, 1e-8)

    def test_invalid_floor(self):
        for floor in [-1, float('nan'), float('inf')]:
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                GradientStep('mag_floor', 0.1, [tf.Variable([1.0])], map_floor_rate=floor)

    def test_denominator(self):
        self.check_rule('inverse_mag', 0.1 / (2 + 1e-8))

    def test_norm_accumulates_before_update(self):
        optimizer, variables, gradients = self.check_rule('adagrad_norm', 0.1 / np.sqrt(21))
        _, rate, _ = optimizer.apply(gradients, variables)
        self.assertAlmostEqual(float(rate), 0.1 / np.sqrt(41), places=6)

    def test_builtin_integer_learning_rate(self):
        for method in ['sgd', 'adam', 'rmsprop', 'adagrad']:
            with self.subTest(method=method):
                variable = tf.Variable([1.0])
                optimizer = GradientStep(method, 1, [variable])
                _, rate, _ = optimizer.apply([tf.constant([0.25])], [variable])
                self.assertEqual(float(rate), 1.0)
                self.assertTrue(np.isfinite(variable.numpy()).all())
                if method == 'sgd':
                    self.assertAlmostEqual(float(variable[0]), 0.75)

    def test_sgd(self):
        self.check_rule('sgd', 0.1)

    def test_zero_gradient(self):
        for method in ['mag', 'mag_floor', 'inverse_mag', 'adagrad_norm']:
            variable = tf.Variable([1.0])
            optimizer = GradientStep(method, 0.1, [variable])
            _, _, norm = optimizer.apply([tf.constant([0.0])], [variable])
            self.assertEqual(float(norm), 0.0)
            self.assertEqual(float(variable[0]), 1.0)

    def test_nonfinite_rejected(self):
        variable = tf.Variable([1.0])
        optimizer = GradientStep('mag', 0.1, [variable])
        with self.assertRaises(tf.errors.InvalidArgumentError):
            optimizer.apply([tf.constant([float('nan')])], [variable])


if __name__ == '__main__':
    unittest.main()
