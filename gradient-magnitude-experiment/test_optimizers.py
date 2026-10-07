import unittest
import numpy as np
import tensorflow as tf
from optimizers import GradientStep


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
                optimizer = GradientStep('mag_floor', 0.1, [variable])
                magnitude, rate, _ = optimizer.apply(
                    [tf.constant([value, -value])], [variable])
                self.assertAlmostEqual(float(magnitude), value)
                self.assertAlmostEqual(float(rate), 0.1)
                np.testing.assert_allclose(variable.numpy(),
                                           [1 - 0.1 * value, 2 + 0.1 * value], rtol=1e-6)

    def test_denominator(self):
        self.check_rule('inverse_mag', 0.1 / (2 + 1e-8))

    def test_norm_accumulates_before_update(self):
        optimizer, variables, gradients = self.check_rule('adagrad_norm', 0.1 / np.sqrt(21))
        _, rate, _ = optimizer.apply(gradients, variables)
        self.assertAlmostEqual(float(rate), 0.1 / np.sqrt(41), places=6)

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
