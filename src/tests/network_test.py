import unittest
import numpy as np
from network import Network


class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.input_layer = 2
        self.hidden_layer = 2
        self.output_layer = 1
        self.network = Network(
            self.input_layer, self.hidden_layer, self.output_layer)

    def test_biases(self):
        self.assertEqual(len(self.network.biases), 2)
        self.assertEqual(self.network.biases[0].shape, (self.hidden_layer, 1))
        self.assertEqual(self.network.biases[1].shape, (self.output_layer, 1))

    def test_weights(self):
        self.assertEqual(len(self.network.weights), 2)
        self.assertEqual(
            self.network.weights[0].shape, (self.hidden_layer, self.input_layer))
        self.assertEqual(
            self.network.weights[1].shape, (self.output_layer, self.hidden_layer))

    def test_feedforward_return(self):
        x = np.random.randn(self.input_layer, 1)
        self.assertEqual(self.network.feedforward(
            x).shape, (self.output_layer, 1))

    def test_feedforward_values(self):
        self.network.weights = [
            np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([[0.5, 0.6]])]
        self.network.biases = [np.array([[0.1], [0.2]]), np.array([[0.3]])]
        feedforward = self.network.feedforward(np.array([[0.2], [0.3]]))[0][0]
        self.assertAlmostEqual(feedforward, 0.7168243, 7)

    def test_backpropagation_values(self):
        self.network.weights = [
            np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([[0.5, 0.6]])]
        self.network.biases = [np.array([[0.1], [0.2]]), np.array([[0.3]])]
        self.network.train(np.array([[0.2], [0.3]]), np.array([[1]]), 0.1)
        biases = [np.array([[0.10154455], [0.20154455]]),
                  np.array([[0.3057481]])]
        weights = [np.array([[0.10014254, 0.20021382], [0.30016636, 0.40024955]]), np.array(
            [[0.50313202, 0.60341364]])]
        np.testing.assert_almost_equal(self.network.biases[0], biases[0], 7)
        np.testing.assert_almost_equal(self.network.biases[1], biases[1], 7)
        np.testing.assert_almost_equal(self.network.weights[0], weights[0], 7)
        np.testing.assert_almost_equal(self.network.weights[1], weights[1], 7)
