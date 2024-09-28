import unittest
import numpy as np
from network import Network

class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.input_layer = 6
        self.hidden_layer = 3
        self.output_layer = 2
        self.network = Network(self.input_layer, self.hidden_layer, self.output_layer)
    
    def test_biases(self):
        self.assertEqual(len(self.network.biases), 2)
        self.assertEqual(self.network.biases[0].shape, (self.hidden_layer, 1))
        self.assertEqual(self.network.biases[1].shape, (self.output_layer, 1))

    def test_weights(self):
        self.assertEqual(len(self.network.weights), 2)
        self.assertEqual(self.network.weights[0].shape, (self.hidden_layer, self.input_layer))
        self.assertEqual(self.network.weights[1].shape, (self.output_layer, self.hidden_layer))

    def test_feedforward(self):
        x = np.random.randn(self.input_layer, 1)
        self.assertEqual(self.network.feedforward(x).shape, (self.output_layer, 1))

    def test_backpropagation(self):
        x = np.random.randn(self.input_layer, 1)
        y = np.array([[0], [1]])

        biases_0 = np.copy(self.network.biases[0])
        biases_1 = np.copy(self.network.biases[1])
        weights_0 = np.copy(self.network.weights[0])
        weights_1 = np.copy(self.network.weights[1])

        self.network.train(x, y, 0.1)

        self.assertFalse(np.array_equal(biases_0, self.network.biases[0]))
        self.assertFalse(np.array_equal(biases_1, self.network.biases[1]))
        self.assertFalse(np.array_equal(weights_0, self.network.weights[0]))
        self.assertFalse(np.array_equal(weights_1, self.network.weights[1]))


    