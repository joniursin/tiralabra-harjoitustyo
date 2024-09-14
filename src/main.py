import numpy as np

class Network:
    """Luokka, joka luo neuroverkon ja kouluttaa sitä.

    Attributes:
        input_layer: Neuroverkon 1 tason koko.
        hidden_layer: Neuroverkon 2 tason koko.
        output_layer: Neuroverkon 3 tason koko.
    """
    def __init__(self, input_layer=784, hidden_layer=100, output_layer=10):
        """Luokan konstruktori, joka luo neuroverkon

        Args:
            input_layer (int, optional): _description_. Defaults to 784.
            hidden_layer (int, optional): _description_. Defaults to 100.
            output_layer (int, optional): _description_. Defaults to 10.
        """
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.biases = []
        self.weights = []

        self.biases.append(np.zeros((1, self.hidden_layer)))
        self.weights.append(np.random.randn(self.input_layer, self.hidden_layer))

        self.biases.append(np.zeros((1, self.output_layer)))
        self.weights.append(np.random.randn(self.hidden_layer, self.output_layer))

    def feedforward(self, input):
        """_summary_

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.hidden_activation = np.dot(input, self.weights[0]) + self.biases[0]
        self.hidden_output = sigmoid(self.hidden_activation)

        self.output_activation = np.dot(self.hidden_output, self.weights[1]) + self.biases[1]
        self.predicted_output = sigmoid(self.output_activation)

        return self.predicted_output
    
    def backpropagation(self, x, y, rate):
        """_summary_

        Args:
            x (_type_): _description_
            y (_type_): _description_
            rate (_type_): _description_
        """
        output_error = y - self.predicted_output
        output_delta = output_error * sigmoid_derivative(self.predicted_output)

        hidden_error = np.dot(output_delta, self.weights[1].T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        self.weights[1] += np.dot(self.hidden_output.T, output_delta) * rate
        self.biases[1] += np.sum(output_delta, axis=0, keepdims=True) * rate
        self.weights[0] += np.dot(x.T, hidden_delta) * rate
        self.biases[0] += np.sum(hidden_delta, axis=0, keepdims=True) * rate
    
    def train(self, x, y, epochs, rate):
        """_summary_

        Args:
            x (_type_): _description_
            y (_type_): _description_
            epochs (_type_): _description_
            rate (_type_): _description_
        """
        for i in range(epochs):
            output = self.feedforward(x)
            self.backpropagation(x, y, rate)
            loss = np.mean(np.square(y - output))

def sigmoid(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return x * (1-x)