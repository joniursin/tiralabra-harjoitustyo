import numpy as np
import idx2numpy

class Network:
    """Class, which creates a neural network and trains it.
    """
    def __init__(self, input_layer, hidden_layer, output_layer):
        """Contructor, which initializes the network and its weights and biases.

        Args:
            input_layer (int): Number of neurons in the input layer.
            hidden_layer (int): Number of neurons in the hidden layer.
            output_layer (int): Number of neurons in the output layer.
        """
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.biases = []
        self.weights = []

        self.biases.append(np.zeros([self.hidden_layer, 1]))
        self.weights.append(np.random.randn(self.hidden_layer, self.input_layer) * np.sqrt(2/self.input_layer))

        self.biases.append(np.zeros([self.output_layer, 1]))
        self.weights.append(np.random.randn(self.output_layer, self.hidden_layer))

    def feedforward(self, x):
        """Forward propagates through the network.

        Args:
            x (np.array): Input data.
        Returns:
            np.array: Predicted output.
        """
        self.hidden_activation = np.dot(self.weights[0], x) + self.biases[0]
        self.hidden_output = sigmoid(self.hidden_activation)

        self.output_activation = np.dot(self.weights[1], self.hidden_output) + self.biases[1]
        self.predicted_output = sigmoid(self.output_activation)

        return self.predicted_output
    
    def backpropagation(self, x, y, rate):
        """Backpropagation algorithm, which updates the weights and biases.

        Args:
            x (np.array): Input data.
            y (np.array): Value of the input data.
            rate (float): Learning rate.
        """
        output_error = y - self.predicted_output
        output_delta = output_error * sigmoid_derivative(self.predicted_output)

        hidden_error = output_delta.T.dot(self.weights[1])
        hidden_delta = hidden_error.T * sigmoid_derivative(self.hidden_output)

        self.weights[1] += self.hidden_output.dot(output_delta.T).T * rate
        self.biases[1] += np.sum(output_delta, axis=0, keepdims=True) * rate
        self.weights[0] += x.dot(hidden_delta.T).T * rate
        self.biases[0] += np.sum(hidden_delta, axis=0, keepdims=True) * rate
    
    def train(self, x, y, rate):
        """Trains the network using input data and it's values.

        Args:
            x (np.array): Input data.
            y (np.array): Value of the input data.
            rate (_float): Learning rate.
        """
        self.feedforward(x)
        self.backpropagation(x, y, rate)

def sigmoid(x):
    """Calculates sigmoid.

    Args:
        x (np.array): Value that gets calculated.

    Returns:
        np.array: Sigmoid.
    """
    print(x)
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    """Calculates sigmoid derivative.

    Args:
        x (np.array): Value that gets calculated.

    Returns:
        np.array: Sigmoid derivative.
    """
    return x * (1-x)

# Move these to functions/files
X_train = idx2numpy.convert_from_file('data/train-images-idx3-ubyte')/255
X_train = X_train.reshape(X_train.shape[0], -1)
y = idx2numpy.convert_from_file('data/train-labels-idx1-ubyte')
y_train = np.zeros((y.size, 10))
y_train[np.arange(y.size), y] = 1

X_test = idx2numpy.convert_from_file('data/t10k-images-idx3-ubyte')/255
X_test = X_test.reshape(X_test.shape[0], -1)
y = idx2numpy.convert_from_file('data/t10k-labels-idx1-ubyte')
y_test = np.zeros((y.size, 10))
y_test[np.arange(y.size), y] = 1

nn = Network(784, 100, 10)

for i in range(X_train.shape[0]):
    x = np.array(X_train[i, :].reshape(-1,1))
    y = np.array(y_train[i, :].reshape(-1,1))
    nn.train(x, y, 0.1)

for i in range(X_test.shape[0]): 
    x = np.array(X_test[i].reshape(-1,1))
    predict = nn.feedforward(x)
    print("was:",np.argmax(y_test[i]), "predict:",np.argmax(predict), predict[np.argmax(predict)])