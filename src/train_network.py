import numpy as np
import idx2numpy
from network import Network

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
print(nn.biases[1])
print("Training network...")
for i in range(X_train.shape[0]):
    x = np.array(X_train[i, :].reshape(-1, 1))
    y = np.array(y_train[i, :].reshape(-1, 1))
    nn.train(x, y, 0.1)

input("Training done, start testing [enter]")
correct = 0
for i in range(X_test.shape[0]):
    x = np.array(X_test[i].reshape(-1, 1))
    predict = nn.feedforward(x)
    #print("was:", np.argmax(y_test[i]), "predict:", np.argmax(
        #predict), predict[np.argmax(predict)])
    if np.argmax(y_test[i]) == np.argmax(predict):
        correct += 1
print("Accuracy", correct/10000)