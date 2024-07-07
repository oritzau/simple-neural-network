import math
import numpy as np

def main():
    sigmoid_activation = DifferentiableFunction(sigmoid, sigmoid_deriv)
    model = SimpleSequential(layers=[
        SimpleLayer(size=2, activation=sigmoid_activation),
        SimpleLayer(size=1, activation=sigmoid_activation),
    ], learning_rate=1)
    X = np.array([[.35], [.7]])
    model.compile(X, loss=DifferentiableFunction(mse, mse_deriv))
    model.layers[0].weights = np.array([[.2, .2], [.3, .3]])
    model.layers[1].weights = np.array([.3, .9])
    model.feed_forward()
    model.back_propagation(y=np.array([[1.0]]))
    print(model.layers[1].weights)

class DifferentiableFunction:
    def __init__(self, inner, deriv):
        self.inner = inner
        self.deriv = deriv

    def __call__(self, *args):
        return self.inner(*args)
    
    def deriv(self, *args):
        return self.deriv(*args)


class SimpleLayer:
    def __init__(self, size, activation=None):
        self.size = size
        self.activation = activation
        self.bias = self.output = np.zeros((size, 1))

class SimpleSequential:
    def __init__(self, layers=[], learning_rate=0.1):
        self.layers = layers
        self.learning_rate = learning_rate

    def compile(self, X, loss):
        self.input = X
        self.loss = loss
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == 0:
                layer.weights = np.random.rand(len(self.input), layer.size)
            else:
                layer.weights = np.random.rand(self.layers[i - 1].size, layer.size)
            layer.output = np.zeros(layer.size)

    def feed_forward(self):
        X = self.input
        for i in range(len(self.layers)):
            layer = self.layers[i]
            print(f"X: {X}")
            print(f"weights: {layer.weights}")
            z = np.dot(np.transpose(X), layer.weights) + layer.bias
            layer.z = z
            print(f"z: {z}")
            X = layer.output = layer.activation(z)
            print(f"X after:{X}")
            if i == len(self.layers) - 1:
                self.output = X
        print(f"output: {self.output}")

    def back_propagation(self, y):
        loss = self.loss(self.output, y)
        print(f"loss: {loss}")

        # Walk down to all but first layer
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            print(f"layer.weights: {layer.weights}")
            print(f"layer.activation.deriv(layer.z): {layer.activation.deriv(layer.z)}")
            weight_times_activ_deriv = np.matmul(layer.weights, np.transpose(layer.activation.deriv(layer.z)))
            print(f"weight_times_activ_deriv: {weight_times_activ_deriv }")
            print(f"self.loss.deriv(layer.output, y): {self.loss.deriv(layer.output, y)}")
            gradient = np.matmul(weight_times_activ_deriv , self.loss.deriv(layer.output, y))
            print(f"gradient: {gradient}")
            layer.weights -= gradient * self.learning_rate

# All activation and cost functions defined here
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    c = np.zeros(shape=x.shape)
    return np.maximum(c, x)

def relu_deriv(x):
    return x > 0

def mse(x, y):
    return .5 * 1 / len(x) * ((x - y) ** 2)

def mse_deriv(x, y):
    return (x - y)

if __name__ == "__main__":
    main()
