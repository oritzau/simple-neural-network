import math
import numpy as np

def main():
    model = SimpleSequential(layers=[SimpleLayer(2, activation=sigmoid), SimpleLayer(1, activation=sigmoid)])
    model.compile([.35, .7])
    model.layers[0].weights = np.array([[.2, .3], [.2, .3]])
    model.layers[1].weights = np.array([[.3], [.9]])
    model.feed_forward()
    model.back_propagation(y=[1.0])

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * 1 - x

class SimpleLayer:
    def __init__(self, size, activation=None):
        self.size = size
        self.activation = activation

class SimpleSequential:
    def __init__(self, layers=[], learning_rate=1):
        self.layers = []
        for layer in layers:
            self.layers.append(layer)
        self.output = 0
        self.learning_rate = learning_rate

    def compile(self, x):
        self.input = x
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == 0:
                layer.weights = np.random.rand(len(x), layer.size)
            else:
                layer.weights = np.random.rand(self.layers[i - 1].size, layer.size)
            layer.output = np.zeros(layer.size)

    def feed_forward(self):
        x = self.input
        for i in range(len(self.layers)):
            layer = self.layers[i]
            print(f"x: {x}")
            print(f"weights: {layer.weights}")
            x = layer.output = np.array(list(map(layer.activation, np.dot(x, layer.weights))))
            print(f"x after:{x}")
            if i == len(self.layers) - 1:
                self.output = x
        print(f"output: {self.output}")

    def back_propagation(self, y):
        cost = np.sum((self.output - y) ** 2)
        print(f"cost: {cost}")
        output_error = sigmoid_derivative(self.output[0]) * y[0]

        hidden_layer = self.layers[1]

if __name__ == "__main__":
    main()
