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
    return x * (1 - x)

class SimpleLayer:
    def __init__(self, size, activation=None):
        self.size = size
        self.activation = activation
        self.bias = self.output = np.zeros(size)

class SimpleSequential:
    def __init__(self, layers=[], learning_rate=1):
        self.layers = []
        for layer in layers:
            self.layers.append(layer)
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
            x = layer.output = np.array(list(map(layer.activation, np.dot(x, layer.weights) + layer.bias)))
            print(f"x after:{x}")
            if i == len(self.layers) - 1:
                self.output = x
        print(f"output: {self.output}")

    def back_propagation(self, y):

        output_layer = self.layers[-1]
        correct = int(np.argmax(output_layer) == np.argmax(y))

        # First doing output layer (assuming there is a hidden layer)
        output_diff = output_layer.output - y
        print(f"output_diff: {output_diff}")
        output_layer.weights += -self.learning_rate * output_diff @ self.layers[-2].output
        prev_output = output_diff

        # Walk down the layers
        for i in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            hidden_diff = next_layer.weights @ output_diff * sigmoid_derivative(current_layer.output)
            prev_output = self.input
            if i != 0:
                prev_output = self.layers[i - 1].output
            current_layer.weights += self.learning_rate * hidden_diff @ prev_output
            output_diff = hidden_diff

if __name__ == "__main__":
    main()
