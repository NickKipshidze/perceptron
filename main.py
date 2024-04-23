import numpy as np
import os

def convert_to_digits(string):
    return np.array([1 if c == "#" else 0 for c in string], dtype=np.float64)

def load_dataset(dir):
    return [(convert_to_digits(open(f"{dir}/{file}", "r").read().replace("\n", "")), 1 if file.split("-")[0] == "rect" else 0) for file in os.listdir(dir)]

shapes_train = load_dataset("dataset")
shapes_test = load_dataset("testset")

class LayerDense():
    def __init__(self, inputs, neurons):
        self.weights = np.random.randn(inputs) * 0.1
        self.biases = np.random.randn(neurons) * 0.1
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationSigmoid():
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
    
    def backward(self, x):
        self.output = x * (1 - x)

layer1 = LayerDense(32 * 32, 1)
activation1 = ActivationSigmoid()

inputs, labels = zip(*shapes_train)

learning_rate = 0.3

for iteration in range(1_000):
    for input, label in zip(inputs, labels):
        layer1.forward(input)
        activation1.forward(layer1.output)
        prediction = activation1.output
        
        loss = label - prediction
        activation1.backward(prediction)
        adjustments = loss * activation1.output
        
        layer1.weights += learning_rate * adjustments * input
        layer1.biases += learning_rate * adjustments
        
    print(f"Iteration: {iteration + 1}, Loss: {loss}")
    
inputs, labels = zip(*shapes_test)
predictions = []

for input, label in zip(inputs, labels):
    layer1.forward(input)
    activation1.forward(layer1.output)
    prediction = activation1.output
    predictions.append(np.round(prediction[0]))

print(np.array(predictions, dtype=np.int32))
print(np.array(labels, dtype=np.int32))
accuracy = np.mean(np.array(predictions) == np.array(labels)) * 100

print(f"Accuracy: {accuracy:.2f}%")