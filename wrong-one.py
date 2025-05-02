import model, pickle, mnist_loader
import matplotlib.pyplot as plt
import numpy as np

with open("network.pkl", "rb") as f:
    weights, biases = pickle.load(f)

net = model.Network([784, 64, 64, 10])

net.weights = weights
net.biases = biases

training_data, validation_data, test_data = mnist_loader.modify_data()


def evaluate():
    count  = 0
    for (x, y) in test_data:
        output = net.predict(x)
        p = int(np.argmax(output))
        if y != p:
            count += 1
            plt.imshow(x.reshape(28, 28), cmap='gray')
            plt.title(f"{y, p}")
            plt.show()
    return count

print(evaluate())
      
evaluate()