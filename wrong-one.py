import model, pickle, mnist_loader
import matplotlib.pyplot as plt

with open("network.pkl", "rb") as f:
    weights, biases = pickle.load(f)

net = model.Network([784, 64, 64, 10])

net.weights = weights
net.biases = biases

training_data, validation_data, test_data = mnist_loader.modify_data()


def evaluate():
    for (x, y) in test_data:
        output = net.feedforward(x)
        index, mx = 0
        # for i in range(11):
        #     if output[i] > mx:
        #         mx = output[i]
        #         index = i

        # if y[index] != 1:
        #     print("wrong")
            
evaluate()
#output = net.predict()
#print(output)