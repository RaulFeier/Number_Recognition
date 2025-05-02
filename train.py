import model, mnist_loader, pickle

training_data, validation_data, test_data = mnist_loader.modify_data()

net = model.Network([784, 64, 64, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

with open("network.pkl", "wb") as f:
    pickle.dump((net.weights, net.biases), f)

