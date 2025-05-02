import model, pickle, gui

with open("network.pkl", "rb") as f:
    weights, biases = pickle.load(f)

net = model.Network([784, 64, 64, 10])

net.weights = weights
net.biases = biases

drawer = gui.DigitDrawer(net)
drawer.run()