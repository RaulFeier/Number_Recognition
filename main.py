import model, pickle, gui
import tkinter as tk

with open("network.pkl", "rb") as f:
    weights, biases = pickle.load(f)

net = model.Network([784, 64, 64, 10])

net.weights = weights
net.biases = biases

root = tk.Tk()
drawer = gui.DigitDrawer(net, root)
drawer.run()