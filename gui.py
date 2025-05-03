import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
# from tkinter import *
import numpy as np
import matplotlib.pyplot as plt

class DigitDrawer:
    def __init__(self, model, master):
        self.master = master
        self.master.title("Digit Recognizer")
        self.model = model

        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=2, pady=2, padx=2)

        self.label = tk.Label(self.master, text="Draw a digit", font=("Helvetica", 16))
        self.label.grid(row=1, column=0, columnspan=2)

        self.button_predict = tk.Button(self.master, text="Predict", command=self.predict_digit)
        self.button_predict.grid(row=2, column=0, sticky="ew")

        self.button_clear = tk.Button(self.master, text="Clear", command=self.clear_canvas)
        self.button_clear.grid(row=2, column=1, sticky="ew")

        # For drawing on canvas
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)


    def draw_lines(self, event):
        x, y = event.x, event.y
        r = 8  # thickness of the stroke
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill="white")
        self.label.configure(text="Draw a digit")

    def preprocess_image(self):
        image_resized = self.image.resize((28, 28)) 

        image_array = np.array(image_resized).astype(np.float32)
        image_array = 255.0 - image_array  # manually invert to get white digits on black background
        image_array /= 255.0  # normalize to [0, 1]

        # plt.imshow(image_array, cmap='gray')
        # plt.title("Preprocessed GUI Input")
        # plt.show()

        # debug_img = Image.fromarray((image_array * 255).astype(np.uint8))
        # debug_img.show(title="Preprocessed 28x28 Digit")

        image_flattened = image_array.reshape((784, 1)) 

        output = self.model.predict(image_flattened)
        #print(output)

        return int(np.argmax(output)), max(output) 

    def predict_digit(self):
        digit, confidence = self.preprocess_image()
        self.label.configure(text=f"Prediction: {digit} ({int(confidence * 100)}%)")

        # img = self.preprocess_image()
        # output = self.model.predict(img)
        # print(np.argmax(output), int(max(output) + 100))
        # prediction = int(np.argmax(output))
        # self.result_label.config(text=f"Predicted: {prediction}")

    def run(self):
        self.master.mainloop()