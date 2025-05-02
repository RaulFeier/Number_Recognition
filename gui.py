import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import matplotlib.pyplot as plt

class DigitDrawer:
    def __init__(self, model):
        self.model = model

        self.window = tk.Tk()
        self.window.title("Digit Recognizer")

        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = tk.Canvas(self.window, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        btn_frame = tk.Frame(self.window)
        btn_frame.pack()

        predict_btn = tk.Button(btn_frame, text="Predict", command=self.predict_digit)
        predict_btn.grid(row=0, column=0, padx=10, pady=10)

        clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear_canvas)
        clear_btn.grid(row=0, column=1, padx=10, pady=10)

        self.result_label = tk.Label(self.window, text="Draw a digit", font=("Arial", 20))
        self.result_label.pack(pady=10)

    def paint(self, event):
        x, y = event.x, event.y
        r = 4
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill=255)
        self.result_label.config(text="Draw a digit")

    def preprocess_image(self):
        img = self.image
        inverted = ImageOps.invert(img)
        bbox = inverted.getbbox()

        if bbox is None:
            return np.zeros((784, 1))  # blank canvas

         # Crop to digit area
        cropped = inverted.crop(bbox)

        # Resize to 20x20 with antialiasing
        resized = cropped.resize((20, 20), Image.LANCZOS)

        # Paste into 28x28 canvas (centered)
        final_img = Image.new("L", (28, 28), 0)
        upper_left = ((28 - 20) // 2, (28 - 20) // 2)
        final_img.paste(resized, upper_left)

        # Convert to array
        img_array = np.array(final_img).astype(np.float32)
        img_array /= 255.0  # normalize to [0,1]
        img_array = img_array.reshape(784, 1)

        plt.imshow(img_array.reshape(28, 28), cmap='gray')
        plt.title("Preprocessed GUI Input")
        plt.show()

        return img_array 

    def predict_digit(self):
        img = self.preprocess_image()
        output = self.model.predict(img)
        print(output)
        prediction = int(np.argmax(output))
        self.result_label.config(text=f"Predicted: {prediction}")

    def run(self):
        self.window.mainloop()
