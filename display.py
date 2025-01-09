import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_path = "train.csv"
data = pd.read_csv(file_path)

print(data.head())

row = data.iloc[3]

label = row["label"]
pixels = row.drop("label").values

image = pixels.reshape(28, 28)

plt.imshow(image, cmap="gray")
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()
