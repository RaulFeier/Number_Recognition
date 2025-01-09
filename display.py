import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_path = "train.csv"
data = pd.read_csv(file_path)

print(data.head())

for i in range(10):
    row = data.iloc[i]

    label = row["label"]
    pixels = row.drop("label").values

    image = pixels.reshape(28, 28)

    plt.imshow(image, cmap="grey")
    plt.title(f"Number: {label}")
    plt.axis("off")

    plt.savefig(f"figure{i}.jpg")
    #plt.show()
