import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_path = "train.csv"
data = pd.read_csv(file_path)

row = data.iloc[0]
print(row)

for i in row:
    print(i)

label = row["label"]
pixels = row.drop("label").values

image = pixels.reshape(28, 28)

plt.imshow(image, cmap="grey")
plt.title(f"Number: {label}")
plt.axis("off")

plt.savefig(f"figure{0}.jpg")
#plt.show()
