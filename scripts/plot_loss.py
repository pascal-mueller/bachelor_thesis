import csv
import matplotlib.pyplot as plt
import numpy as np

train_loss = []
valid_loss = []

with open("training_losses.csv", "r") as f:
  reader = csv.reader(f)
  train_loss = np.array(list(reader), dtype="f")[:,0]

with open("validation_losses.csv", "r") as f:
  reader = csv.reader(f)
  valid_loss = np.array(list(reader), dtype="f")[:,0]

x = range(len(valid_loss))
plt.plot(x, train_loss, ".", color='green', label="Training loss")
plt.plot(x, valid_loss, ".", label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.savefig("losses.png")

plt.show()
