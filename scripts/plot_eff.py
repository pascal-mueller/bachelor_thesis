import csv
import matplotlib.pyplot as plt
import numpy as np

eff_softmax = []
eff_usr = []

with open("efficiencies_softmax.csv", "r") as f:
  reader = csv.reader(f)
  eff_softmax = np.array(list(reader), dtype="f")[:,0]

with open("efficiencies_USR.csv", "r") as f:
  reader = csv.reader(f)
  eff_usr = np.array(list(reader), dtype="f")[:,0]

print(eff_usr)
print(eff_usr)

x = range(len(eff_usr))
plt.plot(x, eff_usr, ".", color='green', label="Efficiency USR")
#plt.plot(x, eff_softmax, ".", label="Efficiency Softmax")

plt.xlabel("Epoch")
plt.ylabel("Efficiency")

plt.legend()

plt.savefig("efficiencies.png")

plt.show()
