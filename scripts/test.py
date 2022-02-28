import numpy as np
import pycbc
import torch

from src.reg_BCELoss import reg_BCELoss

loss_fn = reg_BCELoss(dim=2)

inputs = np.array([[1,2],[3,4]])
targets = np.array([[1.1, 2.2], [3.3, 4.4]])

eps = 1e-6
dim = 2

transformed_input = eps + dim*inputs

print(inputs)
print(transformed_input)
