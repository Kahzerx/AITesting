from helpers import keras_helpers as kh
import numpy as np
import matplotlib.pyplot as plt

model = kh.GenerateDense([4, 8, 1], ["sigmoid", "tanh", "sigmoid"], "mse", "adam")
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

plt.plot(kh.FitModel(model, X, Y, 0.9))
plt.show()

plt.plot(model.predict_proba(X))
plt.show()
