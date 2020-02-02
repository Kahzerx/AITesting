from keras.models import Sequential
from keras.layers import Dense
import numpy as np


def GenerateDense(topology, activations, loss, optimizer):
    model = Sequential()
    for i in range(len(topology)):
        model.add(Dense(units=topology[i], activation=activations[i]))

    model.compile(loss=loss, optimizer=optimizer, metrics=["acc"])

    return model


def FitModel(model, X, Y, min_acc):
    acc = []
    hist = model.fit(np.array(X), np.array(Y), epochs=10000)
    acc.append(hist.history.get("acc"))
    while np.amax(acc) < min_acc:
        hist = model.fit(np.array(X), np.array(Y), epochs=10000)
        acc.append(hist.history.get("acc"))
    return np.array(acc).flatten()
