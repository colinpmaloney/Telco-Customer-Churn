import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_preprocessing import initialize_data
from logistic_regression import gradient_descent_vectorized, sigmoid

ITERS = 10000
LAMBDA = 0.0001
ALPHA = 1

X_train, y_train, X_test, y_test, w, b = initialize_data()

w, b, j_history = gradient_descent_vectorized(
    X_train, y_train, w, b, lambda_=LAMBDA, alpha=ALPHA, iters=ITERS)


fig, ax = plt.subplots(2, 1, figsize=(18, 6))
ax[0].plot(range(ITERS), j_history, label="Cost")
ax[0].set_title(f"Cost over {ITERS} iterations")


predictions = np.where(sigmoid(np.dot(X_test, w) + b) >= 0.5, .9, 0.1)

ax[1].scatter(range(len(predictions)), predictions, label="Predictions", c='b')
ax[1].scatter(range(len(predictions)), y_test, label="Reality", c='r')

plt.show()
