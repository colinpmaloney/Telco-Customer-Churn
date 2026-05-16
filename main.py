import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_processing import initialize_data, visualize_data, create_model
from logistic_regression import gradient_descent_vectorized, sigmoid

LR_ITERS = 10000
LR_LAMBDA = 0.0001
LR_ALPHA = 1

NN_ITERS = 100
NN_ALPHA = 0.001


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, w, b = initialize_data()
    w, b = gradient_descent_vectorized(
        X_train, y_train, w, b, lambda_=LR_LAMBDA, alpha=LR_ALPHA, iters=LR_ITERS
    )
    log_predictions = sigmoid(np.dot(X_test, w) + b)

    X_train, y_train, X_test, y_test, w, b = initialize_data()
    model = create_model(
        X_test, y_test, X_train, y_train,
        epochs=NN_ITERS, alpha=NN_ALPHA
    )
    nn_predictions = model.predict(X_test)

    visualize_data(y_test, log_predictions, nn_predictions)
