import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from data_processing import initialize_data, visualize_data
from logistic_regression import gradient_descent_vectorized, sigmoid
from neural_network import create_model

LR_ITERS = 10000
LR_LAMBDA = 0.0001
LR_ALPHA = 1

NN_ITERS = 300
NN_ALPHA = 0.001
NN_LAMBDA = 0


if __name__ == "__main__":
    # X_train, y_train, X_test, y_test, w, b = initialize_data()
    # w, b = gradient_descent_vectorized(
    #     X_train, y_train, w, b, lambda_=LR_LAMBDA, alpha=LR_ALPHA, iters=LR_ITERS
    # )
    # log_predictions = sigmoid(np.dot(X_test, w) + b)

    X_train, y_train, X_test, y_test, _, _ = initialize_data()

    start_time = time.time()
    model, history = create_model(
        X_test, y_test, X_train, y_train,
        epochs=NN_ITERS, alpha=NN_ALPHA, lambda_=NN_LAMBDA
    )
    training_latency = time.time() - start_time

    nn_predictions = model.predict(X_test)
    nn_y_hat = (nn_predictions >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, nn_y_hat)
    f1 = f1_score(y_test, nn_y_hat)
    cost_j = history.history["val_loss"][-1]

    print("\n========== Neural Network Metrics ==========")
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  F1-Score:          {f1:.4f}")
    print(f"  Training Latency:  {training_latency:.2f}s")
    print(f"  Model Cost (J):    {cost_j:.4f}")
    print("============================================\n")

    # visualize_data(y_test, log_predictions, nn_predictions)
