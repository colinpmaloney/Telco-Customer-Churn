import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from data_processing import initialize_data
from logistic_regression import gradient_descent_vectorized, sigmoid, cost_function_vectorized
from neural_network import create_model
from visualization import visualize_data

THRESHOLD = 0.57

LR_ITERS = 10000
LR_LAMBDA = 0.0001
LR_ALPHA = 1

NN_ITERS = 300
NN_ALPHA = 0.001
NN_LAMBDA = 0.001

DEV_MODE = False

if __name__ == "__main__":
    # ADD CV stfuf
    X_train, y_train, X_cv, y_cv, X_test, y_test, w, b = initialize_data()
    X_set = X_cv if DEV_MODE else X_test
    y_set = y_cv if DEV_MODE else y_test

    # --- Logistic Regression ---
    lr_start = time.time()
    w, b = gradient_descent_vectorized(
        X_train, y_train, w, b, lambda_=LR_LAMBDA, alpha=LR_ALPHA, iters=LR_ITERS
    )
    lr_latency = time.time() - lr_start

    log_predictions = sigmoid(np.dot(X_set, w) + b)
    log_y_hat = (log_predictions >= THRESHOLD).astype(int)
    lr_accuracy = accuracy_score(y_set, log_y_hat)
    lr_f1 = f1_score(y_set, log_y_hat)
    lr_cost_j = cost_function_vectorized(X_set, y_set, w, b, LR_LAMBDA)

    print("\n========== Logistic Regression Metrics ==========")
    print(f"  Accuracy:          {lr_accuracy:.4f}")
    print(f"  F1-Score:          {lr_f1:.4f}")
    print(f"  Training Latency:  {lr_latency:.2f}s")
    print(f"  Model Cost (J):    {lr_cost_j:.4f}")
    print("==================================================\n")

    # --- Neural Network ---
    nn_start = time.time()
    model, history = create_model(
        X_cv, y_cv, X_train, y_train,
        epochs=NN_ITERS, alpha=NN_ALPHA, lambda_=NN_LAMBDA
    )
    nn_latency = time.time() - nn_start

    nn_predictions = model.predict(X_set)
    nn_y_hat = (nn_predictions >= THRESHOLD).astype(int)
    nn_accuracy = accuracy_score(y_set, nn_y_hat)
    nn_f1 = f1_score(y_set, nn_y_hat)
    nn_cost_j = history.history["val_loss"][-1]

    print("\n========== Neural Network Metrics ==========")
    print(f"  Accuracy:          {nn_accuracy:.4f}")
    print(f"  F1-Score:          {nn_f1:.4f}")
    print(f"  Training Latency:  {nn_latency:.2f}s")
    print(f"  Model Cost (J):    {nn_cost_j:.4f}")
    print("============================================\n")

    visualize_data(y_set, log_predictions, nn_predictions, threshold=THRESHOLD)
