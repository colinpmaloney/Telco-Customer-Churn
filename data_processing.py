import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def initialize_data():
    df = pd.read_csv("datasets/Telco-Customer-Churn.csv")
    df.drop(columns=["customerID"], inplace=True)

    # Dealing with " " (Empty String) Fields
    df["TotalCharges"] = np.where(
        df["TotalCharges"].str.strip() == "", np.nan, df["TotalCharges"]
    ).astype(float)
    df.dropna(subset=["TotalCharges"], inplace=True)

    df = df.reset_index(drop=True)

    # One Hot Encoding
    one_hot_fields = ["gender", "Partner", "Dependents", "PhoneService",
                      "MultipleLines", "OnlineSecurity", "OnlineBackup",
                      "DeviceProtection", "TechSupport", "StreamingTV",
                      "StreamingMovies", "PaperlessBilling",
                      "InternetService", "Contract", "PaymentMethod", "Churn"]
    df = pd.get_dummies(df, columns=one_hot_fields,
                        drop_first=True, dtype=np.int8)

    # Create a 80/20 Train/Test split
    train_df = df.sample(frac=0.8, random_state=817)
    test_df = df.drop(train_df.index)

    X_train = train_df.drop(columns=["Churn_Yes"])
    y_train = train_df["Churn_Yes"].to_numpy()

    X_test = test_df.drop(columns=["Churn_Yes"])
    y_test = test_df["Churn_Yes"].to_numpy()

    # Z-Score Normalization
    normalize_columns = ["TotalCharges", "MonthlyCharges", "tenure"]
    for col in normalize_columns:
        train_mean = X_train[col].mean()
        train_std = X_train[col].std()

        X_train[col] = (X_train[col] - train_mean) / train_std
        X_test[col] = (X_test[col] - train_mean) / train_std

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    w = np.zeros(X_train.shape[1])
    b = 0

    return (X_train, y_train, X_test, y_test, w, b)


def visualize_data_individually(y_test, predictions, model_name, threshold=0.5):
    y_hat = np.where(predictions >= threshold, 1, 0)

    fig, ax = plt.subplots(2, 1, figsize=(12, 5))
    fig.canvas.manager.set_window_title(f"Telco Churn Rate - {model_name}")
    cm = confusion_matrix(y_test, y_hat)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0],
                xticklabels=['Stayed (0)', 'Churned (1)'],
                yticklabels=['Stayed (0)', 'Churned (1)'])
    ax[0].set_title(
        f"Confusion Matrix (Error Breakdown) - {model_name}", fontsize=16)
    ax[0].set_xlabel("Predicted Label", fontsize=12)
    ax[0].set_ylabel("True Label", fontsize=12)

    sns.histplot(x=predictions.flatten(), hue=y_test, element="step", stat="density",
                 common_norm=False, kde=True, ax=ax[1], palette=['blue', 'red'])
    ax[1].axvline(x=threshold, color='black', linestyle='--',
                  linewidth=2, label=f'Threshold ({threshold})')
    ax[1].set_title(
        f"Network Confidence & Threshold Analysis - {model_name}", fontsize=16)
    ax[1].set_xlabel("Sigmoid Output Probability ($P(Y=1|X)$)", fontsize=12)
    ax[1].set_ylabel("Density", fontsize=12)
    ax[1].legend(['Threshold', 'Actually Churned (1)', 'Actually Stayed (0)'])

    plt.tight_layout()
    plt.show()


def visualize_data(y_test, log_predictions, nn_predictions, threshold=0.5):
    log_y_hat = np.where(log_predictions >= threshold, 1, 0)
    nn_y_hat = np.where(nn_predictions >= threshold, 1, 0)

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.subplots_adjust(wspace=10, hspace=20)

    fig.canvas.manager.set_window_title(
        "Telco Churn Rate"
    )

    plot_error_confidence(y_test, log_predictions,
                          log_y_hat, threshold, "LR", ax, 0)
    plot_error_confidence(y_test, nn_predictions,
                          nn_y_hat, threshold, "NN", ax, 1)

    plt.tight_layout()
    plt.show()


def plot_error_confidence(y_test, predictions, y_hat, threshold, model_name, ax, col):
    cm = confusion_matrix(y_test, y_hat)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0][col],
                xticklabels=['Stayed (0)', 'Churned (1)'],
                yticklabels=['Stayed (0)', 'Churned (1)'])
    ax[0][col].set_title(
        f"Confusion Matrix (Error Breakdown) - {model_name}", fontsize=14)
    ax[0][col].set_xlabel("Predicted Label", fontsize=12)
    ax[0][col].set_ylabel("True Label", fontsize=12)

    sns.histplot(x=predictions.flatten(), hue=y_test, element="step", stat="density",
                 common_norm=False, kde=True, ax=ax[1][col], palette=['blue', 'red'])
    ax[1][col].axvline(x=threshold, color='black', linestyle='--',
                       linewidth=2, label=f'Threshold ({threshold})')
    ax[1][col].set_title(
        f"Network Confidence & Threshold Analysis - {model_name}", fontsize=14)
    ax[1][col].set_xlabel(
        "Sigmoid Output Probability ($P(Y=1|X)$)", fontsize=12)
    ax[1][col].set_ylabel("Density", fontsize=12)
    ax[1][col].legend(
        ['Threshold', 'Actually Churned (1)', 'Actually Stayed (0)'])
