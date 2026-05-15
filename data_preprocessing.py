import numpy as np
import pandas as pd


def initialize_data():
    df = pd.read_csv("datasets/Telco-Customer-Churn.csv")

    # Dealing with " " (Empty String) Fields
    df["TotalCharges"] = np.where(
        df["TotalCharges"].str.strip() == "", np.nan, df["TotalCharges"]
    ).astype(float)
    df.dropna(subset=["TotalCharges"], inplace=True)

    X: pd.DataFrame = df[df.columns[1:-1]]
    y = df["Churn"]

    # One Hot Encoding
    one_hot_fields = ["gender", "Partner", "Dependents", "PhoneService",
                      "MultipleLines", "OnlineSecurity", "OnlineBackup",
                      "DeviceProtection", "TechSupport", "StreamingTV",
                      "StreamingMovies", "PaperlessBilling",
                      "InternetService", "Contract", "PaymentMethod"]
    X = pd.get_dummies(X, columns=one_hot_fields,
                       drop_first=True, dtype=np.int8)
    y = pd.get_dummies(y, drop_first=True, dtype=np.int8)

    # Mean Value Normalization
    normalize_columns = ["TotalCharges", "MonthlyCharges", "tenure"]
    for col in normalize_columns:
        X[col] = (X[col] - X[col].mean()) / (X[col].max() - X[col].min())

    X = X.to_numpy()
    y = y.to_numpy()

    w = np.zeros(X.shape[1])
    b = 0

    return (X, y, w, b)

