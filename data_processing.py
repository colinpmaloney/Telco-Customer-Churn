import numpy as np
import pandas as pd


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
