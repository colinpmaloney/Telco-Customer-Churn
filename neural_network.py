import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)


def create_model(X_test, y_test, X_train, y_train, epochs, alpha, lambda_):
    model = Sequential([
        tf.keras.Input(shape=(30,)),
        Dense(units=64, activation="relu", kernel_regularizer=l2(lambda_)),
        Dropout(0.3),
        Dense(units=32, activation="relu", kernel_regularizer=l2(lambda_)),
        Dropout(0.1),
        Dense(units=1, activation="sigmoid"),
    ],
        name="Churn-Model"
    )

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(alpha)
    )

    # Calculate weights based on actual class distributions
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced', classes=classes, y=y_train.flatten())
    class_weight_dict = dict(zip(classes, weights))

    history = model.fit(
        X_train, y_train, epochs=epochs,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        callbacks=[early_stop]
    )

    return model, history
