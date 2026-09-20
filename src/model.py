import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
)


def build_hybrid_model(input_shape):
    """Build the CNN-BiLSTM binary classifier used in the project."""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=15, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(
            filters=128,
            kernel_size=10,
            activation="relu",
            padding="same",
            name="last_conv",
        ),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.4),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall")],
    )
    return model
