"""Conv1D + BiLSTM architecture and lazy TensorFlow loading."""

def tensorflow():
    try:
        import tensorflow as tf
        return tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required: python -m pip install -r requirements.txt") from exc



def build_model(num_classes, learning_rate=0.001):
    tf = tensorflow()
    layers = tf.keras.layers
    model = tf.keras.Sequential([
        layers.Input(shape=(32, 128)),
        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(momentum=0.99, epsilon=0.0001),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        layers.Conv1D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(momentum=0.99, epsilon=0.0001),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="fsl_conv1d_bilstm")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

