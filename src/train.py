import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from .data_loader import load_physionet_cinc2017
from .preprocessing import apply_filters, apply_zscore_normalization
from .model import build_hybrid_model
from .evaluate import evaluate_model, save_gradcam_example

DATA_PATH = "./training2017"
SEQ_LEN = 2700
FS = 300
RANDOM_STATE = 42


def main():
    X_raw, y = load_physionet_cinc2017(DATA_PATH, seq_len=SEQ_LEN)
    X_filt = apply_filters(X_raw, fs=FS)
    X_norm = apply_zscore_normalization(X_filt)
    X_final = X_norm.reshape((X_norm.shape[0], SEQ_LEN, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Eğitim seti: {X_train.shape} | Test seti: {X_test.shape}")

    model = build_hybrid_model((SEQ_LEN, 1))
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=0.00001
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    os.makedirs("results", exist_ok=True)
    model.save("results/ecg_cnn_bilstm.keras")
    evaluate_model(model, X_test, y_test, results_dir="results")

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    save_gradcam_example(model, X_test, y_test, y_pred, seq_len=SEQ_LEN, results_dir="results")
    return history


if __name__ == "__main__":
    main()
