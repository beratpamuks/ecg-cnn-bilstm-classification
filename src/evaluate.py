import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from .gradcam import make_gradcam_heatmap_1d


def evaluate_model(model, X_test, y_test, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    report = classification_report(
        y_test, y_pred, target_names=["Normal", "AFib"]
    )
    print("\n--- Sınıflandırma Raporu ---")
    print(report)

    with open(os.path.join(results_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "AFib"],
        yticklabels=["Normal", "AFib"],
    )
    plt.title("Confusion Matrix (Real ECG Data)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    return y_pred_prob, y_pred


def save_gradcam_example(model, X_test, y_test, y_pred, seq_len=2700, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    afib_indices = np.where((y_test == 1) & (y_pred == 1))[0]
    if len(afib_indices) == 0:
        print("Doğru sınıflandırılmış AFib örneği bulunamadı; Grad-CAM grafiği üretilmedi.")
        return

    test_idx = afib_indices[0]
    sample_signal = X_test[test_idx : test_idx + 1]
    heatmap = make_gradcam_heatmap_1d(sample_signal, model)
    heatmap_resized = np.interp(
        np.linspace(0, 1, seq_len), np.linspace(0, 1, len(heatmap)), heatmap
    )

    plt.figure(figsize=(12, 4))
    plt.plot(sample_signal[0].flatten(), alpha=0.7, label="Z-score ECG signal")
    plt.imshow(
        heatmap_resized[np.newaxis, :],
        cmap="jet",
        aspect="auto",
        alpha=0.4,
        extent=[0, seq_len, np.min(sample_signal), np.max(sample_signal)],
    )
    plt.colorbar(label="AI attention intensity")
    plt.title("1-D Grad-CAM for an AFib Prediction")
    plt.xlabel("Time step")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "gradcam_afib_example.png"), dpi=200)
    plt.close()
