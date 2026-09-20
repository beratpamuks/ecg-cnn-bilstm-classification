import os
import numpy as np
import pandas as pd
import scipy.io as sio


def load_physionet_cinc2017(data_path: str, seq_len: int = 2700):
    """Load PhysioNet/CinC 2017 records and keep Normal (N) and AFib (A)."""
    print("Gerçek PhysioNet verileri yükleniyor...")

    csv_path = os.path.join(data_path, "REFERENCE.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} bulunamadı! Lütfen veri setini indirdiğinizden emin olun."
        )

    ref_df = pd.read_csv(csv_path, header=None, names=["record", "label"])
    ref_df = ref_df[ref_df["label"].isin(["N", "A"])]

    X, y = [], []
    for _, row in ref_df.iterrows():
        record_name = row["record"]
        label = row["label"]
        mat_file = os.path.join(data_path, f"{record_name}.mat")
        if not os.path.exists(mat_file):
            continue

        mat_data = sio.loadmat(mat_file)
        ecg_signal = mat_data["val"][0]

        if len(ecg_signal) >= seq_len:
            ecg_signal = ecg_signal[:seq_len]
        else:
            ecg_signal = np.pad(
                ecg_signal, (0, seq_len - len(ecg_signal)), mode="constant"
            )

        X.append(ecg_signal)
        y.append(0 if label == "N" else 1)

    print(f"Toplam {len(X)} adet gerçek EKG kaydı başarıyla yüklendi.")
    return np.asarray(X), np.asarray(y)
