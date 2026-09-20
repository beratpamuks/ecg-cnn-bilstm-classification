# Deep Learning-Based AFib Classification from Single-Lead ECG Signals

A research/educational prototype for classifying **Normal rhythm (N)** versus **Atrial Fibrillation (AFib, A)** from short single-lead ECG recordings using a hybrid **1D-CNN + Bidirectional LSTM** architecture.

> **Medical-use disclaimer:** This repository is a research/educational prototype. It is not a medical device and must not be used as a clinical diagnostic system.

## Project scope

The supplied project materials describe a broader wearable-ECG vision involving continuous monitoring, clinical decision support, explainable AI, edge deployment, and future integration with healthcare systems. The current code implementation is narrower: it loads the PhysioNet/CinC 2017 training data, keeps the Normal (`N`) and AFib (`A`) labels, preprocesses the ECG signals, trains a binary CNN-BiLSTM classifier, evaluates it with classification metrics/confusion matrix, and generates a 1-D Grad-CAM visualization.

## Dataset

The implementation downloads the **PhysioNet/CinC Challenge 2017** training data and uses the `REFERENCE.csv` labels. The current implementation filters the original labels to:

- `N` → Normal rhythm → class `0`
- `A` → Atrial Fibrillation → class `1`

The other labels described in the project report are therefore **not part of the current binary classifier**.

The dataset is downloaded at runtime and is intentionally not committed to this repository.

## Preprocessing

The implemented pipeline is:

1. Load `.mat` ECG recordings.
2. Standardize each signal to 2700 samples (truncation or zero-padding).
3. Apply a 50 Hz notch filter.
4. Apply a 0.5–45 Hz fourth-order Butterworth band-pass filter.
5. Apply per-record Z-score normalization.
6. Reshape the input to `(samples, 2700, 1)`.
7. Split into training and test sets with an 80/20 stratified split.
8. Use a further 20% validation split from the training data during `model.fit`.

## Model architecture

```text
Input: 2700 × 1
        │
        ▼
Conv1D (64 filters, kernel 15)
        │
Batch Normalization
        │
MaxPooling1D
        │
Conv1D (128 filters, kernel 10)
        │
Batch Normalization
        │
MaxPooling1D
        │
Dropout (0.30)
        │
Bidirectional LSTM (64)
        │
Dropout (0.40)
        │
Dense (32, ReLU)
        │
Dense (1, Sigmoid)
```

Training configuration in the supplied implementation:

- Optimizer: Adam
- Learning rate: `0.001`
- Loss: Binary Cross-Entropy
- Batch size: `64`
- Maximum epochs: `50`
- Early stopping: validation loss, patience `5`
- ReduceLROnPlateau: factor `0.5`, patience `2`, minimum LR `1e-5`
- Random state for train/test split: `42`

## Explainability

The project includes a 1-D Grad-CAM implementation intended to visualize which temporal regions contribute to an AFib prediction. The cleaned implementation explicitly uses the model's sigmoid AFib output rather than applying a class-index operation to a single-output sigmoid tensor.

## Results

**Results are intentionally not included yet.** A reproducible training run should be completed before reporting precision, recall, F1-score, or other performance values in this repository.

After a verified run, this section can be expanded with:

- classification report
- confusion matrix
- training/validation curves
- Grad-CAM examples
- exact dataset split counts
- environment and hardware information

## Repository structure

```text
ecg-cnn-bilstm-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
├── notebooks/
│   └── AFib_Tespiti_Clean.ipynb
├── docs/
│   └── TECHNICAL_AUDIT.md
└── results/
    └── .gitkeep
```

## Reproducibility

The current notebook is designed for a Colab-style environment because the original workflow downloads the PhysioNet dataset at runtime. For a full experiment, use a GPU-enabled environment and record the final software versions, hardware, dataset counts, training duration, and resulting metrics.

## Future work

Potential future directions described in the supplied project report include broader arrhythmia classification, explainability improvements, edge deployment, and clinical decision-support integration. These are **future directions**, not capabilities claimed by the current implementation.
