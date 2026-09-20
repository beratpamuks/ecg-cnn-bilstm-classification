# Technical Audit — AFib CNN-BiLSTM Project

This audit records the main differences between the supplied notebook/report and the cleaned GitHub version. It is intended to keep project claims aligned with the implemented code.

## 1. Dataset scope

The report describes the PhysioNet/CinC 2017 dataset as containing four broad label groups: Normal, AFib, Other rhythms, and Noisy signals. The supplied notebook filters the labels to `N` and `A` only. The cleaned project therefore documents the implemented task as **binary Normal vs AFib classification**.

## 2. Signal length

The supplied notebook standardizes every record to 2700 samples, described in the report as a 9-second window at 300 Hz. Longer signals are truncated and shorter signals are zero-padded. This behavior is preserved.

## 3. Filtering

The supplied implementation applies a 50 Hz notch filter followed by a 0.5–45 Hz fourth-order Butterworth band-pass filter. This behavior is preserved.

## 4. Normalization

The supplied implementation performs per-record Z-score normalization. This behavior is preserved.

## 5. Data splitting

The report describes a planned 70/15/15 train/validation/test organization. The supplied notebook does not implement that exact split. Instead, it performs an 80/20 stratified train/test split and uses `validation_split=0.2` inside `model.fit`. The cleaned README documents the actual implemented behavior rather than the report's planned split.

## 6. Model

The supplied notebook implements a CNN-BiLSTM binary classifier with:

- Conv1D 64, kernel 15
- Batch normalization
- Max pooling
- Conv1D 128, kernel 10
- Batch normalization
- Max pooling
- Dropout 0.3
- Bidirectional LSTM 64
- Dropout 0.4
- Dense 32 ReLU
- Dense 1 sigmoid

These settings are preserved in the cleaned source modules.

## 7. Training

The supplied notebook uses Adam with learning rate `0.001`, binary cross-entropy, batch size `64`, up to `50` epochs, early stopping, and ReduceLROnPlateau. These settings are preserved.

## 8. Grad-CAM correction

The original Grad-CAM implementation used `tf.argmax(preds[0])` on a single sigmoid output. For a binary sigmoid model, the cleaned version directly differentiates the sigmoid AFib output (`preds[:, 0]`). The heatmap normalization also uses `tf.math.divide_no_nan` to avoid division by zero.

## 9. Claims intentionally not presented as implemented results

The supplied report discusses targets such as recall >95%, precision >92%, and F1/accuracy >93.5%. These are treated as **project targets**, not achieved results. No performance values are placed in the GitHub Results section until a reproducible run produces and verifies them.

## 10. Future/architectural concepts not claimed as implemented

The report discusses quantization, Edge AI, clinical decision support integration, FDA SaMD/MDR considerations, end-to-end encryption, blockchain, and broader arrhythmia classification. These concepts are retained as future directions/context and are not represented as capabilities of the current code.

## 11. Data and privacy

The dataset is not committed to the repository. No patient-level data should be uploaded to GitHub. Runtime credentials and API keys must not be hard-coded into notebooks or source files.

## 12. Results policy

The repository intentionally starts without final metrics. After a reproducible experiment, the repository should record the exact dataset counts, software versions, hardware, training configuration, and evaluation outputs together with the resulting metrics.
