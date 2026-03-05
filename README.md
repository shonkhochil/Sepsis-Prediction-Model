# Sepsis-Prediction-Model
Early Sepsis detetion during Inpatient Encounter 

# Sepsis Prediction with Hybrid LSTM-BERT

## Project Overview
This project implements a **Hybrid Deep Learning Model** for the early detection of sepsis from Electronic Health Records (EHR). It combines time-series physiological data with unstructured clinical notes to predict sepsis onset up to **8-9 hours in advance**.

## 📊 Dataset

*   **Source**: Synthetic data generated within the notebook (no external dependency).
*   **Size**: 10,000 unique patients.
*   **Features**:
    *   **Vitals**: Heart Rate, Systolic BP, Temperature (Hourly).
    *   **Labs**: Lactate, Creatinine, Bilirubin, Platelets (Simulated draws).
    *   **Notes**: Unstructured clinical text embeddings (Bio_ClinicalBERT).
*   **Ground Truth**: Defined using Sepsis-3 criteria (Suspected Infection + SOFA Score ≥ 2).

## 🧠 Model Architecture

The `HybridModel` allows for multimodal learning by fusing structured and unstructured data:
1.  **LSTM Branch**: Processes sequential numeric data (Vitals & Labs).
2.  **BERT Branch**: Uses pre-computed embeddings from `emilyalsentzer/Bio_ClinicalBERT` for clinical notes.
3.  **Fusion Layer**: Concatenates LSTM hidden states with text embeddings before passing to a fully connected classification head.

## 🛠️ Optimization Techniques

To ensure robustness and calibration, the training pipeline includes:
*   **Early Stopping**: Monitors validation loss to prevent overfitting (Patience=5).
*   **LR Scheduler**: `ReduceLROnPlateau` decays learning rate when loss stagnates.
*   **Weighted Random Sampling**: Addresses class imbalance (25% prevalence) by ensuring training batches contain a 50/50 mix of positive and negative cases.

## 📈 Key Results

*   **Lead Time**: The model successfully predicts sepsis onset an average of **~9 hours** in advance.
*   **Calibration**: After balancing classes, the model maintains high Recall (>95%) across a wide decision threshold window (**0.30 - 0.75**).
*   **Optimal Threshold**: **0.50** was validated as the most robust operational threshold.

## 💻 How to Run

1.  **Install Dependencies**:
    ```bash
    pip install torch pandas numpy scikit-learn transformers faker matplotlib
    ```
2.  **Execute Notebook**: Run the cells sequentially. The notebook handles data generation, preprocessing, training, and evaluation automatically.

## License
MIT
