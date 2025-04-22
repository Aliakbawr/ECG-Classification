# ECG Signal Classification with Deep Learning

A portfolio-style project showcasing two deep learning approaches for ECG signal classification:

- **Transformer-based Multiclass Classifier**
- **Bidirectional GRU Binary Classifier**

This repository is built to demonstrate signal preprocessing, model design, and performance evaluation on real-world biomedical time series data.

---

## Project 1: Multiclass ECG Classification using Transformer

This model classifies ECG beats into multiple heartbeat categories using a Transformer-based architecture.

### Model Architecture

- **Conv1D** for initial local feature extraction
- **Transformer Encoder Layer**
  - MultiHeadAttention + LayerNorm + Dense + Dropout
- **Flatten + Dense** for final classification

> Achieves strong accuracy on the PhysioNet heartbeat classification dataset.

### File

`transformer-multiclass.ipynb`

- Signal loading & preprocessing
- Sequence modeling with Transformer block
- Training + evaluation with performance visualization
- Confusion matrix and classification report

###  Dataset Info

- **Source**: PhysioNet MIT-BIH Arrhythmia Dataset  
- **Classes**: Multiple heartbeat types  
- **Input shape**: Fixed-length 1D signal sequences  

---

##  Project 2: Binary ECG Classification using Bidirectional GRU

A complementary project that classifies ECG signals into **normal vs. abnormal** using a Bidirectional GRU network.

### Model Architecture

- **Bidirectional GRU (64)** → **BatchNorm + Dropout**
- **GRU (32)** → **Dense(1, sigmoid)**
- Binary output with sigmoid activation

### File

`binary-classifier.ipynb`

Includes:

- Data loading (PTB Diagnostic dataset)
- Signal smoothing via **Savitzky-Golay filter**
- Data augmentation (Gaussian noise)
- Model training + callbacks (EarlyStopping, ReduceLROnPlateau)
- Accuracy/loss plots, classification report, confusion matrix
- Model saving (`ecg_model.h5`)

### Dataset Info

- **Source**: PTB Diagnostic ECG Dataset  
- **Files**: `ptbdb_normal.csv`, `ptbdb_abnormal.csv`  
- **Labels**:  
  - `0` → Normal  
  - `1` → Abnormal

---

##  Tech Stack

- Python, NumPy, Pandas
- TensorFlow / Keras
- Scikit-learn
- Matplotlib & Seaborn
- Google Colab

---

Both models showcase:

- Efficient preprocessing pipelines
- Robust architectures for time-series data
- Clear visualizations and evaluations

---
## Folder Structure

```bash
ecg-classification-project/
│
├── transformer-multiclass.ipynb        # Transformer model for multiclass ECG classification
├── binary-classifier.ipynb             # Bidirectional GRU for binary classification
├── data/
│   ├── ptbdb_normal.csv
│   └── ptbdb_abnormal.csv
├── saved_model/
│   └── ecg_model.h5
├── README.md
