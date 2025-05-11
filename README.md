# Neural Network Implementation with TensorFlow 🧠

A low-level TensorFlow implementation of a neural network model using gradient descent optimization.

---

## 📋 Project Overview

This project implements a neural network model from scratch using TensorFlow's low-level API. The implementation includes:
- Custom forward propagation
- Gradient descent optimization
- Early stopping mechanism
- Model evaluation metrics

---

## 📊 Dataset

The dataset contains binary classification data with 5 input features. Distribution:
- Training set: 700 samples
- Validation set: 100 samples
- Test set: 200 samples

---

## 🏗️ Model Architecture

- Input layer: 5 neurons
- Hidden layer: 15 neurons with ReLU activation
- Output layer: 1 neuron with sigmoid activation
- Loss function: Mean Squared Error

---

## 🛠️ Implementation Details

### Key Components

1. Data Processing: 
   - 📈 Standard scaling normalization
   - 🔄 70-10-20 train-validation-test split

2. Model Functions:
   - 🔜 `forward()`: Implements forward propagation
   - 📉 `loss_fn()`: Calculates MSE loss
   - 🎯 `train()`: Single training step using gradient tape
   - 🔄 `fit()`: Training loop with early stopping

3. Training Parameters:
   - Learning rate: 0.01
   - Batch size: 16
   - Maximum epochs: 100
   - Early stopping patience: 5

---

## 📈 Results

- Final test loss: 0.268303
- Classification metrics:
  - Accuracy: 58%
  - Precision: 59%
  - Recall: 58%
  - F1-score: 55%

---

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Git

### Clone the Repository

```bash
git clone https://github.com/danishayman/Low-Level-Tensorflow.git
cd low-level-tensorflow
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

The required packages are:
- TensorFlow 2.x
- pandas
- numpy
- scikit-learn
- matplotlib
- jupyter

---

## 🚀 Usage

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open the `Low_Level_TF.ipynb` notebook:
```python
# Load and preprocess data
dataset = pd.read_csv('classification_dataset.csv')
```

3. Train model:
```python
train_losses, valid_losses = fit(model, optimizer, train_data, valid_data, 
                               weights, biases, epochs, batch_size, patience)
```

---

## 📦 Requirements

See `requirements.txt` for a complete list of dependencies.