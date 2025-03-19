# Web Attack Detection System

A machine learning-based threat detection system for distributed applications that leverages deep learning models to identify various web attacks.

## Overview

This system can detect different types of web attacks:

- SQL Injections
- Cross-Site Scripting (XSS) 
- Path Traversal
- Denial of Service (DoS)

The system uses models trained on synthetic datasets generated from attack patterns in the `patterns` directory.

## Getting Started

### Requirements

- OS: Windows
- Python 3.12.7 (CUDA/Tensorflow supported) or higher (CUDA/Tensorflow not supported)
- PyTorch
- TensorFlow
- scikit-learn
- NumPy
- Pandas

### Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Generating Datasets

To generate datasets for all attack types:

```bash
python generate_dataset.py --attack-type all --num-samples 5000 --malicious-ratio 0.5
```

To generate a dataset for a specific attack type:

```bash
python generate_dataset.py --attack-type sql_injection --num-samples 5000 --malicious-ratio 0.5
```

### Training Models

To train models for all attack types:

```bash
python train_all_models.py --num-samples 5000 --malicious-ratio 0.5 --device auto
```

To train a model for a specific attack type:

```bash
python train_all_models.py --attack-type xss --num-samples 5000 --malicious-ratio 0.5 --device auto
```

## Model Training Details

The system uses a universal trainer that supports multiple model types:

- Logistic Regression
- Random Forest
- Support Vector Machines (SVM)
- Multi-Layer Perceptron (MLP)
- Artificial Neural Network (ANN) with PyTorch

By default, the system uses ANN models with TF-IDF vectorization, which have shown the best performance for detecting web attacks.

### Hardware Acceleration

The training system automatically detects and uses available hardware acceleration:

- CUDA for NVIDIA GPUs
- MPS for Apple Silicon
- Multi-core CPU support

## Adding New Attack Patterns

1. Add new patterns to the appropriate file in the `patterns/` directory
2. Run the dataset generation and model training scripts
