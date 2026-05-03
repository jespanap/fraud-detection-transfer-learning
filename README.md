# Fraud Detection with Transfer Learning

**Universidad EAFIT — Neural Networks and Deep Learning (2026-1)**

**Author:** Jaider España  

**Repository purpose:** project proposal — first deliverable

> A deep-learning-based fraud detection system that leverages transfer learning
> on tabular financial data. The project focuses on improving detection accuracy
> in highly imbalanced transaction datasets by reusing representations learned
> from large-scale pretraining and adapting them to fraud classification.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Data](#2-data)
3. [Base Architecture](#3-base-architecture)
4. [Proposed Architecture](#4-proposed-architecture)
5. [Data-Source Considerations](#5-data-source-considerations)
6. [Transfer Learning Strategy](#6-transfer-learning-strategy)
7. [Roadmap](#7-roadmap)
8. [References](#8-references)

---

## 1. Problem Statement

### 1.1. Domain

Fraud detection in financial transactions is a binary classification problem
where the goal is to identify whether a given transaction is legitimate or fraudulent.

Key challenges:

| Challenge            | Description |
|---------------------|------------|
| Class imbalance     | Fraud cases < 1% of total transactions |
| Concept drift       | Fraud patterns evolve over time |
| Feature complexity  | Non-linear relationships between variables |
| Real-time constraint| Predictions must be fast |

---

### 1.2. Mathematical formulation

We define a supervised learning problem:

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}
$$

- $x_i \in \mathbb{R}^d$ — transaction feature vector  
- $y_i \in \{0,1\}$ — label (0 = normal, 1 = fraud)

The model learns:

$$
f_\theta(x) = P(y=1 \mid x)
$$

Objective:

$$
\mathcal{L}(\theta) = - \sum_{i=1}^{N} \left[ y_i \log f_\theta(x_i) + (1-y_i)\log(1 - f_\theta(x_i)) \right]
$$

---

### 1.3. Why this is a deep-learning problem

1. **Highly non-linear interactions** between transaction features  
2. **Severe imbalance** makes simple models biased toward majority class  
3. **Feature reuse potential** — patterns learned in one dataset can transfer  
4. **Temporal and behavioral signals** can be captured via representation learning  

---

### 1.4. Evaluation metrics

| Metric        | Definition                          | Target |
|--------------|------------------------------------|--------|
| ROC-AUC      | Area under ROC curve               | ≥ 0.95 |
| Precision    | TP / (TP + FP)                     | high   |
| Recall       | TP / (TP + FN)                     | high   |
| F1-score     | harmonic mean                      | ≥ 0.85 |
| PR-AUC       | Precision-recall curve             | key metric |

---

## 2. Data

### 2.1. Dataset

We use the **Credit Card Fraud Detection dataset** (Kaggle):

- ~284,807 transactions  
- 492 fraud cases (~0.17%)  
- PCA-transformed features: `V1`–`V28`  
- Additional features: `Amount`, `Time`

---

### 2.2. Feature representation

| Feature     | Type        | Description |
|------------|------------|-------------|
| V1–V28     | continuous | anonymized PCA features |
| Amount     | continuous | transaction value |
| Time       | continuous | seconds since first transaction |

Preprocessing:
- Standardization of `Amount` and `Time`
- Handling imbalance via weighting or sampling

---

### 2.3. Data challenges

- Extreme class imbalance  
- Lack of interpretability (PCA features)  
- Static dataset (no real temporal evolution)  

---

## 3. Base Architecture

Baseline model: traditional machine learning

### 3.1. Model

- Logistic Regression  
- Random Forest  
- XGBoost  

---

### 3.2. Limitations

1. No representation learning  
2. Poor generalization to unseen fraud patterns  
3. Sensitive to imbalance  
4. No transfer learning capability  

---

## 4. Proposed Architecture

We propose a **deep tabular model with transfer learning** using:

- :contentReference[oaicite:0]{index=0}  

---

### 4.1. Architecture

| Component        | Specification |
|-----------------|--------------|
| Input           | numerical features |
| Embedding layer | feature-wise embeddings |
| Transformer     | multi-head attention |
| Dense layers    | fully connected |
| Output          | sigmoid (fraud probability) |

---

### 4.2. Why this works

1. **Attention mechanism** captures feature interactions  
2. **Better generalization** vs tree models  
3. **Transferable embeddings** across datasets  
4. Handles tabular data effectively  

---

## 5. Data-Source Considerations

### 5.1. Imbalance handling

- Weighted loss  
- SMOTE (optional)  
- Undersampling  

---

### 5.2. Feature engineering

- Transaction frequency  
- Aggregations per user (if available)  
- Time-based features  

---

### 5.3. Limitations

- Dataset lacks user IDs  
- No sequential behavior modeling  

---

## 6. Transfer Learning Strategy

We leverage pretraining from large tabular datasets.

### 6.1. Source model

Pretrained:
- :contentReference[oaicite:1]{index=1}  

---

### 6.2. Transfer approach

| Layer              | Strategy |
|--------------------|----------|
| Embeddings         | reused |
| Transformer layers | fine-tuned |
| Output layer       | re-trained |

---

### 6.3. Training phases

| Phase | Description |
|------|------------|
| 1    | Freeze backbone, train classifier |
| 2    | Unfreeze transformer |
| 3    | Full fine-tuning |

---

### 6.4. Benefits

- Faster convergence  
- Better minority detection  
- Improved generalization  

---

## 7. Roadmap

- Data preprocessing pipeline  
- Baseline implementation  
- Transformer model training  
- Transfer learning experiments  
- Evaluation and comparison  

---

## 8. References

1. Gorishniy, Y. et al. *Revisiting Deep Learning Models for Tabular Data.*
2. Vaswani, A. et al. *Attention is All You Need.*
3. Chen, T. & Guestrin, C. *XGBoost.*
4. He, H. & Garcia, E. *Learning from Imbalanced Data.*
