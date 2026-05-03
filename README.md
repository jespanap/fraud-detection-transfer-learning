# Fraud Detection with Transfer Learning (IEEE-CIS)

**Universidad EAFIT — Neural Networks and Deep Learning (2026-1)**

**Author:** Jaider España  

**Repository purpose:** project proposal — first deliverable

> A deep learning system for fraud detection in online transactions using
> transfer learning over high-dimensional tabular data. This project leverages
> representation learning to improve detection performance under extreme class
> imbalance and complex feature interactions.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Data](#2-data)
3. [Base Architecture](#3-base-architecture)
4. [Proposed Architecture](#4-proposed-architecture)
5. [Data Challenges & Strategy](#5-data-challenges--strategy)
6. [Transfer Learning Strategy](#6-transfer-learning-strategy)
7. [Roadmap](#7-roadmap)
8. [References](#8-references)

---

## 1. Problem Statement

### 1.1. Domain

Fraud detection in e-commerce transactions involves identifying malicious
activities within large-scale payment systems.

This project is based on the  
:contentReference[oaicite:1]{index=1}

released by:

- :contentReference[oaicite:2]{index=2}  
- :contentReference[oaicite:3]{index=3}  

---

### 1.2. Problem formulation

We define a binary classification problem:

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}
$$

- $x_i \in \mathbb{R}^d$ — transaction features (~400+)  
- $y_i \in \{0,1\}$ — fraud label  

Model:

$$
f_\theta(x) = P(y=1 \mid x)
$$

---

### 1.3. Why this is a hard problem

1. **Extreme feature dimensionality (~400 features)**  
2. **Heterogeneous data** (transaction + identity)  
3. **Severe class imbalance (~3–4% fraud)**  
4. **Hidden patterns (anonymized features)**  
5. **Non-linear dependencies across variables**  

---

### 1.4. Evaluation metrics

| Metric   | Importance |
|----------|-----------|
| ROC-AUC  | global performance |
| PR-AUC   | key for imbalance |
| Recall   | fraud detection sensitivity |
| Precision| false positive control |

---

## 2. Data

### 2.1. Dataset description

The dataset consists of two main tables:

| File                     | Description |
|--------------------------|------------|
| `train_transaction.csv`  | transaction-level data |
| `train_identity.csv`     | user/device information |

---

### 2.2. Feature groups

| Group           | Examples |
|----------------|----------|
| Transaction     | amount, product type |
| Identity        | device, browser |
| Temporal        | transaction time |
| Anonymized      | V1–V339 |

---

### 2.3. Data processing pipeline

1. Merge transaction + identity tables  
2. Handle missing values (~high sparsity)  
3. Encode categorical variables  
4. Normalize numerical features  
5. Feature selection / dimensionality reduction  

---

## 3. Base Architecture

### 3.1. Baseline models

- Logistic Regression  
- Random Forest  
- XGBoost  

---

### 3.2. Limitations

1. Manual feature engineering required  
2. Limited generalization  
3. No transfer learning  
4. Poor handling of high-dimensional interactions  

---

## 4. Proposed Architecture

We propose a deep tabular model based on:

- :contentReference[oaicite:4]{index=4}  

---

### 4.1. Architecture overview

| Component        | Description |
|-----------------|------------|
| Input           | numerical + categorical features |
| Embeddings      | feature-wise embeddings |
| Transformer     | multi-head attention layers |
| Dense layers    | nonlinear feature combination |
| Output          | sigmoid probability |

---

### 4.2. Advantages

1. Learns feature interactions automatically  
2. Scales to high-dimensional data  
3. Produces transferable representations  
4. Handles heterogeneous inputs  

---

## 5. Data Challenges & Strategy

### 5.1. Class imbalance

- Weighted loss  
- Focal loss (optional)  
- Stratified sampling  

---

### 5.2. High dimensionality

- Feature selection  
- Regularization  
- Embedding compression  

---

### 5.3. Missing data

- Imputation  
- Learned embeddings for missing values  

---

## 6. Transfer Learning Strategy

### 6.1. Motivation

Training from scratch on tabular data is inefficient.
We leverage pretrained representations.

---

### 6.2. Transfer approach

| Component         | Strategy |
|------------------|----------|
| Embeddings       | reused |
| Transformer      | fine-tuned |
| Output layer     | re-trained |

---

### 6.3. Training schedule

| Phase | Description |
|------|------------|
| 1    | Freeze backbone |
| 2    | Partial unfreeze |
| 3    | Full fine-tuning |

---

### 6.4. Expected benefits

- Faster convergence  
- Better fraud recall  
- Improved robustness  

---

## 7. Roadmap

- Data preprocessing  
- Baseline training  
- Transformer implementation  
- Transfer learning experiments  
- Evaluation and comparison  

---

## 8. References

1. IEEE-CIS Fraud Detection (Kaggle)  
2. Vaswani et al. *Attention is All You Need*  
3. Gorishniy et al. *FT-Transformer*  
4. Chen & Guestrin *XGBoost*  
