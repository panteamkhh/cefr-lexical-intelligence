<div align="center">

# 📚 CEFR Lexical Intelligence

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![NLP](https://img.shields.io/badge/NLP-Semantic%20Modeling-green.svg)
![ML](https://img.shields.io/badge/ML-Clustering-orange.svg)
![Status](https://img.shields.io/badge/Project-Active-brightgreen.svg)

</div>

<br>

## 🧠 Overview

CEFR Lexical Intelligence is an end-to-end NLP system for analyzing vocabulary through **semantic embeddings and CEFR proficiency levels**.

The project covers a full pipeline from raw data to semantic intelligence:

- Data cleaning
- Exploratory Data Analysis (EDA)
- Classical Machine Learning
- Embedding-based semantic modeling
- Unsupervised learning
- CEFR semantic comparison
- Semantic search
- Category discovery

<br>

## 📊 Dataset

| Attribute | Details |
|---|---|
| **Size** | ~14,000+ vocabulary and phrase entries |
| **CEFR Levels** | A1–C2 |
| **Categories** | Main semantic categories |
| **Sub-categories** | Included |

<br>

## 🏗 Project Pipeline

<br>

### 🟢 Phase 1 — Clean Data 

- Load raw dataset (Excel format)
- Handle missing values
- Remove duplicates
- Normalize text (lowercase + strip)
- Standardize column names
- **Output:** `clean_dataset.csv`

<br>

### 🟡 Phase 2 — EDA 

- CEFR level distribution
- Category / sub-category distribution
- Phrase length analysis
- Cross-tab analysis

> **Key Insight:**
> Dataset is imbalanced and semantic embeddings are more effective than pure classical ML.

<br>

### 🟠 Phase 3 — Classical ML

**Feature extraction:** TF-IDF

**Models:**
- Logistic Regression
- SVM
- Random Forest

**Evaluation:**
- Accuracy
- F1-score
- Recall

<br>

### 🔵 Phase 4 — Embeddings / Similarity

- **Model:** `all-MiniLM-L6-v2`
- 384-dimensional sentence embeddings
- Semantic vector space representation
- Cosine similarity for semantic search foundation

**Outputs:**
- `embeddings.npy`
- `metadata.csv`

<br>

### 🟣 Phase 5 — Unsupervised Learning

- KMeans clustering
- DBSCAN clustering
- Hierarchical clustering
- Dimensionality reduction (PCA / UMAP)
- Evaluation using Silhouette Score

<br>

### 🔵 Phase 6 — CEFR vs Embedding Analysis

- Compare CEFR labels with embedding clusters
- Analyze semantic vs linguistic alignment
- Identify mismatch patterns

<br>

### 🟡 Phase 7 — Semantic Search

- Query → embedding conversion
- Cosine similarity ranking
- Top-K retrieval system
- Optional filtering by CEFR/category

<br>

### 🟣 Phase 8 — Category Discovery

- Unsupervised cluster interpretation
- Discovery of hidden semantic groups
- Comparison with existing taxonomy

<br>

## ⚡ Core Principles

- Embeddings are created only once
- Clustering happens only in Phase 5
- EDA is insight, not modeling

<br>

## 📁 Project Structure

```text
data/
├── raw/
├── processed/
├── figures/
├── output/

notebooks/
├── 01_clean_data.ipynb
├── 02_eda.ipynb
├── 03_classical_ml.ipynb
├── 04_embeddings_similarity.ipynb
├── 05_unsupervised_learning.ipynb
├── 06_cefr_vs_embeddings.ipynb
├── 07_semantic_search.ipynb
├── 08_category_discovery.ipynb

src/
├── data_cleaning.py
├── eda.py
├── ml_models.py
├── embeddings.py
├── clustering.py
├── search.py

root/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
```