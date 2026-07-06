<div align="center">

# 📚 CEFR Lexical Intelligence

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![NLP](https://img.shields.io/badge/NLP-Semantic%20Embeddings-green.svg)
![ML](https://img.shields.io/badge/ML-Clustering-orange.svg)
![Status](https://img.shields.io/badge/Project-Active-brightgreen.svg)

</div>

<br>

## 🧠 Overview

CEFR Lexical Intelligence is an end-to-end NLP system for analyzing vocabulary through **semantic embeddings and CEFR proficiency levels**.

The project transforms raw lexical data into a **semantic vector space** and builds multiple intelligence layers on top of it:

- Data cleaning & normalization
- Classical machine learning
- Semantic embedding pipeline
- Unsupervised clustering
- CEFR semantic analysis
- Semantic search system
- Category discovery

<br>

---

## 📊 Dataset

| Attribute | Details |
|---|---|
| Size | ~14,000+ phrases |
| CEFR Levels | A1–C2 |
| Structure | Phrase + Level + Categories |
| Categories | Main + Sub semantic groups |

<br>

---

## 🏗 Project Pipeline

<br>

### 🟢 Phase 1 — Data Cleaning

- Load raw Excel dataset
- Remove duplicates
- Handle missing values
- Normalize text
- Standardize structure

**Output:**
- `clean_dataset.xlsx`

<br>

---

### 🟡 Phase 2 — EDA (Exploratory Data Analysis)

- CEFR distribution analysis
- Category distribution
- Phrase length analysis
- Data imbalance detection

**Insight:**
Semantic modeling is more effective than classical feature engineering for this dataset.

<br>

---

### 🟠 Phase 3 — Classical Machine Learning

**Feature extraction:**
- TF-IDF vectorization

**Models:**
- Logistic Regression
- SVM
- Random Forest

**Goal:**
Baseline comparison against semantic embeddings.

<br>

---

### 🔵 Phase 4 — Embedding Pipeline (Core System)

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- 384-dimensional sentence embeddings
- Batch encoding (batch_size=64)
- Normalized embeddings (cosine-ready space)

This phase converts text into a **semantic vector space representation**.

**Outputs:**

data/embeddings/
 ├── embeddings.npy 
 ├── metadata.csv 
 └── embedding_config.json

 
**Validation:**
- Shape verification: `(14848, 384)`
- NaN / Inf checks
- Vector norm consistency
- Semantic similarity sanity tests

**Key Insight:**
The dataset is now represented as a geometric semantic space, enabling similarity search and clustering.

<br>

---

### 🟣 Phase 5 — Clustering (Unsupervised Learning)

- KMeans clustering
- DBSCAN clustering
- Hierarchical clustering
- Dimensionality reduction (PCA / UMAP)
- Evaluation via silhouette score

**Goal:**
Discover hidden semantic groups in embedding space.

<br>

---

### 🔵 Phase 6 — CEFR Analysis

- Compare CEFR labels with embedding clusters
- Identify mismatches between linguistic level and semantic similarity
- Analyze language proficiency structure in vector space

<br>

---

### 🟡 Phase 7 — Semantic Search System

- Query → embedding conversion
- Cosine similarity ranking
- Top-K retrieval system
- Optional filtering by CEFR / category

This phase transforms embeddings into a **search engine-like system**.

<br>

---

### 🟣 Phase 8 — Category Discovery

- Unsupervised cluster interpretation
- Discovery of hidden semantic categories
- Comparison with original taxonomy

<br>

---

## ⚡ Core Principles

- Embeddings are generated once and reused across all phases
- Phase 4 is the foundation of the entire system
- All downstream tasks depend on embedding space consistency
- Metadata and embeddings are strictly separated
- Reproducibility is ensured via configuration file

<br>

---

## 📁 Project Structure

```text
data/
├── raw/
├── processed/
│   └── clean_dataset.xlsx
├── embeddings/
│   ├── embeddings.npy
│   ├── metadata.csv
│   └── embedding_config.json
├── figures/
└── output/

notebooks/
├── 01_clean_data.ipynb
├── 02_eda.ipynb
├── 03_classical_ml.ipynb
├── 04_embeddings_pipeline.ipynb
├── 05_clustering_analysis.ipynb
├── 06_CEFR analysis
├── 07_similarity search.ipynb
├── 08_category_discovery.ipynb

src/
├── data_cleaning.py
├── eda.py
├── ml_models.py
├── embeddings.py
├── clustering.py
├── search.py
└── utils.py

configs/
└── embedding_config.json

root/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore