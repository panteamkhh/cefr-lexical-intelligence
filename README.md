
---

# 📚 CEFR Lexical Intelligence

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![NLP](https://img.shields.io/badge/NLP-Semantic%20Modeling-green.svg)
![ML](https://img.shields.io/badge/ML-Clustering-orange.svg)
![Status](https://img.shields.io/badge/Project-Active-brightgreen.svg)

---

## 🧠 Overview

**CEFR Lexical Intelligence** is a structured end-to-end NLP system designed to analyze vocabulary through semantic representations and CEFR proficiency levels.

The system transforms raw lexical data into an intelligent pipeline capable of:

* Understanding vocabulary complexity (CEFR levels)
* Discovering semantic relationships between phrases
* Clustering lexical items in embedding space
* Enabling similarity-based search
* Automatically discovering hidden lexical categories

---

## 📁 Project Structure

```text
cefr-lexical-intelligence/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── clean_dataset.xlsx
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_classical_ml.ipynb
│   ├── 04_embeddings.ipynb
│   ├── 05_clustering.ipynb
│   ├── 06_cefr_analysis.ipynb
│   ├── 07_similarity_search.ipynb
│   └── 08_category_discovery.ipynb
│
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── ml_models.py
│   ├── embeddings.py
│   ├── clustering.py
│   ├── similarity.py
│   └── utils.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ System Pipeline

The project is organized into 8 main stages:

---

## 🧼 Stage 1 — Data Cleaning

* Standardization of dataset structure
* Removal of inconsistencies and noise
* Validation of data integrity
* Preparation of analysis-ready dataset

---

## 📊 Stage 2 — Exploratory Data Analysis (EDA)

* CEFR level distribution analysis
* Category frequency exploration
* Phrase length statistics
* Dataset imbalance detection

---

## 🤖 Stage 3 — Classical Machine Learning

* Feature extraction from lexical data
* Baseline CEFR classification models
* Performance evaluation of traditional ML methods

---

## 🧠 Stage 4 — Embeddings

* Conversion of phrases into dense vector space
* Semantic representation of vocabulary items
* Preparation for clustering and similarity search

---

## 🔗 Stage 5 — Unsupervised Clustering

* Clustering on embedding space
* Discovery of hidden semantic structures
* Analysis of lexical grouping patterns

---

## 🎯 Stage 6 — CEFR-Level Analysis

* Comparison between clusters and CEFR labels
* Evaluation of semantic vs linguistic alignment
* Detection of labeling inconsistencies

---

## 🔍 Stage 7 — Semantic Similarity Search

* Vector-based similarity engine
* Retrieval of semantically related phrases
* Foundation for intelligent search systems

---

## 🧩 Stage 8 — Category Discovery

* Discovery of latent semantic categories
* Unsupervised grouping beyond predefined labels
* Extraction of hidden linguistic structure
* Data-driven category formation

---

## 🛠 Tech Stack

* Python 🐍
* Pandas / NumPy
* Scikit-learn
* Matplotlib / Seaborn
* NLP Embeddings (Sentence Representations)
* Unsupervised Learning Algorithms

---

## 📈 Key Insights

* Vocabulary naturally forms semantic clusters in embedding space
* CEFR levels only partially reflect semantic similarity
* Subcategories often overlap across proficiency levels
* Data-driven categories reveal hidden linguistic structure

---

## 🎯 Goals

* Build a semantic intelligence system for vocabulary analysis
* Bridge linguistic structure with machine learning
* Move from rule-based labels → data-driven discovery
* Enable similarity-based lexical exploration

---

## 🚀 Future Work

* Transformer-based embeddings (BERT / Sentence-BERT)
* Deep learning CEFR classification models
* Interactive semantic search interface (Streamlit/Web App)
* API deployment for lexical intelligence
* Visualization dashboard for linguistic structure

---

## 👨‍💻 Author

A structured NLP research project focused on lexical intelligence, semantic modeling, and CEFR-based language understanding.

---

## ⭐ Support

If this project helps your learning or research, consider starring the repository.
