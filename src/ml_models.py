import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# =========================
# 1. Load Dataset
# =========================
file_path = r"D:\list of words\cefr-project\data\processed\clean_dataset.xlsx"
df = pd.read_excel(file_path)

# Normalize column names
df.columns = df.columns.str.lower().str.strip()

print("Dataset shape:", df.shape)
print(df.head())


# =========================
# 2. Features & Labels
# =========================
texts = df["phrase"].astype(str)
labels = df["level"]


# =========================
# 3. TF-IDF Vectorization
# =========================
tfidf = TfidfVectorizer(
    max_features=8000,   # limit vocabulary size
    ngram_range=(1, 2),  # unigrams + bigrams
    min_df=2             # ignore very rare terms
)

X = tfidf.fit_transform(texts)
print("TF-IDF shape:", X.shape)


# =========================
# 4. Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels  # keep class distribution balanced
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# =========================
# 5. Define Models
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}


# =========================
# 6. Train & Evaluate
# =========================
results = {}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    
    # Store results
    results[name] = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "predictions": y_pred,
        "model": model
    }
    
    # Print report
    print(classification_report(y_test, y_pred))


# =========================
# 7. Compare Models
# =========================
df_results = pd.DataFrame([
    {
        "model": name,
        "accuracy": res["accuracy"],
        "f1_macro": res["f1_macro"],
        "f1_weighted": res["f1_weighted"]
    }
    for name, res in results.items()
])

print("\n=== Model Comparison ===")
print(df_results.sort_values(by="f1_weighted", ascending=False))


# =========================
# 8. Confusion Matrix (SVM)
# =========================
best_model_name = "Linear SVM"
best_model = results[best_model_name]["model"]
y_pred_svm = results[best_model_name]["predictions"]

cm = confusion_matrix(y_test, y_pred_svm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=best_model.classes_
)

disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix - Linear SVM")
plt.show()


# =========================
# 9. Confusion Matrix (Random Forest)
# =========================
rf_model = results["Random Forest"]["model"]
y_pred_rf = results["Random Forest"]["predictions"]

cm = confusion_matrix(y_test, y_pred_rf)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=rf_model.classes_
)

disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix - Random Forest")
plt.show()