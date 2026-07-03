import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Load cleaned dataset
file_path = "D:\\list of words\\cefr-project\\data\\processed\\clean_dataset.xlsx"
df = pd.read_excel(file_path)

# Normalize column names
df.columns = df.columns.str.lower().str.strip()

print(df.shape)
print(df.head())

# Input text (phrases)
texts = df["phrase"].astype(str)

# Labels (CEFR levels)
labels = df["level"]

# TF-IDF Vectorizer (baseline NLP representation)
tfidf = TfidfVectorizer(
    max_features=8000,   # limit vocabulary size
    ngram_range=(1, 2),  # unigrams + bigrams
    min_df=2             # ignore rare words
)

# Convert text → numeric matrix
X = tfidf.fit_transform(texts)

print("TF-IDF Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels  # keeps CEFR distribution balanced
)

print(X_train.shape, X_test.shape)

# Linear probabilistic classifier
logreg = LogisticRegression(max_iter=1000)

# Train model
logreg.fit(X_train, y_train)

# Predict
y_pred_log = logreg.predict(X_test)

# Evaluation
print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_log))

# Linear SVM works very well with sparse TF-IDF
svm = LinearSVC()

svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

print("=== SVM ===")
print(classification_report(y_test, y_pred_svm))

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))

results = []

models = {
    "LogReg": y_pred_log,
    "SVM": y_pred_svm,
    "RF": y_pred_rf
}

for name, preds in models.items():
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro"),
        "f1_weighted": f1_score(y_test, preds, average="weighted")
    })

df_results = pd.DataFrame(results)

print(df_results.sort_values(by="f1_weighted", ascending=False))


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred_svm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=svm.classes_)

disp.plot(xticks_rotation=45)
plt.title("SVM Confusion Matrix")
plt.show()


cm = confusion_matrix(y_test, y_pred_rf)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=rf.classes_)

disp.plot(xticks_rotation=45)
plt.title("Random Forest Confusion Matrix")
plt.show()