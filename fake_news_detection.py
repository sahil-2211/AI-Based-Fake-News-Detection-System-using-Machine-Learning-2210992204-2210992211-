# =========================================================
# AI BASED FAKE NEWS DETECTION SYSTEM
# =========================================================

# ===================== IMPORT LIBRARIES ===================
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

import pickle


# ===================== LOAD DATASET =======================
def load_dataset(filepath):
    df = pd.read_csv(filepath)
    print("\nDataset Shape:", df.shape)
    print("\nClass Distribution:\n", df['label'].value_counts())
    return df


# ===================== PREPROCESS TEXT ====================
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()

    stop_words = set([
        'the','is','in','and','to','of','for','on','with','as','at','by',
        'an','be','this','that','it','from','are'
    ])

    words = [w for w in words if w not in stop_words and len(w) > 2]

    return " ".join(words)


# ===================== PREPROCESS DATASET =================
def preprocess_dataset(df):
    if 'title' in df.columns and 'text' in df.columns:
        df['content'] = df['title'] + " " + df['text']
    else:
        df['content'] = df['text']

    df['processed_text'] = df['content'].apply(preprocess_text)

    df['label'] = df['label'].map({'REAL':1, 'FAKE':0})

    return df


# ===================== FEATURE EXTRACTION =================
def extract_features(X_train, X_test):
    tfidf = TfidfVectorizer(max_features=5000)

    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)

    print("\nTF-IDF Features Created")

    return X_train, X_test, tfidf


# ===================== TRAIN MODELS =======================
def train_models(X_train, y_train):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    for name in models:
        print(f"\nTraining {name}...")
        models[name].fit(X_train, y_train)

    return models


# ===================== EVALUATION =========================
def evaluate_models(models, X_test, y_test):

    results = {}

    for name, model in models.items():

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\n===================================")
        print(name)
        print("===================================")
        print("Accuracy :", acc)
        print("Precision:", prec)
        print("Recall   :", rec)
        print("F1 Score :", f1)

        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        results[name] = acc

    return results


# ===================== PLOT ACCURACY ======================
def plot_accuracy(results):

    names = list(results.keys())
    values = list(results.values())

    plt.figure()
    plt.bar(names, values)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.show()


# ===================== ROC CURVE ==========================
def plot_roc(models, X_test, y_test):

    plt.figure()

    for name, model in models.items():

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


# ===================== SAVE MODEL =========================
def save_model(model, tfidf):

    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(tfidf, open("tfidf.pkl", "wb"))

    print("\nModel Saved Successfully!")


# ===================== MAIN FUNCTION ======================
def main():

    print("\n===== FAKE NEWS DETECTION SYSTEM =====")

    # Load data
    df = load_dataset("fake_news_dataset.csv")

    # Preprocess
    df = preprocess_dataset(df)

    X = df['processed_text']
    y = df['label']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # TF-IDF
    X_train, X_test, tfidf = extract_features(X_train, X_test)

    # Train
    models = train_models(X_train, y_train)

    # Evaluate
    results = evaluate_models(models, X_test, y_test)

    # Plot graphs
    plot_accuracy(results)
    plot_roc(models, X_test, y_test)

    # Best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    print("\nBest Model:", best_model_name)

    # Save
    save_model(best_model, tfidf)


# ===================== RUN ===============================
if __name__ == "__main__":
    main()