import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# =========================
# LOAD DATA
# =========================
def load_data():
    df = pd.read_csv("student_performance_academic_5000.csv")
    df.columns = df.columns.str.strip()
    return df

# =========================
# PREPROCESS DATA
# =========================
def preprocess_data(df):
    possible_targets = ["Exam_Score", "Final_Score", "Score", "Performance"]

    target_column = None
    for col in df.columns:
        if col in possible_targets:
            target_column = col
            break

    if target_column is None:
        target_column = df.columns[-1]

    if df[target_column].dtype != "object":
        threshold = df[target_column].median()
        df["Target"] = (df[target_column] >= threshold).astype(int)
    else:
        df["Target"] = df[target_column].astype("category").cat.codes

    numeric_df = df.select_dtypes(include=[np.number])

    X = numeric_df.drop(columns=["Target"])
    y = numeric_df["Target"]

    return X, y

# =========================
# SPLIT + SCALE
# =========================
def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

# =========================
# TRAIN MODELS
# =========================
def train_models(X_train, y_train):
    models = {
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models

# =========================
# EVALUATION
# =========================
def evaluate(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": auc(fpr, tpr),
            "fpr": fpr,
            "tpr": tpr,
            "y_pred": y_pred
        }

    return results

# =========================
# PLOTS
# =========================
def plot_cm(y_test, y_pred, name):
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{name} Confusion Matrix")
    st.pyplot(fig)

def plot_roc(results):
    fig, ax = plt.subplots()

    for name in results:
        ax.plot(results[name]["fpr"], results[name]["tpr"],
                label=f"{name} (AUC={results[name]['auc']:.2f})")

    ax.plot([0, 1], [0, 1], "--")
    ax.legend()
    ax.set_title("ROC Curve Comparison")

    st.pyplot(fig)

# =========================
# MAIN FUNCTION
# =========================
def main():
    st.set_page_config(layout="wide")
    st.title("🎓 Student Performance Prediction System")

    df = load_data()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    models = train_models(X_train, y_train)

    results = evaluate(models, X_test, y_test)

    # =====================
    # RESULTS
    # =====================
    st.subheader("📊 Model Comparison")

    best_model = max(results, key=lambda x: results[x]["accuracy"])
    st.success(f"🏆 Best Model: {best_model}")

    for name in results:
        st.write(f"### {name}")
        st.write(results[name])

        plot_cm(y_test, results[name]["y_pred"], name)

    plot_roc(results)

    # =====================
    # INPUT SECTION
    # =====================
    st.subheader("🔧 Prediction")

    input_data = []

    for col in X.columns:
        val = st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))
        input_data.append(val)

    model_choice = st.selectbox("Select Model", list(models.keys()))

    sample = scaler.transform([input_data])
    pred = models[model_choice].predict(sample)[0]

    prob = models[model_choice].predict_proba(sample)[0][1]

    st.metric("Prediction", int(pred))
    st.progress(float(prob))

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    main()
