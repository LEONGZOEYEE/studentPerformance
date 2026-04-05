import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# =========================
# GRADE FUNCTION
# =========================
def get_grade(score):
    if score >= 90: return "A+"
    elif score >= 80: return "A"
    elif score >= 75: return "A-"
    elif score >= 70: return "B+"
    elif score >= 65: return "B"
    elif score >= 60: return "B-"
    elif score >= 55: return "C+"
    elif score >= 50: return "C"
    else: return "F"


# =========================
# GRADE MAPPING (FIXED ORDER)
# =========================
grade_map = {
    "F": 0,
    "C": 1,
    "C+": 2,
    "B-": 3,
    "B": 4,
    "B+": 5,
    "A-": 6,
    "A": 7,
    "A+": 8
}

reverse_grade = {v: k for k, v in grade_map.items()}


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)

    data["Grade"] = data["Exam_Score"].apply(get_grade)
    data["Grade"] = data["Grade"].map(grade_map)

    X = data[['Attendance', 'Hours_Studied', 'Previous_Scores']]
    y = data['Grade']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, data


# =========================
# TRAIN MODELS
# =========================
@st.cache_resource
def train_models(X_train, y_train):
    models = {}

    models["SVM"] = SVC(
        probability=True,
        class_weight="balanced",
        C=10,
        gamma="scale",
        random_state=42
    ).fit(X_train, y_train)

    models["KNN"] = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="manhattan"
    ).fit(X_train, y_train)

    models["ANN"] = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=800,
        alpha=0.001,
        early_stopping=True,
        random_state=42
    ).fit(X_train, y_train)

    return models


# =========================
# EVALUATION
# =========================
def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "y_pred": y_pred
        }

    return results


# =========================
# CONFUSION MATRIX
# =========================
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)

    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    labels = [reverse_grade[i] for i in unique_labels]

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=labels,
                yticklabels=labels)

    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Grade")
    plt.ylabel("Actual Grade")

    st.pyplot(plt.gcf())
    plt.clf()


# =========================
# MAIN APP
# =========================
def main():
    st.set_page_config(layout="wide")

    st.title("🎓 Student Grade Prediction System")

    file_path = "StudentPerformanceFactors.csv"

    X_train, X_test, y_train, y_test, scaler, raw_data = load_data(file_path)

    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    # =========================
    # BEST MODEL (FIXED → F1)
    # =========================
    st.subheader("📊 Model Comparison")

    best_model = max(results, key=lambda x: results[x]["f1"])
    st.success(f"🏆 Best Model (based on F1 Score): {best_model}")

    st.info("Dataset is imbalanced (more B/B+ students), so models may favor mid-range grades.")

    # =========================
    # GRADE DISTRIBUTION (ONCE ONLY)
    # =========================
    st.subheader("📊 Grade Distribution")

    grade_counts = raw_data["Grade"].value_counts().sort_index()
    grade_labels = [reverse_grade[i] for i in grade_counts.index]

    st.bar_chart(pd.Series(grade_counts.values, index=grade_labels))

    # =========================
    # MODEL TABS
    # =========================
    tabs = st.tabs(["SVM", "KNN", "ANN"])

    for i, name in enumerate(["SVM", "KNN", "ANN"]):
        with tabs[i]:
            res = results[name]

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Accuracy", f"{res['accuracy']*100:.2f}%")
            col2.metric("Precision", f"{res['precision']:.2f}")
            col3.metric("Recall", f"{res['recall']:.2f}")
            col4.metric("F1 Score", f"{res['f1']:.2f}")

            st.subheader("Confusion Matrix")
            plot_confusion_matrix(y_test, res["y_pred"], name)

    # =========================
    # PREDICTION
    # =========================
    st.subheader("🔧 Predict Student Grade")

    with st.form("form"):
        attendance = st.slider("Attendance (%)", 0, 100, 80)
        study = st.slider("Study Hours", 0, 30, 10)
        prev = st.slider("Previous Score", 0, 100, 70)

        model_choice = st.selectbox("Model", ["SVM", "KNN", "ANN"])
        submit = st.form_submit_button("Predict")

    if submit:
        sample = scaler.transform([[attendance, study, prev]])
        model = models[model_choice]

        pred = model.predict(sample)[0]
        grade = reverse_grade[pred]

        if hasattr(model, "predict_proba"):
            prob = np.max(model.predict_proba(sample))
        else:
            prob = 1.0

        st.success(f"🎓 Predicted Grade: {grade}")
        st.write(f"Confidence: {prob*100:.2f}%")
        st.progress(float(prob))


if __name__ == "__main__":
    main()
