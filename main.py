import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
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


# 🔥 CORRECT ORDER (IMPORTANT)
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

    # Create grade
    data["Grade"] = data["Exam_Score"].apply(get_grade)

    # 🔥 FIX: use manual mapping (NOT LabelEncoder)
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

    # 🔥 FIX: class_weight balanced
    models["SVM"] = SVC(
        probability=True,
        class_weight="balanced",
        C=10,
        gamma="scale",
        random_state=42
    ).fit(X_train, y_train)

    models["KNN"] = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    ).fit(X_train, y_train)

    models["ANN"] = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
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

        auc = None
        if hasattr(model, "predict_proba"):
            try:
                auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")
            except:
                pass

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "auc": auc,
            "y_pred": y_pred
        }

    return results


# =========================
# CONFUSION MATRIX
# =========================
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d")

    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

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
    # MODEL COMPARISON
    # =========================
    st.subheader("📊 Model Comparison")

    best_model = max(results, key=lambda x: results[x]["accuracy"])
    st.success(f"🏆 Best Model: {best_model}")

    tabs = st.tabs(["SVM", "KNN", "ANN"])

    for i, name in enumerate(["SVM", "KNN", "ANN"]):
        with tabs[i]:
            res = results[name]

            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric("Accuracy", f"{res['accuracy']*100:.2f}%")
            col2.metric("Precision", f"{res['precision']:.2f}")
            col3.metric("Recall", f"{res['recall']:.2f}")
            col4.metric("F1 Score", f"{res['f1']:.2f}")
            col5.metric("AUC", f"{res['auc']:.2f}" if res["auc"] else "N/A")

            st.subheader("Confusion Matrix")
            plot_confusion_matrix(y_test, res["y_pred"], name)

    # =========================
    # PREDICTION
    # =========================
    st.subheader("🔧 Predict Student Grade")

    with st.form("form"):
        attendance = st.slider("Attendance", 0, 100, 80)
        study = st.slider("Study Hours", 0, 30, 10)
        prev = st.slider("Previous Score", 0, 100, 70)

        model_choice = st.selectbox("Model", ["SVM", "KNN", "ANN"])
        submit = st.form_submit_button("Predict")

    if submit:
        sample = scaler.transform([[attendance, study, prev]])
        model = models[model_choice]

        pred = model.predict(sample)[0]
        grade = reverse_grade[pred]

        prob = np.max(model.predict_proba(sample))

        st.success(f"🎓 Predicted Grade: {grade}")
        st.progress(float(prob))


if __name__ == "__main__":
    main()
