import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)

    # =========================
    # CREATE GRADES
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

    data["Grade"] = data["Exam_Score"].apply(get_grade)

    le = LabelEncoder()
    data["Grade"] = le.fit_transform(data["Grade"])

    data = data[['Attendance', 'Hours_Studied', 'Previous_Scores', 'Grade']]

    X = data.drop('Grade', axis=1)
    y = data['Grade']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, data, le


# =========================
# TRAIN MODELS
# =========================
@st.cache_resource
def train_models(X_train, y_train):
    models = {}

    models["SVM"] = SVC(probability=True, random_state=42).fit(X_train, y_train)

    models["KNN"] = KNeighborsClassifier(
        n_neighbors=7,
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

        # AUC SAFE (multiclass)
        auc = None
        if hasattr(model, "predict_proba"):
            try:
                auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")
            except:
                auc = None

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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

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
    st.caption("Predict final academic grade using ML models")

    file_path = "StudentPerformanceFactors.csv"

    X_train, X_test, y_train, y_test, scaler, raw_data, le = load_data(file_path)

    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    # =========================
    # MODEL COMPARISON
    # =========================
    st.subheader("📊 Model Comparison")
    st.divider()

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
            
            # AUC (safe check)
            if res.get("auc") is not None:
                col5.metric("AUC (OvR)", f"{res['auc']:.2f}")
            else:
                col5.metric("AUC (OvR)", "N/A")

    # =========================
    # INPUT
    # =========================
    st.subheader("🔧 Predict Student Grade")

    with st.form("prediction_form"):
        attendance = st.slider("Attendance (%)", 0, 100, 75)
        study = st.slider("Study Hours", 0, 30, 10)
        prev = st.slider("Previous Score", 0, 100, 60)

        model_choice = st.selectbox("Choose Model", ["SVM", "KNN", "ANN"])
        submit = st.form_submit_button("🚀 Predict")

    if submit:
        sample = scaler.transform([[attendance, study, prev]])
        model = models[model_choice]

        pred_class = model.predict(sample)[0]
        grade = le.inverse_transform([pred_class])[0]

        prob = np.max(model.predict_proba(sample)) if hasattr(model, "predict_proba") else 1.0

        st.success(f"🎓 Predicted Grade: {grade}")
        st.progress(float(prob))

        # =========================
        # FIXED EXPLANATION (IMPORTANT)
        # =========================
        st.subheader("🧠 Performance Insight")

        high_students = raw_data[raw_data["Grade"] >= raw_data["Grade"].quantile(0.75)]

        avg = high_students[["Attendance", "Hours_Studied", "Previous_Scores"]].mean()

        input_vals = {
            "Attendance": attendance,
            "Hours_Studied": study,
            "Previous_Scores": prev
        }

        for k in input_vals:
            if input_vals[k] >= avg[k]:
                st.write(f"✔ {k}: above strong students average")
            else:
                st.write(f"⚠ {k}: below strong students average")

        # =========================
        # DOWNLOAD
        # =========================
        df = pd.DataFrame({
            "Attendance": [attendance],
            "Study Hours": [study],
            "Previous Score": [prev],
            "Model": [model_choice],
            "Predicted Grade": [grade]
        })

        st.download_button("📥 Download Result", df.to_csv(index=False), "grade_prediction.csv")


if __name__ == "__main__":
    main()
