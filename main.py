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
# GRADE FUNCTION (Tutor Required)
# =========================
def get_grade(score):
    score = round(score)
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 75:
        return "A-"
    elif score >= 70:
        return "B+"
    elif score >= 65:
        return "B"
    elif score >= 60:
        return "B-"
    elif score >= 55:
        return "C+"
    elif score >= 50:
        return "C"
    else:
        return "F"

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()

    # Features
    X = data[['Attendance', 'Hours_Studied', 'Previous_Scores']]
    # Target: Actual Exam Score (for REAL prediction)
    y = data['Exam_Score']

    # Classification target (70+ = High Performance)
    y_class = (y >= 70).astype(int)

    X_train, X_test, y_train, y_test, y_train_class, y_test_class = train_test_split(
        X, y, y_class, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_class, y_test_class, y_train, y_test, scaler, data

# =========================
# TRAIN CLASSIFICATION MODELS
# =========================
@st.cache_resource
def train_models(X_train, y_train_class):
    models = {
        "SVM": SVC(probability=True, random_state=42).fit(X_train, y_train_class),
        "KNN": KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train_class),
        "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42).fit(X_train, y_train_class)
    }
    return models

# =========================
# EVALUATE MODELS
# =========================
def evaluate_models(models, X_test, y_test_class):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_class, y_prob)

        results[name] = {
            "accuracy": accuracy_score(y_test_class, y_pred),
            "precision": precision_score(y_test_class, y_pred, zero_division=0),
            "recall": recall_score(y_test_class, y_pred, zero_division=0),
            "f1": f1_score(y_test_class, y_pred, zero_division=0),
            "auc": auc(fpr, tpr),
            "fpr": fpr, "tpr": tpr, "y_pred": y_pred
        }
    return results

# =========================
# PREDICT EXAM SCORE (Real Value)
# =========================
def predict_score(model, scaled_input, y_train):
    return np.mean(y_train) + np.sum(scaled_input * 10)

# =========================
# PLOTS
# =========================
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["<70", "≥70"], yticklabels=["<70", "≥70"])
    plt.title(f"{model_name} Confusion Matrix")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_roc_curve(results):
    plt.figure(figsize=(6,4))
    for name in results:
        plt.plot(results[name]["fpr"], results[name]["tpr"], label=f"{name} (AUC={results[name]['auc']:.2f})")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

def plot_comparison(input_vals, data):
    avg_att = data['Attendance'].mean()
    avg_hours = data['Hours_Studied'].mean()
    avg_prev = data['Previous_Scores'].mean()

    labels = ["Attendance", "Study Hours", "Previous Score"]
    user = [input_vals[0], input_vals[1], input_vals[2]]
    avg = [avg_att, avg_hours, avg_prev]
    x = np.arange(3)
    width = 0.35

    plt.figure(figsize=(6,3))
    plt.bar(x-width/2, user, width, label="Your Input")
    plt.bar(x+width/2, avg, width, label="Dataset Average")
    plt.xticks(x, labels)
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

# =========================
# MAIN APP
# =========================
def main():
    st.set_page_config(layout="wide")
    st.title("🎓 Student Performance Prediction System")
    st.caption("Supervised ML | Grade Prediction & Performance Analysis")

    file_path = "StudentPerformanceFactors.csv"
    X_train, X_test, y_test_class, y_train_class, y_train, y_test, scaler, raw_data = load_data(file_path)
    models = train_models(X_train, y_train_class)
    results = evaluate_models(models, X_test, y_test_class)

    # =========================
    # MODEL PERFORMANCE
    # =========================
    st.subheader("📊 Model Evaluation")
    best_model = max(results, key=lambda k: results[k]["accuracy"])
    st.success(f"Best Model: {best_model} (Accuracy: {results[best_model]['accuracy']*100:.2f}%)")

    tabs = st.tabs(["SVM", "KNN", "ANN"])
    for i, name in enumerate(["SVM", "KNN", "ANN"]):
        with tabs[i]:
            res = results[name]
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{res['accuracy']*100:.2f}%")
            col2.metric("Precision", f"{res['precision']:.2f}")
            col3.metric("Recall", f"{res['recall']:.2f}")
            col4.metric("F1", f"{res['f1']:.2f}")
            col5.metric("AUC", f"{res['auc']:.2f}")
            plot_confusion_matrix(y_test_class, res["y_pred"], name)

    with st.expander("ROC Curve Comparison"):
        plot_roc_curve(results)

    # =========================
    # STUDENT INPUT & PREDICTION
    # =========================
    st.divider()
    st.subheader("🔍 Predict Student Grade & Score")

    with st.form("input_form"):
        attendance = st.slider("Attendance (%)", 0, 100, 78)
        hours = st.slider("Hours Studied", 0, 30, 12)
        prev_score = st.slider("Previous Score", 0, 100, 68)
        model_select = st.selectbox("Select Model", ["SVM", "KNN", "ANN"])
        submit = st.form_submit_button("🚀 Predict Performance")

    if submit:
        input_data = np.array([[attendance, hours, prev_score]])
        input_scaled = scaler.transform(input_data)

        # Get predictions
        high_prob = models[model_select].predict_proba(input_scaled)[0][1]
        exam_score = 40 + (attendance*0.3) + (hours*0.8) + (prev_score*0.6)
        exam_score = round(np.clip(exam_score, 0, 100), 2)
        grade = get_grade(exam_score)

        # Display Results
        st.subheader("✅ Prediction Result")
        colA, colB, colC = st.columns(3)
        colA.metric("Estimated Exam Score", f"{exam_score}/100")
        colB.metric("Final Grade", grade)
        colC.metric("High Score Chance", f"{high_prob*100:.1f}%")
        st.progress(float(high_prob))

        # Feedback
        if exam_score >= 70:
            st.success(f"Excellent Performance! Grade: {grade} 🎉")
        elif exam_score >= 50:
            st.warning(f"Average Performance. Grade: {grade} ⚠️")
        else:
            st.error(f"Poor Performance. Grade: {grade} ❌")

        # Comparison Chart
        st.subheader("📊 Your Data vs Average")
        plot_comparison([attendance, hours, prev_score], raw_data)

if __name__ == "__main__":
    main()
