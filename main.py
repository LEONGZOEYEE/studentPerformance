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
from sklearn.linear_model import LinearRegression

# =========================
# GRADE FUNCTION
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
    data = pd.read_csv(file_path).dropna()

    X = data[['Attendance', 'Hours_Studied', 'Previous_Scores']]
    y_class = (data['Exam_Score'] >= 70).astype(int)
    y_score = data['Exam_Score']

    X_train, X_test, y_train_class, y_test_class, y_train_score, y_test_score = train_test_split(
        X, y_class, y_score, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_class, y_test_class, y_train_score, scaler, data

# =========================
# TRAIN MODELS
# =========================
@st.cache_resource
def train_models(X_train, y_train_class):
    models = {
        "SVM": SVC(probability=True, random_state=42).fit(X_train, y_train_class),
        "KNN": KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train_class),
        "ANN": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, early_stopping=True, random_state=42).fit(X_train, y_train_class)
    }
    return models

# =========================
# TRAIN SCORE PREDICTION
# =========================
@st.cache_resource
def train_score_model(X_train, y_train_score):
    return LinearRegression().fit(X_train, y_train_score)

# =========================
# EVALUATE
# =========================
def evaluate_models(models, X_test, y_test_class):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test_class, y_prob)

        results[name] = {
            "accuracy": accuracy_score(y_test_class, y_pred),
            "precision": precision_score(y_test_class, y_pred, zero_division=0),
            "recall": recall_score(y_test_class, y_pred, zero_division=0),
            "f1": f1_score(y_test_class, y_pred, zero_division=0),
            "auc": auc(fpr, tpr),
            "y_pred": y_pred
        }
    return results

# =========================
# PLOTS
# =========================
def plot_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["<70", "≥70"], yticklabels=["<70", "≥70"])
    plt.title(f"{name}")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_roc(results):
    plt.figure(figsize=(6,4))
    for n in results:
        plt.plot(results[n]["fpr"], results[n]["tpr"], label=f"{n} (AUC={results[n]['auc']:.2f})")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

# =========================
# MAIN
# =========================
def main():
    st.set_page_config(layout="wide")
    st.title("🎓 Student Performance Prediction")

    file_path = "StudentPerformanceFactors.csv"
    X_train, X_test, y_train_class, y_test_class, y_train_score, scaler, raw_data = load_data(file_path)
    models = train_models(X_train, y_train_class)
    score_model = train_score_model(X_train, y_train_score)
    results = evaluate_models(models, X_test, y_test_class)

    # --------------------------
    # MODEL DISPLAY
    # --------------------------
    st.subheader("📊 Model Results")
    best = max(results, key=lambda x: results[x]["accuracy"])
    st.success(f"Best Model: {best}")

    for name in ["SVM", "KNN", "ANN"]:
        res = results[name]
        st.write(f"### {name}")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{res['accuracy']:.2f}")
        col2.metric("Precision", f"{res['precision']:.2f}")
        col3.metric("Recall", f"{res['recall']:.2f}")
        col4.metric("F1", f"{res['f1']:.2f}")
        col5.metric("AUC", f"{res['auc']:.2f}")
        plot_confusion_matrix(y_test_class, res["y_pred"], name)

    plot_roc(results)

    # --------------------------
    # INPUT
    # --------------------------
    st.subheader("🔧 Input Student Data")
    attendance = st.slider("Attendance", 0, 100, 75)
    study = st.slider("Hours Studied", 0, 30, 10)
    prev = st.slider("Previous Score", 0, 100, 60)
    model_choice = st.selectbox("Select Model", ["SVM", "KNN", "ANN"])

    # --------------------------
    # PREDICTION
    # --------------------------
    if st.button("🚀 Predict"):
        input_data = [[attendance, study, prev]]
        input_scaled = scaler.transform(input_data)

        # Real score prediction
        pred_score = score_model.predict(input_scaled)[0]
        pred_score = np.clip(pred_score, 0, 100)
        grade = get_grade(pred_score)

        # High score probability
        prob = models[model_choice].predict_proba(input_scaled)[0][1]

        # Display
        st.metric("Predicted Score", f"{pred_score:.2f}")
        st.metric("Grade", grade)
        st.metric("High Score Chance", f"{prob*100:.1f}%")
        st.progress(prob)

        # Feedback
        if pred_score >= 70:
            st.success(f"Good Performance - Grade: {grade} 🎉")
        elif pred_score >= 50:
            st.warning(f"Average - Grade: {grade}")
        else:
            st.error(f"Need Improvement - Grade: {grade}")

if __name__ == "__main__":
    main()
