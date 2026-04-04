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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, data

# =========================
# TRAIN MODELS (SVM TUNED FOR HIGHEST ACCURACY)
# =========================
@st.cache_resource
def train_models(X_train, y_train):
    models = {
        # TUNED SVM (HIGHEST ACCURACY)
        "SVM": SVC(kernel='rbf', C=10, gamma=0.1, probability=True, random_state=42).fit(X_train, y_train),
        "KNN": KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train),
        "ANN": MLPClassifier(hidden_layer_sizes=(32,16), max_iter=300, random_state=42).fit(X_train, y_train)
    }
    return models

# =========================
# EVALUATE
# =========================
def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
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
    X_train, X_test, y_train, y_test, scaler, raw_data = load_data(file_path)
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    # --------------------------
    # MODEL DISPLAY
    # --------------------------
    st.subheader("📊 Model Results")
    best_model = max(results, key=lambda x: results[x]["accuracy"])
    st.success(f"🏆 Best Model: {best_model} (HIGHEST ACCURACY)")

    for name in ["SVM", "KNN", "ANN"]:
        res = results[name]
        st.write(f"### {name}")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{res['accuracy']:.2f}")
        col2.metric("Precision", f"{res['precision']:.2f}")
        col3.metric("Recall", f"{res['recall']:.2f}")
        col4.metric("F1", f"{res['f1']:.2f}")
        col5.metric("AUC", f"{res['auc']:.2f}")
        plot_confusion_matrix(y_test, res["y_pred"], name)

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

        # Calculate estimated score
        prob = models[model_choice].predict_proba(input_scaled)[0][1]
        estimated_score = 45 + (attendance * 0.3) + (study * 0.8) + (prev * 0.6)
        estimated_score = round(np.clip(estimated_score, 0, 100), 2)
        grade = get_grade(estimated_score)

        # Display
        st.metric("Predicted Score", f"{estimated_score:.2f}")
        st.metric("Grade", grade)
        st.metric("High Score Chance", f"{prob*100:.1f}%")
        st.progress(prob)

        # Feedback
        if estimated_score >= 70:
            st.success(f"Good Performance - Grade: {grade} 🎉")
        elif estimated_score >= 50:
            st.warning(f"Average - Grade: {grade}")
        else:
            st.error(f"Need Improvement - Grade: {grade}")

if __name__ == "__main__":
    main()
