import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, roc_curve, auc
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
# LOAD DATA
# =========================
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)

    data['High_Score'] = (data['Exam_Score'] >= 70).astype(int)

    features = ['Attendance', 'Hours_Studied', 'Previous_Scores']
    data = data[features + ['High_Score']]

    X = data[features]
    y = data['High_Score']

    # IMPORTANT FIX: stratify prevents single-class train split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
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
    models = {
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=800,
            early_stopping=True,
            random_state=42
        )
    }

    for name in models:
        models[name].fit(X_train, y_train)

    return models

# =========================
# EVALUATE
# =========================
def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        # FIX: ROC SAFE CHECK
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        # FIX: ROC only if 2 classes exist
        if len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = [0,1], [0,1], 0.0

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr,
            "y_pred": y_pred
        }

    return results

# =========================
# PLOTS
# =========================
def plot_confusion_matrix(y_true, y_pred, model_name):
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low","High"],
                yticklabels=["Low","High"],
                ax=ax)

    ax.set_title(f"{model_name} Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)

def plot_roc_curve(results):
    fig, ax = plt.subplots()

    for name in results:
        ax.plot(results[name]["fpr"], results[name]["tpr"],
                label=f"{name} (AUC={results[name]['auc']:.2f})")

    ax.plot([0,1],[0,1],'--')
    ax.legend()
    ax.set_title("ROC Curve Comparison")

    st.pyplot(fig)
    plt.close(fig)

def plot_input_vs_average(input_vals, raw_data):
    averages = raw_data[['Attendance','Hours_Studied','Previous_Scores']].mean()

    labels = list(input_vals.keys())
    input_data = list(input_vals.values())
    avg_data = [averages[f] for f in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()

    ax.bar(x - width/2, input_data, width, label="Your Input")
    ax.bar(x + width/2, avg_data, width, label="Average")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Input vs Average")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

# =========================
# MAIN
# =========================
def main():
    st.set_page_config(layout="wide")
    st.title("🎓 Student Performance Prediction System")

    file_path = "StudentPerformanceFactors.csv"

    try:
        X_train, X_test, y_train, y_test, scaler, raw_data = load_data(file_path)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return

    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    st.subheader("📊 Model Comparison")

    best_model = max(results, key=lambda x: results[x]["accuracy"])
    st.success(f"🏆 Best Model: {best_model}")

    for name in results:
        res = results[name]
        st.write(f"### {name}")
        st.write(f"Accuracy: {res['accuracy']:.2f}")
        st.write(f"Precision: {res['precision']:.2f}")
        st.write(f"Recall: {res['recall']:.2f}")
        st.write(f"F1 Score: {res['f1']:.2f}")
        st.write(f"AUC: {res['auc']:.2f}")

        plot_confusion_matrix(y_test, res["y_pred"], name)

    plot_roc_curve(results)

    # =========================
    # INPUT
    # =========================
    st.subheader("🔧 Input Student Data")

    attendance = st.slider("Attendance (%)", 0, 100, 75)
    study = st.slider("Study Hours", 0, 30, 10)
    prev = st.slider("Previous Score", 0, 100, 60)

    model_choice = st.selectbox("Select Model", ["SVM", "KNN", "ANN"])

    if st.button("🚀 Predict"):
        sample = scaler.transform(np.array([[attendance, study, prev]]))

        prob = models[model_choice].predict_proba(sample)[0][1]
        estimated_score = prob * 100
        grade = get_grade(estimated_score)

        st.metric("Estimated Score", f"{estimated_score:.2f}")
        st.metric("Predicted Grade", grade)
        st.progress(float(prob))

        if prob > 0.7:
            st.success("High chance of success 🎉")
        elif prob > 0.5:
            st.warning("Moderate performance ⚠️")
        else:
            st.error("Low performance risk ❌")

        input_vals = {
            "Attendance": attendance,
            "Hours_Studied": study,
            "Previous_Scores": prev
        }

        st.subheader("📊 Input vs Average")
        plot_input_vs_average(input_vals, raw_data)

if __name__ == "__main__":
    main()
