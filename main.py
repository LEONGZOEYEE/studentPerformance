import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# -------------------------
# Load & preprocess
# -------------------------
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)

    data['High_Score'] = data['Exam_Score'].apply(lambda x: 1 if x >= 70 else 0)
    data = data.drop(['Exam_Score'], axis=1)

    categorical_cols = data.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    X = data.drop('High_Score', axis=1)
    y = data['High_Score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X.columns, scaler, data


# -------------------------
# Train models
# -------------------------
@st.cache_resource
def train_models(X_train, y_train):
    models = {}

    models['SVM'] = SVC(probability=True, random_state=42).fit(X_train, y_train)
    models['KNN'] = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    models['ANN'] = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        early_stopping=True,
        random_state=42
    ).fit(X_train, y_train)

    return models


# -------------------------
# Evaluate models
# -------------------------
def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc": auc(fpr, tpr),
            "fpr": fpr,
            "tpr": tpr,
            "y_pred": y_pred
        }

    return results


# -------------------------
# Plots
# -------------------------
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(3, 2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    st.pyplot(plt.gcf())
    plt.clf()


def plot_roc_curve(results):
    plt.figure()
    for model_name in results:
        plt.plot(results[model_name]["fpr"], results[model_name]["tpr"],
                 label=f"{model_name} (AUC={results[model_name]['auc']:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()


def plot_attendance_impact(data):
    temp = data.copy()
    bins = [0, 60, 80, 100]
    labels = ['Low', 'Medium', 'High']

    temp['Attendance_Group'] = pd.cut(temp['Attendance'], bins=bins, labels=labels)
    grouped = temp.groupby('Attendance_Group', observed=True)['High_Score'].mean()

    plt.figure()
    sns.barplot(x=grouped.index, y=grouped.values)
    plt.title("Attendance Impact")
    st.pyplot(plt.gcf())
    plt.clf()


# -------------------------
# Main App
# -------------------------
def main():
    st.set_page_config(layout="wide")
    st.title("🎓 Student Performance Prediction System")

    file_path = "StudentPerformanceFactors.csv"

    X_train, X_test, y_train, y_test, feature_names, scaler, raw_data = load_data(file_path)
    models = train_models(X_train, y_train)

    # ✅ FIXED: results before using
    results = evaluate_models(models, X_test, y_test)

    # -------------------------
    # Model Tabs
    # -------------------------
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
            col5.metric("AUC", f"{res['auc']:.2f}")

            plot_confusion_matrix(y_test, res["y_pred"], name)

            plt.figure()
            plt.plot(res["fpr"], res["tpr"], label=f"AUC={res['auc']:.2f}")
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.legend()
            st.pyplot(plt.gcf())
            plt.clf()

    # -------------------------
    # Combined ROC
    # -------------------------
    st.subheader("📈 ROC Curve Comparison")
    plot_roc_curve(results)

    # -------------------------
    # Attendance Analysis
    # -------------------------
    st.subheader("📊 Attendance Analysis")
    plot_attendance_impact(raw_data)

    # -------------------------
    # Prediction
    # -------------------------
    st.subheader("🔍 Prediction")

    attendance = st.slider("Attendance", 0, 100, 75)
    study = st.slider("Study Hours", 0, 30, 10)
    prev = st.slider("Previous Score", 0, 100, 60)

    sample = []
    for col in feature_names:
        if col == "Attendance":
            sample.append(attendance)
        elif col == "Hours_Studied":
            sample.append(study)
        elif col == "Previous_Scores":
            sample.append(prev)
        else:
            sample.append(raw_data[col].median())

    sample = scaler.transform([sample])

    model_choice = st.selectbox("Model", ["SVM", "KNN", "ANN"])
    prob = models[model_choice].predict_proba(sample)[0][1]

    st.metric("Probability", f"{prob*100:.2f}%")

    # Explanation
    st.subheader("🧠 Explanation")
    if attendance < 60:
        st.write("• Low attendance")
    if study < 10:
        st.write("• Low study hours")
    if prev < 50:
        st.write("• Weak past performance")

    # Download
    df = pd.DataFrame({
        "Attendance": [attendance],
        "Study": [study],
        "Previous": [prev],
        "Model": [model_choice],
        "Probability": [prob]
    })

    st.download_button("Download Result", df.to_csv(index=False), "result.csv")


if __name__ == "__main__":
    main()
