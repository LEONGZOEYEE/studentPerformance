import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# -------------------------
# Load & preprocess
# -------------------------
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)

    # Convert Exam Score → High Marks (1/0)
    data['High_Score'] = data['Exam_Score'].apply(lambda x: 1 if x >= 70 else 0)
    data = data.drop(['Exam_Score'], axis=1)

    # Encode categorical columns
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
@st.cache_data
def train_models(X_train, y_train):
    models = {}

    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    models['SVM'] = svm

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    models['KNN'] = knn

    ann = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    ann.fit(X_train, y_train)
    models['ANN'] = ann

    return models

# -------------------------
# Evaluate model accuracy
# -------------------------
def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = (acc, y_pred)
    return results

# -------------------------
# Attendance vs Performance Graph
# -------------------------
def plot_attendance_impact(data):
    bins = [0, 60, 80, 100]
    labels = ['Low (0-60)', 'Medium (60-80)', 'High (80-100)']
    data['Attendance_Group'] = pd.cut(data['Attendance'], bins=bins, labels=labels)
    grouped = data.groupby('Attendance_Group')['High_Score'].mean()

    plt.figure(figsize=(5,3))
    sns.barplot(x=grouped.index, y=grouped.values, palette="Blues_d")
    plt.title("Attendance vs Probability of High Marks")
    plt.ylabel("Probability of High Score")
    plt.xlabel("Attendance Level")
    st.pyplot(plt.gcf())
    plt.clf()

# -------------------------
# Plot Confusion Matrix
# -------------------------
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(3,2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    st.pyplot(plt.gcf())
    plt.clf()

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(page_title="Student Performance Prediction", layout="wide")
    st.title("🎓 Student Performance Prediction System")
    st.write("Predict probability of achieving high marks (≥70)")

    file_path = "StudentPerformanceFactors.csv"

    # Load original data for graph
    X_train, X_test, y_train, y_test, feature_names, scaler, raw_data = load_data(file_path)
    models = train_models(X_train, y_train)

    # -------------------------
    # Model Accuracy & Confusion Matrix
    # -------------------------
    st.subheader("📊 Model Comparison & Confusion Matrices")
    results = evaluate_models(models, X_test, y_test)

    # Use tabs for each model
    tabs = st.tabs(["SVM", "KNN", "ANN"])
    for i, model_name in enumerate(["SVM", "KNN", "ANN"]):
        with tabs[i]:
            acc, y_pred = results[model_name]
            st.metric("Accuracy", f"{round(acc*100,2)}%")
            plot_confusion_matrix(y_test, y_pred, model_name)

    # -------------------------
    # Attendance Impact Graph
    # -------------------------
    st.subheader("📈 Attendance Impact Analysis")
    plot_attendance_impact(raw_data)

    # -------------------------
    # Predict New Student
    # -------------------------
    st.subheader("🔍 Predict Student Performance")

    attendance = st.slider("Attendance (%)", 0, 100, 75)
    study_hours = st.slider("Study Hours", 0, 30, 10)
    previous_score = st.slider("Previous Score", 0, 100, 60)

    # Define feature list from feature names
    feature_list = list(feature_names)

    # Use median values for all features initially
    sample = np.array([raw_data[col].median() if col != 'High_Score' else 0 for col in feature_list], dtype=float)

    if "Attendance" in feature_list:
        sample[feature_list.index("Attendance")] = attendance
    if "Hours_Studied" in feature_list:
        sample[feature_list.index("Hours_Studied")] = study_hours
    if "Previous_Scores" in feature_list:
        sample[feature_list.index("Previous_Scores")] = previous_score

    sample = sample.reshape(1, -1)
    sample = scaler.transform(sample)

    # Model selection
    model_choice = st.selectbox("Choose Model", ["SVM", "KNN", "ANN"])
    model = models[model_choice]

    prob = model.predict_proba(sample)[0][1]
    percentage = round(prob * 100, 2)
    st.metric("Probability of High Marks", f"{percentage}%")

    if prob > 0.7:
        st.success("High chance of good performance 🎉")
    elif prob > 0.4:
        st.warning("Moderate chance ⚠️")
    else:
        st.error("Low chance ❌")

if __name__ == "__main__":
    main()
