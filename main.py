import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# -------------------------
# Functions
# -------------------------

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # ✅ Convert Attendance into categories
    # (Adjust based on your dataset scale if needed)
    def attendance_category(a):
        if a < 50:
            return 0   # Low
        elif a < 75:
            return 1   # Medium
        else:
            return 2   # High

    data['attendance_level'] = data['Attendance'].apply(attendance_category)

    # Drop original attendance column from features
    X = data.drop(['Attendance', 'attendance_level'], axis=1)
    y = data['attendance_level']

    # Encode categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_svm(X_train, y_train):
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model


def train_ann(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_true, y_pred):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average='weighted'), 4),
        "Recall": round(recall_score(y_true, y_pred, average='weighted'), 4),
        "F1 Score": round(f1_score(y_true, y_pred, average='weighted'), 4),
        "Report": classification_report(y_true, y_pred)
    }


# -------------------------
# Streamlit UI
# -------------------------

def main():
    st.title("📊 Student Performance System")
    st.write("Predict Student Performance")

    file_path = "StudentPerformanceFactors.csv"

    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    # -------------------------
    # SVM
    # -------------------------
    if st.button("Train and Evaluate SVM"):
        model = train_svm(X_train, y_train)
        y_pred = model.predict(X_test)
        results = evaluate_model(y_test, y_pred)

        st.subheader("SVM Results")
        st.write(results)

    # -------------------------
    # KNN
    # -------------------------
    if st.button("Train and Evaluate KNN"):
        model = train_knn(X_train, y_train)
        y_pred = model.predict(X_test)
        results = evaluate_model(y_test, y_pred)

        st.subheader("KNN Results")
        st.write(results)

    # -------------------------
    # ANN (MLP)
    # -------------------------
    if st.button("Train and Evaluate ANN"):
        model = train_ann(X_train, y_train)
        y_pred = model.predict(X_test)
        results = evaluate_model(y_test, y_pred)

        st.subheader("ANN Results (MLPClassifier)")
        st.write(results)


if __name__ == "__main__":
    main()
