import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)

    # 🎯 Convert score → Grade
    def assign_grade(score):
        if score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"

    data['Grade'] = data['Exam_Score'].apply(assign_grade)

    # Use selected features
    data = data[['Attendance', 'Hours_Studied', 'Previous_Scores', 'Grade']]

    X = data.drop('Grade', axis=1)
    y = data['Grade']

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X.columns, scaler, data, label_encoder


# =========================
# TRAIN MODELS
# =========================
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


# =========================
# EVALUATE MODELS
# =========================
def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted'),
            "y_pred": y_pred
        }

    return results


# =========================
# PLOTS
# =========================
def plot_confusion_matrix(y_true, y_pred, model_name, label_encoder):
    cm = confusion_matrix(y_true, y_pred)

    labels = label_encoder.classes_

    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels,
                yticklabels=labels)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"{model_name} Confusion Matrix")
    st.pyplot(plt.gcf())
    plt.clf()


# =========================
# MAIN APP
# =========================
def main():
    st.set_page_config(layout="wide")

    st.title("🎓 Student Grade Prediction System")
    st.caption("Predict student final grade (A–F)")

    file_path = "StudentPerformanceFactors.csv"

    X_train, X_test, y_train, y_test, feature_names, scaler, raw_data, label_encoder = load_data(file_path)

    with st.spinner("Training models..."):
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

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{res['accuracy']*100:.2f}%")
            col2.metric("Precision", f"{res['precision']:.2f}")
            col3.metric("Recall", f"{res['recall']:.2f}")
            col4.metric("F1 Score", f"{res['f1']:.2f}")

            st.subheader("📈 Confusion Matrix")
            plot_confusion_matrix(y_test, res["y_pred"], name, label_encoder)

    # =========================
    # INPUT SECTION
    # =========================
    st.subheader("🔧 Input Parameters")

    with st.form("prediction_form"):
        attendance = st.slider("Attendance (%)", 0, 100, 75)
        study = st.slider("Study Hours", 0, 30, 10)
        prev = st.slider("Previous Score", 0, 100, 60)

        model_choice = st.selectbox("Choose Model", ["SVM", "KNN", "ANN"])

        submit = st.form_submit_button("🚀 Predict")

    # =========================
    # PREDICTION
    # =========================
    if submit:
        sample = scaler.transform([[attendance, study, prev]])

        prediction = models[model_choice].predict(sample)[0]
        grade = label_encoder.inverse_transform([prediction])[0]

        probs = models[model_choice].predict_proba(sample)[0]
        confidence = np.max(probs)

        st.subheader("🔍 Prediction Result")
        st.metric("Predicted Grade", grade)
        st.metric("Confidence", f"{confidence*100:.2f}%")

        # Feedback
        if grade == "A":
            st.success("Excellent performance 🎉")
        elif grade == "B":
            st.success("Good job 👍")
        elif grade == "C":
            st.warning("Average performance ⚠️")
        elif grade == "D":
            st.warning("Below average 😬")
        else:
            st.error("High risk of failure ❌")

        # =========================
        # DOWNLOAD RESULT
        # =========================
        df = pd.DataFrame({
            "Attendance": [attendance],
            "Study Hours": [study],
            "Previous Score": [prev],
            "Model": [model_choice],
            "Predicted Grade": [grade],
            "Confidence": [confidence]
        })

        st.download_button("📥 Download Result", df.to_csv(index=False), "result.csv")


if __name__ == "__main__":
    main()
