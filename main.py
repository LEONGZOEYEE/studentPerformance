import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# =========================
# GRADE CONVERTER (MISSING BEFORE)
# =========================
def convert_to_grade(prob):
    if prob >= 0.80:
        return "A"
    elif prob >= 0.60:
        return "B"
    elif prob >= 0.40:
        return "C"
    else:
        return "D"


# =========================
# LOAD + PREPROCESS DATA
# =========================
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)

    df = df.drop("student_id", axis=1)

    # encode categorical
    le_gender = LabelEncoder()
    df["gender"] = le_gender.fit_transform(df["gender"])

    le_edu = LabelEncoder()
    df["education_level"] = le_edu.fit_transform(df["education_level"])

    # convert numeric safely
    df["final_grade"] = pd.to_numeric(df["final_grade"], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    # ✅ CREATE TARGET FIRST (FIXED)
    df["High_Grade"] = (df["final_grade"] >= 70).astype(int)

    # NOW drop final_grade
    df = df.drop("final_grade", axis=1)

    X = df.drop("High_Grade", axis=1)
    y = df["High_Grade"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X.columns, scaler, df


# =========================
# TRAIN MODELS
# =========================
@st.cache_resource
def train_models(X_train, y_train):
    models = {
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }

    for m in models.values():
        m.fit(X_train, y_train)

    return models


# =========================
# EVALUATION
# =========================
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


# =========================
# PREDICTION
# =========================
def predict(model, scaler, input_data):
    sample = scaler.transform([input_data])
    return model.predict_proba(sample)[0][1]


# =========================
# MAIN APP
# =========================
def main():
    st.set_page_config(layout="wide")
    st.title("🎓 Student Performance Prediction")

    file_path = "student_performance_academic_5000.csv"

    X_train, X_test, y_train, y_test, feature_names, scaler, raw_data = load_data(file_path)

    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    st.subheader("📊 Model Comparison")

    best_model = max(results, key=lambda x: results[x]["accuracy"])
    st.success(f"🏆 Best Model: {best_model}")

    tabs = st.tabs(list(models.keys()))

    for i, name in enumerate(models.keys()):
        with tabs[i]:
            res = results[name]

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{res['accuracy']*100:.2f}%")
            col2.metric("Precision", f"{res['precision']:.2f}")
            col3.metric("Recall", f"{res['recall']:.2f}")
            col4.metric("F1", f"{res['f1']:.2f}")
            col5.metric("AUC", f"{res['auc']:.2f}")

            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, res["y_pred"]), annot=True, fmt="d", ax=ax)
            st.pyplot(fig)

    # ROC
    fig, ax = plt.subplots()
    for name in results:
        ax.plot(results[name]["fpr"], results[name]["tpr"], label=name)
    ax.plot([0,1],[0,1],"--")
    ax.legend()
    st.pyplot(fig)

    # =========================
    # INPUT
    # =========================
    st.subheader("🔧 Prediction")

    with st.form("form"):
        age = st.slider("Age", 15, 30, 18)
        gender = st.selectbox("Gender (0=F,1=M)", [0,1])
        education = st.selectbox("Education Level", [0,1,2])

        study = st.slider("Study Hours", 0, 50, 10)
        attendance = st.slider("Attendance", 0, 100, 75)
        assignment = st.slider("Assignment Score", 0, 100, 70)
        exam = st.slider("Exam Score", 0, 100, 65)
        extra = st.selectbox("Extra Curricular", [0,1])

        model_choice = st.selectbox("Model", list(models.keys()))
        submit = st.form_submit_button("Predict")

    if submit:
        input_data = [age, gender, education, study, attendance, assignment, exam, extra]

        prob = predict(models[model_choice], scaler, input_data)

        grade = convert_to_grade(prob)

        st.metric("Predicted Grade", grade)
        st.progress(float(prob))

        st.write(f"Probability: {prob*100:.2f}%")

        if grade == "A":
            st.success("Excellent 🎉")
        elif grade == "B":
            st.info("Good 👍")
        elif grade == "C":
            st.warning("Average ⚠️")
        else:
            st.error("Risk ❌")


if __name__ == "__main__":
    main()
