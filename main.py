import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


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

    # Features & target
    X = data[['Attendance', 'Hours_Studied', 'Previous_Scores']]
    y = data['Exam_Score']   # 🔥 IMPORTANT: regression target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
    models = {}

    models["SVM"] = SVR(kernel='rbf').fit(X_train, y_train)

    models["KNN"] = KNeighborsRegressor(
        n_neighbors=7,
        weights="distance"
    ).fit(X_train, y_train)

    models["ANN"] = MLPRegressor(
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

        results[name] = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "y_pred": y_pred
        }

    return results


# =========================
# PLOT ACTUAL VS PREDICTED
# =========================
def plot_results(y_test, y_pred, model_name):
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title(f"{model_name} Prediction")

    st.pyplot(plt.gcf())
    plt.clf()


# =========================
# MAIN APP
# =========================
def main():
    st.set_page_config(layout="wide")

    st.title("🎓 Student Score & Grade Prediction System")
    st.caption("Predict exam score using ML → convert to grade")

    file_path = "StudentPerformanceFactors.csv"

    X_train, X_test, y_train, y_test, scaler, raw_data = load_data(file_path)

    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    # =========================
    # MODEL COMPARISON
    # =========================
    st.subheader("📊 Model Comparison")
    st.divider()

    best_model = max(results, key=lambda x: results[x]["r2"])
    st.success(f"🏆 Best Model: {best_model}")

    tabs = st.tabs(["SVM", "KNN", "ANN"])

    for i, name in enumerate(["SVM", "KNN", "ANN"]):
        with tabs[i]:
            res = results[name]

            col1, col2, col3 = st.columns(3)

            col1.metric("MAE", f"{res['mae']:.2f}")
            col2.metric("RMSE", f"{res['rmse']:.2f}")
            col3.metric("R² Score", f"{res['r2']:.2f}")

            st.subheader("Actual vs Predicted")
            plot_results(y_test, res["y_pred"], name)

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

        pred_score = model.predict(sample)[0]
        grade = get_grade(pred_score)

        st.success(f"📊 Predicted Score: {pred_score:.2f}")
        st.success(f"🎓 Predicted Grade: {grade}")

        # =========================
        # PERFORMANCE INSIGHT
        # =========================
        st.subheader("🧠 Performance Insight")

        avg = raw_data[['Attendance', 'Hours_Studied', 'Previous_Scores']].mean()

        input_vals = {
            "Attendance": attendance,
            "Hours_Studied": study,
            "Previous_Scores": prev
        }

        for k in input_vals:
            if input_vals[k] >= avg[k]:
                st.write(f"✔ {k}: above average")
            else:
                st.write(f"⚠ {k}: below average")

        # =========================
        # DOWNLOAD
        # =========================
        df = pd.DataFrame({
            "Attendance": [attendance],
            "Study Hours": [study],
            "Previous Score": [prev],
            "Model": [model_choice],
            "Predicted Score": [pred_score],
            "Predicted Grade": [grade]
        })

        st.download_button(
            "📥 Download Result",
            df.to_csv(index=False),
            "grade_prediction.csv"
        )


if __name__ == "__main__":
    main()
