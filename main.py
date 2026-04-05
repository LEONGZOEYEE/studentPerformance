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
# LOAD DATA
# =========================
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)

    # Safety check (prevents crash)
    if "Exam_Score" not in data.columns:
        raise ValueError("Dataset must contain 'Exam_Score' column")

    data["High_Score"] = (data["Exam_Score"] >= 70).astype(int)

    data = data[["Attendance", "Hours_Studied", "Previous_Scores", "High_Score"]]

    X = data.drop("High_Score", axis=1)
    y = data["High_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X.columns, scaler, data


# =========================
# TRAIN MODELS
# =========================
@st.cache_resource
def train_models(X_train, y_train):
    return {
        "SVM": SVC(probability=True, random_state=42).fit(X_train, y_train),
        "KNN": KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train),
        "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                              early_stopping=True, random_state=42).fit(X_train, y_train)
    }


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
# PLOTS
# =========================
def plot_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "High"],
                yticklabels=["Low", "High"], ax=ax)

    ax.set_title(f"{name} Confusion Matrix")
    st.pyplot(fig)


def plot_roc(results):
    fig, ax = plt.subplots()

    for name in results:
        ax.plot(results[name]["fpr"], results[name]["tpr"],
                label=f"{name} (AUC={results[name]['auc']:.2f})")

    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title("ROC Curve Comparison")
    ax.legend()

    st.pyplot(fig)


def plot_input_vs_average(input_vals, averages):
    fig, ax = plt.subplots()

    labels = list(input_vals.keys())
    input_data = list(input_vals.values())
    avg_data = [averages[f] for f in labels]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, input_data, width, label="Input")
    ax.bar(x + width/2, avg_data, width, label="Successful Avg")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Input vs Successful Students")

    ax.legend()
    st.pyplot(fig)


# =========================
# PREDICTION FUNCTION (IMPORTANT FIX)
# =========================
def predict_performance(model, scaler, input_values):
    sample = scaler.transform([input_values])
    return model.predict_proba(sample)[0][1]


# =========================
# MAIN APP
# =========================
def main():
    st.set_page_config(layout="wide")
    st.title("🎓 Student Performance Prediction System")

    file_path = "StudentPerformanceFactors.csv"

    X_train, X_test, y_train, y_test, feature_names, scaler, raw_data = load_data(file_path)

    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    # =========================
    # MODEL SECTION
    # =========================
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

            plot_confusion_matrix(y_test, res["y_pred"], name)

    plot_roc(results)  # FIXED (only once)


    # =========================
    # INPUT SECTION
    # =========================
    st.subheader("🔧 Predict Performance")

    with st.form("input_form"):
        attendance = st.slider("Attendance (%)", 0, 100, 75)
        study = st.slider("Study Hours", 0, 30, 10)
        prev = st.slider("Previous Score", 0, 100, 60)

        model_choice = st.selectbox("Choose Model", list(models.keys()))
        submit = st.form_submit_button("Predict")

    if submit:
        input_values = [attendance, study, prev]

        prob = predict_performance(models[model_choice], scaler, input_values)

        st.metric("Probability of High Score", f"{prob*100:.2f}%")
        st.progress(float(prob))

        # feedback
        if prob > 0.7:
            st.success("High chance of success 🎉")
        elif prob > 0.4:
            st.warning("Moderate performance ⚠️")
        else:
            st.error("Low performance risk ❌")

        # explanation
        st.subheader("🧠 Explanation")

        averages = raw_data[raw_data["High_Score"] == 1][
            ["Attendance", "Hours_Studied", "Previous_Scores"]
        ].mean()

        input_vals = dict(zip(["Attendance", "Hours_Studied", "Previous_Scores"], input_values))

        for k in input_vals:
            if input_vals[k] >= averages[k]:
                st.write(f"✅ {k}: above average")
            else:
                st.write(f"⚠️ {k}: below average")

        plot_input_vs_average(input_vals, averages)

        # download
        df = pd.DataFrame([{
            "Attendance": attendance,
            "Study Hours": study,
            "Previous Score": prev,
            "Model": model_choice,
            "Probability": prob
        }])

        st.download_button("Download Result", df.to_csv(index=False), "result.csv")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()
