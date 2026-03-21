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

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)

    data['High_Score'] = data['Exam_Score'].apply(lambda x: 1 if x >= 70 else 0)

    # 👉 ONLY USE IMPORTANT FEATURES (LOGIC FIX)
    data = data[['Attendance', 'Hours_Studied', 'Previous_Scores', 'High_Score']]

    X = data.drop('High_Score', axis=1)
    y = data['High_Score']

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
def plot_roc_curve(results):
    plt.figure()
    for name in results:
        plt.plot(results[name]["fpr"], results[name]["tpr"],
                 label=f"{name} (AUC={results[name]['auc']:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.legend()
    plt.title("ROC Comparison")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_roc_curve(results):
    plt.figure(figsize=(6,4))
    for name in results:
        plt.plot(results[name]["fpr"], results[name]["tpr"],
                 label=f"{name} (AUC={results[name]['auc']:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

def plot_input_vs_average(input_vals, averages):
    plt.figure(figsize=(5,3))
    
    labels = list(input_vals.keys())
    input_data = list(input_vals.values())
    avg_data = [averages[f] for f in labels]
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, input_data, width, label="Your Input", color='skyblue')
    plt.bar(x + width/2, avg_data, width, label="Average Success", color='orange')
    plt.xticks(x, labels)
    plt.ylabel("Value")
    plt.title("Your Input vs Average of Successful Students")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

# =========================
# MAIN APP
# =========================
def main():
    st.set_page_config(layout="wide")

    st.title("🎓 Student Performance Prediction System")
    st.caption("Predict probability of achieving high marks (≥70)")

    file_path = "StudentPerformanceFactors.csv"

    X_train, X_test, y_train, y_test, feature_names, scaler, raw_data = load_data(file_path)

    with st.spinner("Training models..."):
        models = train_models(X_train, y_train)

    results = evaluate_models(models, X_test, y_test)

    for i, name in enumerate(["SVM", "KNN", "ANN"]):
        res = results[name]
        plot_confusion_matrix(y_test, res["y_pred"], name)

    # =========================
    # MODEL SECTION
    # =========================
    st.subheader("📊 Model Comparison")
    st.divider()

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

    st.subheader("📈 ROC Curve Comparison")
    plot_roc_curve(results)

    # =========================
    # SUBHEADER INPUT
    # =========================
    st.subheader("🔧 Input Parameters")

    attendance = st.slider("Attendance (%)", 0, 100, 75)
    study = st.slider("Study Hours", 0, 30, 10)
    prev = st.slider("Previous Score", 0, 100, 60)

    model_choice = st.selectbox("Choose Model", ["SVM", "KNN", "ANN"])

    # =========================
    # PREDICTION
    # =========================
    st.subheader("🔍 Prediction Result")
    st.divider()

    sample = scaler.transform([[attendance, study, prev]])

    prob = models[model_choice].predict_proba(sample)[0][1]

    st.metric("Probability of High Marks", f"{prob*100:.2f}%")
    st.progress(float(prob))

    # Feedback
    if prob > 0.7:
        st.success("High chance of success 🎉")
    elif prob > 0.4:
        st.warning("Moderate performance ⚠️")
    else:
        st.error("Low performance risk ❌")

    # =========================
    # EXPLANATION (dynamic)
    # =========================
    st.subheader("🧠 Explanation")
    averages = raw_data[raw_data['High_Score']==1][['Attendance','Hours_Studied','Previous_Scores']].mean()
    input_vals = {"Attendance": attendance, "Hours_Studied": study, "Previous_Scores": prev}

    explanations = []
    for f in input_vals:
        if input_vals[f] >= averages[f]:
            explanations.append(f"{f} ({input_vals[f]}) is above average of successful students ({averages[f]:.1f}) ✅")
        else:
            explanations.append(f"{f} ({input_vals[f]}) is below average of successful students ({averages[f]:.1f}) ⚠️")
    for line in explanations:
        st.write("•", line)

    plot_input_vs_average(input_vals, averages)

    # =========================
    # DOWNLOAD
    # =========================
    df = pd.DataFrame({
        "Attendance": [attendance],
        "Study Hours": [study],
        "Previous Score": [prev],
        "Model": [model_choice],
        "Probability": [prob]
    })

    st.download_button("📥 Download Result", df.to_csv(index=False), "result.csv")


if __name__ == "__main__":
    main()
