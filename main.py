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

    # Target
    data['High_Score'] = data['Exam_Score'].apply(lambda x: 1 if x >= 70 else 0)

    # 🔥 Use ONLY 3 features (SAFE + MATCH INPUT)
    data = data[['Attendance','Hours_Studied','Previous_Scores','High_Score']]

    X = data.drop('High_Score', axis=1)
    y = data['High_Score']

    # ✅ 70/30 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
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
# EVALUATE
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
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low","High"],
                yticklabels=["Low","High"])
    plt.title(f"{model_name} Confusion Matrix")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_roc_curve(results):
    plt.figure()
    for name in results:
        plt.plot(results[name]["fpr"], results[name]["tpr"],
                 label=f"{name} (AUC={results[name]['auc']:.2f})")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve Comparison")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_input_vs_average(input_vals, raw_data):
    averages = raw_data[['Attendance','Hours_Studied','Previous_Scores']].mean()

    labels = list(input_vals.keys())
    input_data = list(input_vals.values())
    avg_data = [averages[f] for f in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, input_data, width, label="Your Input")
    plt.bar(x + width/2, avg_data, width, label="Average")

    plt.xticks(x, labels)
    plt.legend()
    plt.title("Your Input vs Average")

    st.pyplot(plt.gcf())
    plt.clf()

# =========================
# MAIN
# =========================
def main():
    st.set_page_config(layout="wide")
    st.title("🎓 Student Performance Prediction System")

    file_path = "StudentPerformanceFactors.csv"

    X_train, X_test, y_train, y_test, scaler, raw_data = load_data(file_path)

    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    # =========================
    # MODEL RESULTS
    # =========================
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

    # =========================
    # PREDICT
    # =========================
    if st.button("🚀 Predict"):
        sample = scaler.transform([[attendance, study, prev]])
        prob = models[model_choice].predict_proba(sample)[0][1]

        estimated_score = prob * 100
        grade = get_grade(estimated_score)

        st.metric("Estimated Score", f"{estimated_score:.2f}")
        st.metric("Predicted Grade", grade)
        st.progress(float(prob))

        # Feedback
        if prob > 0.7:
            st.success("High chance of success 🎉")
        elif prob > 0.4:
            st.warning("Moderate performance ⚠️")
        else:
            st.error("Low performance risk ❌")

        # =========================
        # EXTRA VISUAL
        # =========================
        input_vals = {
            "Attendance": attendance,
            "Hours_Studied": study,
            "Previous_Scores": prev
        }

        st.subheader("📊 Input vs Average")
        plot_input_vs_average(input_vals, raw_data)

# =========================
if __name__ == "__main__":
    main()
