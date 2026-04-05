import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import seaborn as sns

# =========================
# TITLE
# =========================
st.title("Student Grade Prediction - ML Comparison System")

# =========================
# LOAD DATASET
# =========================
@st.cache_data
def load_data():
    # You can replace this with your dataset path
    df = pd.read_csv("student_data.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.write(df.head())

# =========================
# PREPROCESSING
# =========================

# Example: assume last column is target
target_column = df.columns[-1]

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(target_column, axis=1)
y = df[target_column]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# MODELS
# =========================

models = {
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "ANN": MLPClassifier(hidden_layer_sizes=(50,50), max_iter=500)
}

results = {}

# =========================
# TRAIN + EVALUATE
# =========================
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

# =========================
# RESULTS TABLE
# =========================
st.subheader("Model Comparison")

results_df = pd.DataFrame(results).T[["Accuracy", "Precision", "Recall", "F1 Score"]]
st.write(results_df)

# =========================
# CONFUSION MATRIX DISPLAY
# =========================
model_choice = st.selectbox("Select Model for Confusion Matrix", list(models.keys()))

cm = results[model_choice]["Confusion Matrix"]

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.title(f"{model_choice} Confusion Matrix")

st.pyplot(fig)

# =========================
# PREDICTION SECTION
# =========================
st.subheader("Make a Prediction")

input_data = []

for col in X.columns:
    value = st.number_input(f"{col}", value=0.0)
    input_data.append(value)

input_array = np.array(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_array)

best_model = models["SVM"]  # you can change or choose best automatically
prediction = best_model.predict(input_scaled)

st.write("Predicted Output:", prediction[0])
