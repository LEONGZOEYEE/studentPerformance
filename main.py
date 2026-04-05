import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import seaborn as sns
import matplotlib.pyplot as plt

st.title("Student Performance Prediction System")

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("student_performance_academic_5000.csv")

st.write(df.head())

# =====================
# CLEAN COLUMN NAMES
# =====================
df.columns = df.columns.str.strip()

st.write("Columns:", df.columns)

# =====================
# TARGET COLUMN (CHANGE THIS IF NEEDED)
# =====================
target_column = st.selectbox("Select Target Column", df.columns)

# =====================
# ENCODING
# =====================
label_encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# =====================
# SPLIT DATA
# =====================
X = df.drop(target_column, axis=1)
y = df[target_column]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =====================
# MODELS
# =====================
models = {
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "ANN": MLPClassifier(hidden_layer_sizes=(50,50), max_iter=500)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "CM": confusion_matrix(y_test, y_pred)
    }

# =====================
# RESULTS
# =====================
st.subheader("Model Comparison")
st.dataframe(pd.DataFrame(results).T)

# =====================
# CONFUSION MATRIX
# =====================
model_choice = st.selectbox("Select Model", list(models.keys()))

cm = results[model_choice]["CM"]

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)
