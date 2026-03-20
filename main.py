# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from joblib import load

# ===============================
# 2. LOAD DATASET
# ===============================
# Replace with the path where you saved the CSV
data = pd.read_csv("StudentPerformanceFactors.csv")

print("First 10 rows:")
print(data.head(10))

# =============================== 
# 3. CREATE PASS/FAIL TARGET 
# =============================== 
# Assuming Exam_Score column exists 
# Check min/max to see if all scores are >=50 (causing all Pass=1)
print("Exam_Score min:", data["Exam_Score"].min())
print("Exam_Score max:", data["Exam_Score"].max())

# Adjust threshold if needed (e.g., change 50 to 60 to create more fails)
data["Pass"] = data["Exam_Score"].apply(lambda x: 1 if x >= 50 else 0)

# Check for class imbalance in the original data
print("Unique values in Pass:", data["Pass"].unique())
print("Value counts in Pass:", data["Pass"].value_counts())

# Check for class imbalance in the original data
print("Unique values in Pass:", data["Pass"].unique())
print("Value counts in Pass:", data["Pass"].value_counts())

# ===============================
# 4. PREPROCESSING
# ===============================
# Convert categorical columns to numeric using one-hot encoding
data_encoded = pd.get_dummies(data)

# Drop original target
X = data_encoded.drop(["Exam_Score", "Pass"], axis=1)
y = data_encoded["Pass"]

# ===============================
# 5. TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Check classes in train/test splits
print("Unique values in y_train:", y_train.unique())
print("Unique values in y_test:", y_test.unique())

# If y_train has only 1 class, remove stratify to allow random split
if len(y_train.unique()) == 1:
    print("Warning: y_train has only 1 class. Removing stratify for random split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # No stratify
    )
    print("After random split - Unique in y_train:", y_train.unique())
    print("After random split - Unique in y_test:", y_test.unique())

# ===============================
# 6. NORMALIZATION
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================== 
# 7. MODEL 1: KNN 
# =============================== 
if len(y_train.unique()) > 1:
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
else:
    print("Warning: y_train has only 1 class. Skipping KNN.")
    y_pred_knn = None

# =============================== 
# 8. MODEL 2: SVM 
# =============================== 
if len(y_train.unique()) > 1:
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
else:
    print("Warning: y_train has only 1 class. Skipping SVM.")
    y_pred_svm = None

# =============================== 
# 9. MODEL 3: ANN 
# =============================== 
if len(y_train.unique()) > 1:
    ann = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000, random_state=42)
    ann.fit(X_train, y_train)
    y_pred_ann = ann.predict(X_test)
else:
    print("Warning: y_train has only 1 class. Skipping ANN.")
    y_pred_ann = None

# =============================== 
# 10. EVALUATION FUNCTION 
# =============================== 
def evaluate_model(name, y_test, y_pred):
    if y_pred is None:
        print(f"--- {name} Skipped (insufficient classes) ---")
        return
    print(f"--- {name} Results ---")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("Precision:", round(precision_score(y_test, y_pred), 3))
    print("Recall:", round(recall_score(y_test, y_pred), 3))
    print("F1 Score:", round(f1_score(y_test, y_pred), 3))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
# Evaluate all models
evaluate_model("KNN", y_test, y_pred_knn)
evaluate_model("SVM", y_test, y_pred_svm)
evaluate_model("ANN", y_test, y_pred_ann)

# Only include models that were fitted
fitted_models = []
accuracies = []
precisions = []
recalls = []
f1s = []

if y_pred_knn is not None:
    fitted_models.append("KNN")
    accuracies.append(round(accuracy_score(y_test, y_pred_knn), 3))
    precisions.append(round(precision_score(y_test, y_pred_knn), 3))
    recalls.append(round(recall_score(y_test, y_pred_knn), 3))
    f1s.append(round(f1_score(y_test, y_pred_knn), 3))

if y_pred_svm is not None:
    fitted_models.append("SVM")
    accuracies.append(round(accuracy_score(y_test, y_pred_svm), 3))
    precisions.append(round(precision_score(y_test, y_pred_svm), 3))
    recalls.append(round(recall_score(y_test, y_pred_svm), 3))
    f1s.append(round(f1_score(y_test, y_pred_svm), 3))

if y_pred_ann is not None:
    fitted_models.append("ANN")
    accuracies.append(round(accuracy_score(y_test, y_pred_ann), 3))
    precisions.append(round(precision_score(y_test, y_pred_ann), 3))
    recalls.append(round(recall_score(y_test, y_pred_ann), 3))
    f1s.append(round(f1_score(y_test, y_pred_ann), 3))

if fitted_models:
    results = pd.DataFrame({
        "Model": fitted_models,
        "Accuracy": accuracies,
        "Precision": precisions,
        "Recall": recalls,
        "F1 Score": f1s
    })
    print("=== Model Comparison ===")
    print(results)
else:
    print("No models could be fitted due to insufficient classes.")

# =============================== 
# 12. BAR CHART FOR COMPARISON 
# =============================== 
if fitted_models:
    results.set_index("Model")[["Accuracy","F1 Score"]].plot(kind="bar", figsize=(8,5))
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.show()
else:
    print("Skipping chart due to no fitted models.")
