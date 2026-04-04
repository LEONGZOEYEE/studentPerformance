# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# ---------------------- Step 1: Load and Prepare Data ----------------------
# Load dataset
df = pd.read_csv('StudentPerformanceFactors.csv')

# Show basic info
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Drop rows with missing values (if any)
df = df.dropna()

# ---------------------- Step 2: Data Preprocessing ----------------------
# Encode categorical features
categorical_cols = [
    'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
    'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
    'School_Type', 'Peer_Influence', 'Learning_Disabilities',
    'Parental_Education_Level', 'Distance_from_Home', 'Gender'
]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Create target variable: Exam Score Levels (Classification)
# Convert continuous score to 3 classes
df['Performance_Level'] = pd.cut(
    df['Exam_Score'],
    bins=[0, 60, 80, 100],
    labels=['Low', 'Medium', 'High']
)

# Encode target
target_le = LabelEncoder()
df['Performance_Level'] = target_le.fit_transform(df['Performance_Level'])

# Split features (X) and target (y)
X = df.drop(['Exam_Score', 'Performance_Level'], axis=1)
y = df['Performance_Level']

# Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------- Step 3: Train 3 Different Models ----------------------
# Model 1: K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# Model 2: Support Vector Machine (SVM)
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

# Model 3: Artificial Neural Network (ANN)
ann = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
ann.fit(X_train_scaled, y_train)
y_pred_ann = ann.predict(X_test_scaled)

# ---------------------- Step 4: Evaluate All Models ----------------------
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation Metrics")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_le.classes_))
    
    return [accuracy, precision, recall, f1]

# Evaluate each model
metrics_knn = evaluate_model(y_test, y_pred_knn, "K-Nearest Neighbors")
metrics_svm = evaluate_model(y_test, y_pred_svm, "Support Vector Machine")
metrics_ann = evaluate_model(y_test, y_pred_ann, "Artificial Neural Network")

# ---------------------- Step 5: Compare Models ----------------------
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'KNN': metrics_knn,
    'SVM': metrics_svm,
    'ANN': metrics_ann
})

print("\n" + "="*60)
print("MODEL COMPARISON TABLE")
print("="*60)
print(metrics_df)

# Plot comparison
metrics_df.plot(x='Metric', y=['KNN', 'SVM', 'ANN'], kind='bar', figsize=(10, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------- Step 6: Predict New Student Performance ----------------------
print("\n" + "="*50)
print("Sample New Student Prediction")
print("="*50)

# Create sample new data (use mean values for demonstration)
new_student = pd.DataFrame({
    'Hours_Studied': [20],
    'Attendance': [90],
    'Parental_Involvement': [1],  # Medium
    'Access_to_Resources': [2],   # High
    'Extracurricular_Activities': [1], # Yes
    'Sleep_Hours': [8],
    'Previous_Scores': [85],
    'Motivation_Level': [2], # High
    'Internet_Access': [1], # Yes
    'Tutoring_Sessions': [2],
    'Family_Income': [1], # Medium
    'Teacher_Quality': [1], # Medium
    'School_Type': [1], # Public
    'Peer_Influence': [2], # Positive
    'Physical_Activity': [3],
    'Learning_Disabilities': [0], # No
    'Parental_Education_Level': [2], # Postgraduate
    'Distance_from_Home': [1], # Near
    'Gender': [1] # Female
})

# Scale new data
new_student_scaled = scaler.transform(new_student)

# Predict with all models
pred_knn = knn.predict(new_student_scaled)
pred_svm = svm.predict(new_student_scaled)
pred_ann = ann.predict(new_student_scaled)

print(f"KNN Prediction: {target_le.inverse_transform(pred_knn)[0]}")
print(f"SVM Prediction: {target_le.inverse_transform(pred_svm)[0]}")
print(f"ANN Prediction: {target_le.inverse_transform(pred_ann)[0]}")
