import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score,
    f1_score, roc_curve, auc, classification_report
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression

# ==============================================
# 0. GRADE FUNCTION (Tutor Requirement)
# ==============================================
def get_grade(score):
    """Convert numerical score to letter grade (A+ to F)"""
    score = round(score, 0)
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 75:
        return "A-"
    elif score >= 70:
        return "B+"
    elif score >= 65:
        return "B"
    elif score >= 60:
        return "B-"
    elif score >= 55:
        return "C+"
    elif score >= 50:
        return "C"
    else:
        return "F"

# ==============================================
# 1. DATASET (Step 1: Load & Inspect)
# ==============================================
@st.cache_data
def load_and_inspect_data(file_path):
    """Step 1: Load dataset and show basic info"""
    st.subheader("📦 Step 1: Dataset Loading & Inspection")
    data = pd.read_csv(file_path)
    
    # Show dataset info
    with st.expander("View Raw Dataset & Info"):
        st.write("First 5 Rows:")
        st.dataframe(data.head(), use_container_width=True)
        st.write(f"Dataset Shape: {data.shape}")
        st.write("Missing Values:")
        st.dataframe(data.isnull().sum(), use_container_width=True)
    
    return data

# ==============================================
# 2. PREPROCESS (Step 2: Clean & Prepare)
# ==============================================
@st.cache_data
def preprocess_data(data):
    """Step 2: Clean data, handle missing values, encode if needed"""
    st.subheader("🧹 Step 2: Data Preprocessing (Cleaning)")
    
    # Drop missing values
    data_clean = data.dropna()
    st.success(f"✅ Cleaned dataset: {data_clean.shape[0]} rows (removed {data.shape[0] - data_clean.shape[0]} missing rows)")
    
    # Create classification target (High Score = ≥70)
    data_clean['High_Score'] = (data_clean['Exam_Score'] >= 70).astype(int)
    
    return data_clean

# ==============================================
# 3. FEATURE EXTRACTION (Step 3: Select Features)
# ==============================================
@st.cache_data
def extract_features(data_clean):
    """Step 3: Select relevant features for prediction"""
    st.subheader("🔍 Step 3: Feature Extraction & Selection")
    
    # Select core features (matches your input sliders)
    feature_cols = ['Attendance', 'Hours_Studied', 'Previous_Scores']
    X = data_clean[feature_cols]
    
    # Targets: 1) Classification (High/Low Score), 2) Regression (Actual Score)
    y_class = data_clean['High_Score']
    y_reg = data_clean['Exam_Score']
    
    st.write(f"✅ Selected Features: {', '.join(feature_cols)}")
    st.write(f"Target 1 (Classification): High Score (≥70) → {y_class.value_counts().to_dict()}")
    st.write(f"Target 2 (Regression): Exam Score (0-100)")
    
    return X, y_class, y_reg, feature_cols

# ==============================================
# 4. SPLIT (Step 4: Train/Test Split)
# ==============================================
@st.cache_data
def split_data(X, y_class, y_reg):
    """Step 4: Split data into training and test sets"""
    st.subheader("✂️ Step 4: Train/Test Split")
    
    # 70% train, 30% test (stratified for balanced classes)
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        X, y_class, y_reg, test_size=0.3, random_state=42, stratify=y_class
    )
    
    # Scale features (required for SVM/ANN/KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.success(f"✅ Split complete: Train={X_train.shape[0]} rows, Test={X_test.shape[0]} rows")
    st.write(f"Scaled features: Mean=0, Std=1 (for model stability)")
    
    return X_train_scaled, X_test_scaled, y_train_class, y_test_class, y_train_reg, y_test_reg, scaler, X_train, X_test

# ==============================================
# 5. TRAIN (Step 5: Train ML Models)
# ==============================================
@st.cache_resource
def train_models(X_train_scaled, y_train_class, y_train_reg):
    """Step 5: Train classification + regression models"""
    st.subheader("🤖 Step 5: Model Training")
    
    # Classification models (for High/Low Score prediction)
    class_models = {
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            early_stopping=True,
            random_state=42
        )
    }
    
    # Regression model (for actual score/grade prediction)
    reg_model = LinearRegression()
    
    # Train all models
    trained_class_models = {}
    for name, model in class_models.items():
        trained_class_models[name] = model.fit(X_train_scaled, y_train_class)
        st.write(f"✅ Trained {name} classification model")
    
    trained_reg_model = reg_model.fit(X_train_scaled, y_train_reg)
    st.write(f"✅ Trained Linear Regression model (for score prediction)")
    
    return trained_class_models, trained_reg_model

# ==============================================
# 6. EVALUATE (Step 6: Evaluate Models)
# ==============================================
def evaluate_models(class_models, X_test_scaled, y_test_class, reg_model, y_test_reg):
    """Step 6: Evaluate models with metrics, confusion matrix, ROC-AUC"""
    st.subheader("📊 Step 6: Model Evaluation")
    
    # Evaluate classification models
    class_results = {}
    for name, model in class_models.items():
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        fpr, tpr, _ = roc_curve(y_test_class, y_prob)
        class_results[name] = {
            "accuracy": accuracy_score(y_test_class, y_pred),
            "precision": precision_score(y_test_class, y_pred, zero_division=0),
            "recall": recall_score(y_test_class, y_pred, zero_division=0),
            "f1": f1_score(y_test_class, y_pred, zero_division=0),
            "auc": auc(fpr, tpr),
            "fpr": fpr,
            "tpr": tpr,
            "y_pred": y_pred,
            "y_true": y_test_class
        }
    
    # Evaluate regression model (for score prediction)
    y_pred_reg = reg_model.predict(X_test_scaled)
    reg_rmse = np.sqrt(np.mean((y_test_reg - y_pred_reg)**2))
    reg_r2 = reg_model.score(X_test_scaled, y_test_reg)
    
    # Show best classification model
    best_model = max(class_results, key=lambda x: class_results[x]["accuracy"])
    st.success(f"🏆 Best Classification Model: {best_model} (Accuracy: {class_results[best_model]['accuracy']*100:.2f}%)")
    st.info(f"📈 Regression Model (Score Prediction): RMSE={reg_rmse:.2f}, R²={reg_r2:.2f}")
    
    # Show model comparison table
    st.subheader("Classification Model Metrics")
    metrics_df = pd.DataFrame({
        "Accuracy": [class_results[m]["accuracy"] for m in class_results],
        "Precision": [class_results[m]["precision"] for m in class_results],
        "Recall": [class_results[m]["recall"] for m in class_results],
        "F1 Score": [class_results[m]["f1"] for m in class_results],
        "AUC": [class_results[m]["auc"] for m in class_results]
    }, index=class_results.keys()).round(4)
    st.dataframe(metrics_df.style.highlight_max(axis=0), use_container_width=True)
    
    # Plot confusion matrix for each model
    st.subheader("Confusion Matrices")
    tabs = st.tabs(list(class_models.keys()))
    for i, name in enumerate(class_models.keys()):
        with tabs[i]:
            cm = confusion_matrix(class_results[name]["y_true"], class_results[name]["y_pred"])
            fig, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Low Score (<70)", "High Score (≥70)"],
                        yticklabels=["Low Score (<70)", "High Score (≥70)"], ax=ax)
            ax.set_title(f"{name} Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)
    
    # Plot ROC curve comparison
    st.subheader("ROC Curve Comparison")
    fig, ax = plt.subplots(figsize=(6,4))
    for name in class_results:
        ax.plot(class_results[name]["fpr"], class_results[name]["tpr"],
                label=f"{name} (AUC={class_results[name]['auc']:.2f})")
    ax.plot([0,1],[0,1],'--', color='gray', label="Random Guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (All Models)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    
    return class_results, reg_rmse, reg_r2

# ==============================================
# 7. PREDICTION (Step 7: Predict New Student)
# ==============================================
def predict_new_student(class_models, reg_model, scaler, feature_cols, raw_data):
    """Step 7: Predict performance for a new student"""
    st.subheader("🎯 Step 7: New Student Prediction")
    st.divider()
    
    # Input form for new student
    with st.form("prediction_form"):
        st.subheader("Enter Student Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            attendance = st.slider("Attendance (%)", 0, 100, 75, help="Percentage of classes attended")
        with col2:
            study_hours = st.slider("Hours Studied per Week", 0, 30, 10, help="Average weekly study hours")
        with col3:
            prev_score = st.slider("Previous Exam Score", 0, 100, 60, help="Score from last exam")
        
        model_choice = st.selectbox("Select Classification Model", list(class_models.keys()))
        submit = st.form_submit_button("🚀 Predict Performance")
    
    if submit:
        # Prepare input data
        input_data = pd.DataFrame([[attendance, study_hours, prev_score]], columns=feature_cols)
        input_scaled = scaler.transform(input_data)
        
        # 1. Classification prediction (High/Low Score)
        high_prob = class_models[model_choice].predict_proba(input_scaled)[0][1]
        high_pred = class_models[model_choice].predict(input_scaled)[0]
        
        # 2. Regression prediction (Actual Score + Grade)
        predicted_score = reg_model.predict(input_scaled)[0]
        predicted_score = round(np.clip(predicted_score, 0, 100), 2)  # Clamp to 0-100
        predicted_grade = get_grade(predicted_score)
        
        # Display results
        st.subheader("✅ Prediction Results")
        colA, colB, colC = st.columns(3)
        colA.metric("Predicted Exam Score", f"{predicted_score}/100")
        colB.metric("Final Grade", predicted_grade)
        colC.metric("Chance of High Score (≥70)", f"{high_prob*100:.1f}%")
        st.progress(float(high_prob))
        
        # Feedback based on grade
        if predicted_score >= 70:
            st.success(f"🎉 Great performance! Predicted Grade: {predicted_grade}")
        elif predicted_score >= 50:
            st.warning(f"⚠️ Average performance. Predicted Grade: {predicted_grade}")
        else:
            st.error(f"❌ Low performance risk. Predicted Grade: {predicted_grade}")
        
        # Compare input to average of high-scoring students
        st.subheader("📊 Your Input vs Average High-Scoring Students")
        high_score_avg = raw_data[raw_data['High_Score'] == 1][feature_cols].mean()
        input_vals = {"Attendance": attendance, "Hours_Studied": study_hours, "Previous_Scores": prev_score}
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(6,3))
        x = np.arange(len(feature_cols))
        width = 0.35
        ax.bar(x - width/2, list(input_vals.values()), width, label="Your Input", color='skyblue')
        ax.bar(x + width/2, high_score_avg.values, width, label="Average High-Scoring Students", color='orange')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_cols)
        ax.set_ylabel("Value")
        ax.set_title("Your Performance vs Successful Students")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        
        # Explanation of each feature
        st.subheader("🧠 Feature Explanation")
        for feat in feature_cols:
            user_val = input_vals[feat]
            avg_val = high_score_avg[feat]
            if user_val >= avg_val:
                st.write(f"✅ {feat}: {user_val} (above average of {avg_val:.1f} for high-scoring students)")
            else:
                st.write(f"⚠️ {feat}: {user_val} (below average of {avg_val:.1f} for high-scoring students)")
        
        # Download result
        result_df = pd.DataFrame({
            "Attendance (%)": [attendance],
            "Study Hours/Week": [study_hours],
            "Previous Score": [prev_score],
            "Model Used": [model_choice],
            "Predicted Score": [predicted_score],
            "Predicted Grade": [predicted_grade],
            "High Score Chance (%)": [round(high_prob*100, 1)]
        })
        st.download_button(
            label="📥 Download Prediction Result",
            data=result_df.to_csv(index=False),
            file_name="student_performance_prediction.csv",
            mime="text/csv"
        )

# ==============================================
# MAIN APP (Orchestrate the 7-Step Flow)
# ==============================================
def main():
    st.set_page_config(page_title="Student Performance Predictor", layout="wide", page_icon="🎓")
    st.title("🎓 Student Performance Prediction System")
    st.caption("Supervised Machine Learning | 7-Step ML Pipeline (Matching Your Whiteboard Flow)")
    st.divider()
    
    # File path (update if your CSV is in a different location)
    file_path = "StudentPerformanceFactors.csv"
    
    # Step 1: Load Dataset
    raw_data = load_and_inspect_data(file_path)
    st.divider()
    
    # Step 2: Preprocess Data
    clean_data = preprocess_data(raw_data)
    st.divider()
    
    # Step 3: Extract Features
    X, y_class, y_reg, feature_cols = extract_features(clean_data)
    st.divider()
    
    # Step 4: Split Data
    X_train_scaled, X_test_scaled, y_train_class, y_test_class, y_train_reg, y_test_reg, scaler, X_train, X_test = split_data(X, y_class, y_reg)
    st.divider()
    
    # Step 5: Train Models
    class_models, reg_model = train_models(X_train_scaled, y_train_class, y_train_reg)
    st.divider()
    
    # Step 6: Evaluate Models
    class_results, reg_rmse, reg_r2 = evaluate_models(class_models, X_test_scaled, y_test_class, reg_model, y_test_reg)
    st.divider()
    
    # Step 7: Predict New Student
    predict_new_student(class_models, reg_model, scaler, feature_cols, clean_data)

if __name__ == "__main__":
    main()
