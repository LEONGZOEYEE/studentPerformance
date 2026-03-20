import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===============================
# Streamlit App
# ===============================
def main():
    st.title("Student Performance Prediction (Practical Exercise)")
    st.write("Predict Pass/Fail based on student dataset using KNN, SVM, ANN.")

    # Upload CSV
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("First 5 rows of dataset:")
        st.dataframe(data.head())

        # Ensure Exam_Score exists
        if "Exam_Score" not in data.columns:
            st.error("CSV must include 'Exam_Score' column.")
            return

        # Pass threshold
        threshold = st.slider("Pass threshold (Exam_Score >= ?)", 0, 100, 50)
        data["Pass"] = data["Exam_Score"].apply(lambda x: 1 if x >= threshold else 0)

        st.write("Pass/Fail distribution:")
        st.bar_chart(data["Pass"].value_counts())

        # Encode categorical features
        data_encoded = pd.get_dummies(data)
        X = data_encoded.drop(["Exam_Score", "Pass"], axis=1)
        y = data_encoded["Pass"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train models
        models = {
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "SVM": SVC(),
            "ANN": MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000, random_state=42)
        }

        st.write("### Model Results")
        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            st.write(f"**{name}**")
            st.write(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1 Score: {f1:.3f}")
            results.append({"Model": name, "Accuracy": acc, "F1 Score": f1})

        # Comparison chart
        if results:
            df_results = pd.DataFrame(results).set_index("Model")
            st.write("### Model Comparison (Accuracy & F1 Score)")
            st.bar_chart(df_results)

if __name__ == "__main__":
    main()
