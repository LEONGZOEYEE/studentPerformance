import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------
# Functions
# -------------------------

def load_and_preprocess_data(file_path):
    data = pd.read_csv("student-mat.csv", sep=';')

    def grade_category(g):
        if g < 10:
            return 0
        elif g < 15:
            return 1
        else:
            return 2

    data['performance'] = data['G3'].apply(grade_category)
    data = data.drop(['G1','G2','G3'], axis=1)

    categorical_cols = data.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    X = data.drop('performance', axis=1)
    y = data['performance']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def train_svm(X_train, y_train):
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model

def train_ann(X_train, y_train, input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    return model

def evaluate_model(y_true, y_pred):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred),4),
        "Precision": round(precision_score(y_true, y_pred, average='weighted'),4),
        "Recall": round(recall_score(y_true, y_pred, average='weighted'),4),
        "F1 Score": round(f1_score(y_true, y_pred, average='weighted'),4),
        "Report": classification_report(y_true, y_pred)
    }

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.title("Student Performance Prediction")
    st.write("Click the button to train models and view performance metrics.")

    file_path = "student-mat.csv"  # Local CSV file
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

    if st.button("Train and Evaluate SVM"):
        svm_model = train_svm(X_train, y_train)
        y_pred = svm_model.predict(X_test)
        results = evaluate_model(y_test, y_pred)
        st.subheader("SVM Results")
        st.write(results)

    if st.button("Train and Evaluate KNN"):
        knn_model = train_knn(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        results = evaluate_model(y_test, y_pred)
        st.subheader("KNN Results")
        st.write(results)

    if st.button("Train and Evaluate ANN"):
        ann_model = train_ann(X_train, y_train, X_train.shape[1])
        y_pred = np.argmax(ann_model.predict(X_test), axis=1)
        results = evaluate_model(y_test, y_pred)
        st.subheader("ANN Results")
        st.write(results)

if __name__ == "__main__":
    main()
