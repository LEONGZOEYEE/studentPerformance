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

# 设置matplotlib中文/防乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['High_Score'] = data['Exam_Score'].apply(lambda x: 1 if x >= 70 else 0)
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
def plot_confusion_matrix(y_true, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(4,3))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["低分", "高分"],
                yticklabels=["低分", "高分"])
    ax.set_ylabel("真实值")
    ax.set_xlabel("预测值")
    ax.set_title(f"{model_name} 混淆矩阵")
    st.pyplot(fig)

def plot_roc_curve(results):
    fig, ax = plt.subplots(figsize=(6,4))
    for name in results:
        ax.plot(results[name]["fpr"], results[name]["tpr"],
                 label=f"{name} (AUC={results[name]['auc']:.2f})")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label="随机猜测")
    ax.set_xlabel("假正率")
    ax.set_ylabel("真正率")
    ax.set_title("ROC曲线对比")
    ax.legend()
    st.pyplot(fig)

def plot_input_vs_average(input_vals, averages):
    fig, ax = plt.subplots(figsize=(5,3))
    labels = list(input_vals.keys())
    input_data = list(input_vals.values())
    avg_data = [averages[f] for f in labels]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, input_data, width, label="你的输入", color='skyblue')
    ax.bar(x + width/2, avg_data, width, label="优秀学生均值", color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("数值")
    ax.set_title("你的数据 VS 优秀学生均值")
    ax.legend()
    st.pyplot(fig)

# =========================
# MAIN APP
# =========================
def main():
    st.set_page_config(layout="wide")
    st.title("🎓 学生成绩预测系统")
    st.caption("预测考试高分概率（≥70分）")

    file_path = "StudentPerformanceFactors.csv"
    X_train, X_test, y_train, y_test, feature_names, scaler, raw_data = load_data(file_path)

    with st.spinner("模型训练中..."):
        models = train_models(X_train, y_train)

    results = evaluate_models(models, X_test, y_test)

    # 模型对比区
    st.subheader("📊 模型性能对比")
    st.divider()
    best_model = max(results, key=lambda x: results[x]["accuracy"])
    st.success(f"🏆 最优模型：{best_model}")

    tabs = st.tabs(["SVM", "KNN", "ANN"])
    for i, name in enumerate(["SVM", "KNN", "ANN"]):
        with tabs[i]:
            res = results[name]
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("准确率", f"{res['accuracy']*100:.2f}%")
            c2.metric("精确率", f"{res['precision']:.2f}")
            c3.metric("召回率", f"{res['recall']:.2f}")
            c4.metric("F1分数", f"{res['f1']:.2f}")
            c5.metric("AUC", f"{res['auc']:.2f}")
            st.subheader("混淆矩阵")
            plot_confusion_matrix(y_test, res["y_pred"], name)
            st.subheader("ROC曲线")
            plot_roc_curve(results)

    # 输入区域
    st.subheader("🔧 输入学生参数")
    with st.form("pred_form"):
        attendance = st.slider("出勤率(%)", 0, 100, 75)
        study = st.slider("每日学习时长", 0, 30, 10)
        prev = st.slider("过往成绩", 0, 100, 60)
        model_choice = st.selectbox("选择预测模型", ["SVM", "KNN", "ANN"])
        submit = st.form_submit_button("🚀 开始预测")

    # 预测逻辑（只保留一处，消除重复bug）
    if submit:
        sample = scaler.transform([[attendance, study, prev]])
        prob = models[model_choice].predict_proba(sample)[0][1]

        st.subheader("🔍 预测结果")
        st.metric("高分概率", f"{prob*100:.2f}%")
        st.progress(float(prob))

        # 评价提示
        if prob > 0.7:
            st.success("🎉 高分概率很高！")
        elif prob > 0.4:
            st.warning("⚠️ 中等水平，还有提升空间")
        else:
            st.error("❌ 成绩风险较高，建议加强学习/出勤")

        # 对比解释
        st.subheader("🧠 结果解释")
        avg_high = raw_data[raw_data['High_Score']==1][['Attendance','Hours_Studied','Previous_Scores']].mean()
        input_vals = {
            "出勤率": attendance,
            "学习时长": study,
            "过往成绩": prev
        }
        avg_map = {
            "出勤率": avg_high["Attendance"],
            "学习时长": avg_high["Hours_Studied"],
            "过往成绩": avg_high["Previous_Scores"]
        }

        for k,v in input_vals.items():
            if v >= avg_map[k]:
                st.write(f"✅ {k}({v}) 高于优秀学生均值({avg_map[k]:.1f})")
            else:
                st.write(f"⚠️ {k}({v}) 低于优秀学生均值({avg_map[k]:.1f})")

        plot_input_vs_average(input_vals, avg_map)

        # 下载报告
        df_out = pd.DataFrame({
            "出勤率":[attendance],
            "学习时长":[study],
            "过往成绩":[prev],
            "选用模型":[model_choice],
            "高分概率(%)":[round(prob*100,2)]
        })
        st.download_button("📥 下载预测报告", df_out.to_csv(index=False), "预测结果.csv")

if __name__ == "__main__":
    main()
    
