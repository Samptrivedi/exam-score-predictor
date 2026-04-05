import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import train_model

# ======================
# 🎨 STYLING
# ======================

st.set_page_config(page_title="Exam Dashboard", layout="wide")

st.markdown("""
<style>
.main {background-color: #f5f7fa;}
h1 {color: #2c3e50; text-align: center;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>📊 Student Performance Dashboard</h1>", unsafe_allow_html=True)

# ======================
# 📂 LOAD DATA
# ======================

data = pd.read_csv("data.csv")

model, feature_columns, r2, mse, mae, y_test, y_pred, X = train_model()

# ======================
# 📊 KPI CARDS
# ======================

st.markdown("### 📊 Model Performance")

c1, c2, c3 = st.columns(3)

c1.markdown(f"""
<div style='background-color:#ffffff;padding:15px;border-radius:10px;text-align:center;'>
<h4>📈 R² Score</h4>
<h2 style='color:green;'>{r2:.3f}</h2>
</div>
""", unsafe_allow_html=True)

c2.markdown(f"""
<div style='background-color:#ffffff;padding:15px;border-radius:10px;text-align:center;'>
<h4>📉 MSE</h4>
<h2 style='color:red;'>{mse:.2f}</h2>
</div>
""", unsafe_allow_html=True)

c3.markdown(f"""
<div style='background-color:#ffffff;padding:15px;border-radius:10px;text-align:center;'>
<h4>📊 MAE</h4>
<h2 style='color:blue;'>{mae:.2f}</h2>
</div>
""", unsafe_allow_html=True)

st.info(f"🔍 Model Confidence: {r2*100:.2f}%")

st.markdown("---")

# ======================
# 🎛 SIDEBAR
# ======================

st.sidebar.header("🎛 Control Panel")

input_data = {}

for col in feature_columns:
    input_data[col] = st.sidebar.slider(
        col,
        float(data[col].min()),
        float(data[col].max()),
        float(data[col].mean())
    )

if st.sidebar.button("🚀 Predict Score"):
    arr = np.array([list(input_data.values())])
    pred = model.predict(arr)

    st.success(f"🎯 Predicted Exam Score: {pred[0]:.2f}")
    st.progress(int(pred[0]))

# ======================
# 📊 FILTER
# ======================

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Filter Data")

min_score = st.sidebar.slider("Minimum Score", 0, 100, 0)
filtered_data = data[data["exam_score"] >= min_score]

# ======================
# 📑 TABS
# ======================

tab1, tab2, tab3 = st.tabs(["📂 Data", "📈 Visualizations", "📉 Analysis"])

# ======================
# 📂 TAB 1
# ======================

with tab1:
    st.subheader("Dataset Preview")
    st.write(filtered_data.head())

    st.subheader("Statistics")
    st.write(filtered_data.describe())

# ======================
# 📈 TAB 2 (CLEAN GRAPHS ONLY)
# ======================

with tab2:

    col1, col2 = st.columns(2)

    # 1️⃣ Scatter (Study)
    with col1:
        fig, ax = plt.subplots(figsize=(3.5,2.5))
        ax.scatter(data["hours_studied"], data["exam_score"])
        ax.set_title("Hours Studied vs Score")
        plt.tight_layout()
        st.pyplot(fig)

    # 2️⃣ Scatter (Attendance)
    with col2:
        fig, ax = plt.subplots(figsize=(3.5,2.5))
        ax.scatter(data["attendance_percent"], data["exam_score"])
        ax.set_title("Attendance vs Score")
        plt.tight_layout()
        st.pyplot(fig)

    # 3️⃣ Histogram
    with col1:
        fig, ax = plt.subplots(figsize=(3.5,2.5))
        ax.hist(data["exam_score"], bins=10)
        ax.set_title("Score Distribution")
        plt.tight_layout()
        st.pyplot(fig)

    # 4️⃣ Bar Chart (Categories)
    with col2:
        bins = [0, 40, 70, 100]
        labels = ["Low", "Medium", "High"]
        data["category"] = pd.cut(data["exam_score"], bins=bins, labels=labels)

        fig, ax = plt.subplots(figsize=(3.5,2.5))
        data["category"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Score Categories")
        plt.tight_layout()
        st.pyplot(fig)

# ======================
# 📉 TAB 3 (ANALYSIS)
# ======================

with tab3:

    col1, col2 = st.columns(2)

    # Residual Plot
    with col1:
        fig, ax = plt.subplots(figsize=(3.5,2.5))
        res = y_test - y_pred
        ax.scatter(y_pred, res)
        ax.axhline(0)
        ax.set_title("Residual Plot")
        plt.tight_layout()
        st.pyplot(fig)

    # Actual vs Predicted
    with col2:
        fig, ax = plt.subplots(figsize=(3.5,2.5))
        sns.regplot(x=y_test, y=y_pred, ax=ax)
        ax.set_title("Actual vs Predicted")
        plt.tight_layout()
        st.pyplot(fig)

    # Feature Importance
    st.subheader("📊 Feature Importance")

    coef = model.coef_
    fig, ax = plt.subplots(figsize=(4,2.5))
    ax.bar(feature_columns, coef)
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)

    st.info("""
📌 Insights:
- More study hours → higher score  
- Attendance improves performance  
- Previous scores indicate consistency  
""")