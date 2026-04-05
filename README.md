# exam-score-predictor

# 📊 Online Exam Score Prediction Dashboard

## 📌 Project Overview

This project is an **interactive data analytics dashboard** that predicts student exam scores using **Least Squares Regression (Linear Regression)**.

It helps analyze how different factors like:
- Study hours
- Sleep hours
- Attendance
- Previous scores  

affect student performance.

The dashboard is built using **Streamlit** and provides:
- Real-time predictions  
- Visual insights  
- Error analysis  

---

## 🌐 IoT Relevance

Although this is a data science project, it relates to **IoT (Internet of Things)** in real-world scenarios:

👉 In IoT-based education systems:
- Smart devices can track **attendance automatically**
- Wearables can monitor **sleep patterns**
- Study apps can track **study hours**

These real-time data inputs can be used to:
➡️ Predict student performance  
➡️ Improve learning strategies  

---

## 🎯 Features

✅ Predict exam score using machine learning  
✅ Interactive dashboard (Streamlit)  
✅ Clean and simple UI  
✅ Data visualizations (graphs & charts)  
✅ Error analysis (Residual & Accuracy)  
✅ Feature importance analysis  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## 📂 Dataset

The dataset contains the following columns:

- `hours_studied`
- `sleep_hours`
- `attendance_percent`
- `previous_scores`
- `exam_score`

---

## ⚙️ Installation & Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/exam-score-predictor.git
cd exam-score-predictor
Step 2: Install Dependencies
Bash
pip install -r requirements.txt
Step 3: Run Application
Bash
streamlit run app.py
🚀 How to Use
Open the dashboard in browser
Enter student details from sidebar:
Study hours
Sleep hours
Attendance
Previous scores
Click Predict Score
View:
Predicted score
Graphs
Analysis
📊 Visualizations Included
Scatter plots (relationship analysis)
Histogram (score distribution)
Bar chart (score categories)
Residual plot (error analysis)
Actual vs Predicted plot
Feature importance graph
📉 Model Details
Algorithm: Linear Regression
Technique: Least Squares Method
Evaluation Metrics:
R² Score (Accuracy)
Mean Squared Error (MSE)
Mean Absolute Error (MAE)
🧠 Insights
More study hours → higher scores
Better attendance → improved performance
Previous scores → strong predictor
📌 Future Improvements
Add real-time IoT data integration
Use advanced ML models (Random Forest, XGBoost)
Deploy on cloud for public access
Add login system
👨‍💻 Author
Sampooorn Trivedi
B.E. Computer Science
⭐ Contribution
If you like this project, give it a ⭐ on GitHub!
