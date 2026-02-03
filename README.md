# 📊 Student Performance Prediction using Linear Regression

This project demonstrates a **simple Machine Learning model** that predicts a student's **final score** based on academic and behavioral factors using **Linear Regression**.

It is designed for **beginners** to understand the complete ML workflow:
data loading → preprocessing → training → evaluation → prediction.

---

## 🚀 Features

- Load real-world student performance data from a CSV file
- Train a **Linear Regression** model
- Evaluate model performance using **Mean Squared Error (MSE)**
- Predict final scores for new students

---

## 🧠 Dataset Description

The dataset (`student_performance_real.csv`) contains the following columns:

| Column Name | Description |
|------------|------------|
| `study_hours` | Number of hours studied |
| `attendance` | Attendance percentage |
| `previous_marks` | Marks from previous exams |
| `assignments_submitted` | Number of assignments submitted |
| `final_score` | Final exam score (target variable) |

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **Scikit-learn**

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gaurab-arl/StudentPerformancePrediction-ML.git
   ```

2. Navigate to the project folder:
   ```bash
   cd student-performance-prediction
   ```

3. Install required libraries:
   ```bash
   pip install pandas scikit-learn
   ```

---

## ▶️ How to Run

1. Make sure `student_performance_real.csv` is in the same directory.

2. Run the Python script:
   ```bash
   python main.py
   ```

---

## 📈 Model Workflow

1. Load dataset using Pandas
2. Separate features and target variable
3. Split data into training and testing sets (80/20)
4. Train a Linear Regression model
5. Evaluate using Mean Squared Error
6. Predict scores for new students

---

## 🔍 Sample Prediction Output

```
Predicted Final Scores for New Students:
Student 1: 74.52
Student 2: 86.13
Student 3: 58.90
```

---

## 📚 Learning Outcomes

- Understanding supervised learning
- Hands-on experience with Linear Regression
- Real-world dataset handling
- Model evaluation basics

---

## 📌 Future Improvements

- Add data visualization
- Try other ML models (Decision Tree, Random Forest)
- Add feature scaling
- Save and load trained models

---

## 👤 Author

**Gaurab Aryal**  
Beginner ML & Web Developer 🚀