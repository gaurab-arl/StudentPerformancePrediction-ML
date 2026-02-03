import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load CSV dataset
df = pd.read_csv("student_performance_real.csv")

print("Dataset Preview:")
print(df.head())

# Step 2: Prepare data (features and target)
X = df[['study_hours', 'attendance', 'previous_marks', 'assignments_submitted']]
y = df['final_score']

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")

# Step 7: Demonstrate prediction on new data
new_student_data = pd.DataFrame({
    'study_hours': [5, 8, 3],
    'attendance': [85, 92, 60],
    'previous_marks': [70, 85, 50],
    'assignments_submitted': [8, 10, 4]
})

predicted_scores = model.predict(new_student_data)

print("\nPredicted Final Scores for New Students:")
for i, score in enumerate(predicted_scores):
    print(f"Student {i+1}: {score:.2f}")

    
