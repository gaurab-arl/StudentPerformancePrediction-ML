# Student Pass/Fail Prediction

This project trains a binary classifier to predict whether a student will pass
based on academic features. A final score of **50 or higher** is labeled Pass;
a lower score is labeled Fail.

## Features

- Load student records from `student_performance_real.csv`
- Impute missing feature values and scale numeric features
- Train a Logistic Regression classifier
- Evaluate held-out results with accuracy, precision, recall, F1, and a confusion matrix
- Predict Pass/Fail outcomes and pass probabilities for example students

## Dataset

The CSV contains `study_hours`, `attendance`, `previous_marks`,
`assignments_submitted`, and `final_score`. The final score creates the Pass/Fail
label and is not included as a model feature.

The current dataset has 63 records, so the results are an initial estimate and
may not generalize to other schools or student populations. Use a larger,
representative set of actual records and a separate evaluation dataset before
using this model for consequential decisions. Do not present generated records
as real observations.

## Run

Install the dependencies:

```bash
pip install pandas scikit-learn
```

Run the pipeline from the project directory:

```bash
python main.py
```

The script uses a stratified 80/20 train/test split with a fixed random seed.
It prints the held-out metrics, confusion matrix, and example predictions.

## Technologies

- Python
- Pandas
- Scikit-learn

## Author

Gaurab Aryal
