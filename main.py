from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PASS_MARK = 50
FEATURES = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignments_submitted",
]


def main() -> None:
    dataset_path = Path(__file__).resolve().with_name("student_performance_real.csv")
    df = pd.read_csv(dataset_path)

    required_columns = FEATURES + ["final_score"]
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing_columns)}"
        )

    if df["final_score"].isna().any():
        raise ValueError("The final_score column contains missing values.")

    X = df[FEATURES]
    y = (df["final_score"] >= PASS_MARK).astype(int)
    if y.nunique() != 2:
        raise ValueError("The dataset must contain both Pass and Fail examples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Dataset rows: {len(df)}")
    print(f"Pass mark: {PASS_MARK} (inclusive)")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
    print(
        f"Pass precision: {precision_score(y_test, predictions, zero_division=0):.2%}"
    )
    print(f"Pass recall: {recall_score(y_test, predictions, zero_division=0):.2%}")
    print(f"Pass F1 score: {f1_score(y_test, predictions, zero_division=0):.2%}")

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    print("\nConfusion matrix (rows: actual Fail/Pass; columns: predicted Fail/Pass):")
    print(matrix)

    new_students = pd.DataFrame(
        {
            "study_hours": [5, 8, 3],
            "attendance": [85, 92, 60],
            "previous_marks": [70, 85, 50],
            "assignments_submitted": [8, 10, 4],
        }
    )
    outcomes = model.predict(new_students)
    pass_probabilities = model.predict_proba(new_students)[:, 1]

    print("\nPredictions for example students:")
    for index, (outcome, probability) in enumerate(
        zip(outcomes, pass_probabilities), start=1
    ):
        label = "Pass" if outcome == 1 else "Fail"
        print(f"Student {index}: {label} (pass probability: {probability:.1%})")


if __name__ == "__main__":
    main()
