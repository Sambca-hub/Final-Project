"""Dataset validation, student marks prediction, and model evaluation."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REQUIRED_COLUMNS = ["Department", "Semester", "Subject", "Marks", "Attendance"]
FEATURE_COLUMNS = ["Department", "Semester", "Subject", "Attendance"]


def load_training_data(df):
    """Return a cleaned copy of the uploaded dataset or raise a useful error."""
    if df is None:
        raise ValueError("Please upload a student dataset first.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    clean_df = df.copy()
    for col in ("Marks", "Attendance", "Semester"):
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    for col in ("Department", "Subject"):
        clean_df[col] = clean_df[col].astype("string").str.strip()
        clean_df[col] = clean_df[col].replace("", pd.NA)

    clean_df = clean_df.dropna(subset=REQUIRED_COLUMNS)
    if clean_df.empty:
        raise ValueError("No valid records remain after checking required fields.")
    if not clean_df["Marks"].between(0, 100).all():
        raise ValueError("Marks must be between 0 and 100.")
    if not clean_df["Attendance"].between(0, 100).all():
        raise ValueError("Attendance must be between 0 and 100.")
    if (clean_df["Semester"] <= 0).any():
        raise ValueError("Semester must be a positive number.")

    return clean_df.reset_index(drop=True)


def _build_model():
    categorical = ["Department", "Subject"]
    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)
    ], remainder="passthrough")
    return Pipeline([
        ("preprocessing", preprocessing),
        ("regression", LinearRegression())
    ])


def _validate_prediction_inputs(department, semester, subject, attendance):
    if not str(department).strip() or not str(subject).strip():
        raise ValueError("Department and subject must be selected.")
    try:
        semester = float(semester)
        attendance = float(attendance)
    except (TypeError, ValueError):
        raise ValueError("Semester and attendance must be numeric.") from None
    if semester <= 0:
        raise ValueError("Semester must be a positive number.")
    if not 0 <= attendance <= 100:
        raise ValueError("Attendance must be between 0 and 100.")
    return semester, attendance


def predict_marks(department, semester, subject, attendance, df):
    """Fit a model on valid uploaded records and predict marks on a 0–100 scale."""
    df = load_training_data(df)
    if len(df) < 5:
        raise ValueError(
            "At least 5 valid student records are required for a prediction. "
            "More records generally produce a more reliable estimate."
        )
    semester, attendance = _validate_prediction_inputs(
        department, semester, subject, attendance
    )

    model = _build_model()
    model.fit(df[FEATURE_COLUMNS], df["Marks"])
    student = pd.DataFrame([{
        "Department": str(department).strip(),
        "Semester": semester,
        "Subject": str(subject).strip(),
        "Attendance": attendance,
    }])
    prediction = float(model.predict(student)[0])
    return round(max(0.0, min(100.0, prediction)), 2)


def evaluate_model(df):
    """Evaluate with a held-out test set; require enough rows for useful metrics."""
    df = load_training_data(df)
    if len(df) < 10:
        raise ValueError(
            "At least 10 valid student records are required for evaluation "
            "(with a held-out test set of at least 3 records)."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURE_COLUMNS], df["Marks"], test_size=0.30, random_state=42
    )
    model = _build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions = [round(max(0.0, min(100.0, float(p))), 2) for p in predictions]

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions) if len(y_test) >= 2 else float("nan")

    comparison_df = pd.DataFrame({
        "Actual Marks": y_test.reset_index(drop=True),
        "Predicted Marks": predictions,
    })
    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "R2 Score": round(float(r2), 4),
        "Training Records": len(X_train),
        "Testing Records": len(X_test),
        "Comparison": comparison_df,
        "Note": "Metrics are estimates from one random holdout split; they may vary with small or unrepresentative datasets.",
    }
