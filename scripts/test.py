import os
import pandas as pd
import joblib
import numpy as np


def load_data():
    """
    Load the cleaned test data.
    """
    test_path = os.path.join("..", "data", "test_clean.csv")
    return pd.read_csv(test_path)


def load_model():
    """
    Load the trained model.
    """
    model_path = os.path.join("..", "models", "best_model.pkl")
    return joblib.load(model_path)


def generate_predictions(model, test_df):
    """
    Generate predictions for test data.
    """
    y_pred = np.exp(model.predict(test_df))
    return y_pred


def save_submission(predictions):
    """
    Save predictions to final_submission.csv
    """
    sample_path = os.path.join("..", "sample_submission.csv")
    submission_df = pd.read_csv(sample_path)
    submission_df["SalePrice"] = predictions

    output_path = os.path.join("..", "final_submission.csv")
    submission_df.to_csv(output_path, index=False)

    print(f"Submission file saved to {output_path}")


def main():
    """
    Run the test pipeline.
    """
    test_df = load_data()
    model = load_model()
    predictions = generate_predictions(model, test_df)
    save_submission(predictions)


if __name__ == "__main__":
    main()
