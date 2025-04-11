import os
import pandas as pd
import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
OUTPUT_PATH = os.path.join(ROOT, "final_submission.csv")
SAMPLE_PATH = os.path.join(ROOT, "sample_submission.csv")


def load_data():
    """
    Load the cleaned test dataset.
    """
    test_path = os.path.join(DATA_DIR, "test_clean.csv")
    return pd.read_csv(test_path)


def load_model():
    """
    Load the best trained model.
    """
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    return joblib.load(model_path)


def generate_predictions(model, test_df):
    """
    Generate predictions using the trained model.
    """
    y_pred = np.exp(model.predict(test_df))
    return y_pred


def save_submission(predictions):
    """
    Save predictions to final_submission.csv using sample_submission.csv as a template.
    """
    submission_df = pd.read_csv(SAMPLE_PATH)
    submission_df["SalePrice"] = predictions
    submission_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Submission file saved to: {OUTPUT_PATH}")


def main():
    """
    Run the test/prediction pipeline.
    """
    test_df = load_data()
    model = load_model()
    predictions = generate_predictions(model, test_df)
    save_submission(predictions)


if __name__ == "__main__":
    main()
