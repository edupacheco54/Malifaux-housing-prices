import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")


def load_data():
    """
    Load cleaned training dataset.
    """
    train_path = os.path.join(DATA_DIR, "train_clean.csv")
    return pd.read_csv(train_path)


def get_tree_method():
    """
    Determine appropriate tree method for XGBoost.
    """
    try:
        tree_method = os.environ.get("XGB_TREE_METHOD", None)
        if tree_method:
            return tree_method

        if xgb.config_context().get("gpu_id", None) is not None:
            return "gpu_hist"

        return "hist"

    except Exception:
        return "hist"


def train_models(X_train, y_train):
    """
    Train multiple models and select the best one.
    """
    tree_method = get_tree_method()
    print(f"Using XGBoost tree method: {tree_method}")

    models = {
        "Ridge": Ridge(alpha=10),
        "Lasso": Lasso(alpha=0.001),
        "XGBoost": XGBRegressor(
            n_estimators=500, learning_rate=0.05, tree_method=tree_method, verbosity=0
        ),
    }

    best_model = None
    best_score = float("inf")

    for name, model in models.items():
        score = -cross_val_score(
            model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error"
        ).mean()
        print(f"{name} RMSE: {score:.5f}")

        if score < best_score:
            best_score = score
            best_model = model

    best_model.fit(X_train, y_train)
    return best_model


def save_model(model):
    """
    Save the trained model.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(model, model_path)
    print("Model saved to best_model.pkl")


def main():
    """
    Run training pipeline.
    """
    df = load_data()
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    best_model = train_models(X_train, y_train)
    save_model(best_model)


if __name__ == "__main__":
    main()
