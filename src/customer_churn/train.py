from customer_churn.models.factory import create_model
from sklearn.pipeline import Pipeline
from pathlib import Path
import pandas as pd
import joblib


def build_model(config: dict):
    """
    Builds a model based on the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration for model initialization.

    Returns:
        A scikit-learn model initialized based on the provided configuration.
    """
    model = create_model(config["model"])
    return model


def train_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Trains the provided pipeline on the training data.

    Args:
        pipeline (Pipeline): A scikit-learn Pipeline that includes preprocessing and model initialization.
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training labels.

    Returns:
        Pipeline: The trained pipeline.
    """
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(pipeline: Pipeline, output_path: str | Path):
    """
    Saves the trained pipeline to the specified output path.

    Args:
        pipeline (Pipeline): The trained scikit-learn Pipeline to save.
        output_path (str | Path): The path where the model should be saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)


def load_model(model_path: str | Path):
    """
    Loads a trained pipeline from the specified model path.

    Args:
        model_path (str | Path): The path to the saved model.

    Returns:
        Pipeline: The loaded scikit-learn Pipeline.
    """
    return joblib.load(model_path)
