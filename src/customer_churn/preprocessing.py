from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

from customer_churn.models.factory import create_model


def build_preprocessor(
    categorical_features: list, numerical_features: list
) -> ColumnTransformer:
    """
    Builds a preprocessor for categorical and numerical features.

    Args:
        categorical_features (list): List of categorical feature names.
        numerical_features (list): List of numerical feature names.

    Returns:
        ColumnTransformer: A ColumnTransformer that applies OneHotEncoder to categorical features and StandardScaler to numerical features.
    """
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numerical_transformer = StandardScaler()

    column_transformer = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numerical_transformer, numerical_features),
        ]
    )

    return column_transformer


def fit_preprocessor(
    preprocessor: ColumnTransformer, X_train: pd.DataFrame
) -> ColumnTransformer:
    """
    Fits the preprocessor to the training data.

    Args:
        preprocessor (ColumnTransformer): The preprocessor to fit.
        X_train: The training data.

    Returns:
        ColumnTransformer: The fitted preprocessor.
    """
    preprocessor.fit(X_train)
    return preprocessor


def transform_features(preprocessor: ColumnTransformer, data: pd.DataFrame):
    """
    Transforms the features using the fitted preprocessor.

    Args:
        preprocessor (ColumnTransformer): The fitted preprocessor.
        data (pd.DataFrame): The data to transform.

    Returns:
        np.ndarray: The transformed data.
    """
    transformed_data = preprocessor.transform(data)

    return transformed_data


def build_training_pipeline(config: dict):
    """
    Builds a training pipeline that includes preprocessing and model initialization based on the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration for preprocessing and model initialization.

    Returns:
        Pipeline: A scikit-learn Pipeline that includes preprocessing and model initialization.
    """
    preprocessor = build_preprocessor(
        categorical_features=config["categorical_features"],
        numerical_features=config["numerical_features"],
    )

    classifier = create_model(config["model"])

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )

    return pipeline
