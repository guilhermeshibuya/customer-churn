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


def build_training_pipeline(preprocessor: ColumnTransformer, model: object) -> Pipeline:
    """
    Builds a training pipeline that combines the preprocessor and the model.

    Args:
        preprocessor (ColumnTransformer): The fitted preprocessor.
        model (object): The machine learning model.

    Returns:
        Pipeline: A scikit-learn Pipeline that applies the preprocessor and then the model.
    """
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    return pipeline
