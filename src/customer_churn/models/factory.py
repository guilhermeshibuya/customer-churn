from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
}


def create_model(model_config: dict):
    """
    Create a machine learning model based on the provided configuration.

    Args:
        model_config (dict): A dictionary containing the model type and its parameters.

    Returns:
        An instance of the specified machine learning model.
    """
    model_name = model_config.get("name")
    params = model_config.get("params", {})

    if model_name not in MODEL_REGISTRY:
        available_models = ", ".join(MODEL_REGISTRY)
        raise ValueError(
            f"Model '{model_name}' is not supported. \nAvailable models are: {available_models}"
        )

    model_class = MODEL_REGISTRY[model_name]

    return model_class(**params)
