from contextlib import contextmanager
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn


def init_tracking(
    tracking_uri: str | None = None, experiment_name: str = "customer-churn"
):
    """
    Initializes MLflow tracking.

    Args:
        tracking_uri (str | None): The URI of the MLflow tracking server. If None, the default local file-based
            tracking will be used.
        experiment_name (str): The name of the MLflow experiment to use. If the experiment does not exist, it will
            be created.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def flatten_config(config: dict) -> dict:
    params = {
        "model_name": config.model.get("name"),
        "test_size": config.test_size,
        "validation_size": config.validation_size,
        "random_state": config.random_state,
        "cv_folds": config.cv_folds,
    }
    params.update(
        {
            f"model__{key}": value
            for key, value in config.model.get("params", {}).items()
        }
    )
    return params


@contextmanager
def training_run(config, run_name: str | None = None):
    """
    Context manager for an MLflow training run.

    Args:
        config (dict): The configuration dictionary containing model parameters and other settings.
        run_name (str | None): The name of the MLflow run. If None, a default name will be used.
    """
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(flatten_config(config))
        mlflow.log_dict({"model": config.model}, "config.json")
        yield run


def log_sklearn_model(
    pipeline: Pipeline, X_example, registered_model_name: str | None = None
):
    """
    Logs a scikit-learn model to MLflow.

    Args:
        pipeline: The scikit-learn pipeline or model to log.
        X_example: Example input data for the model, used to infer the input signature.
        registered_model_name (str | None): The name of the registered model in MLflow. If provided, the model will
            be registered under this name.
    """
    signature = infer_signature(X_example, pipeline.predict(X_example))
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="model",
        input_example=X_example,
        registered_model_name=registered_model_name,
        signature=signature,
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
