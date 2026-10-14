from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score


def generate_predictions(model: Pipeline, X):
    """
    Generates predictions using the provided model and input features.

    Args:
        model: A trained machine learning model.
        X: Input features for prediction.

    Returns:
        np.ndarray: The predicted values.
    """
    predictions = model.predict(X)
    return predictions


def calculate_metrics(y_true, predictions) -> dict:
    """
    Calculates precision, recall, and F1 score based on true labels and predictions.

    Args:
        y_true: True labels.
        predictions: Predicted labels.

    Returns:
        dict: A dictionary containing precision, recall, and F1 score.
    """
    precision = precision_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)

    return {"precision": precision, "recall": recall, "f1_score": f1}


def evaluate_model(model: Pipeline, X_test, y_test) -> dict:
    """
    Evaluates the model on the test data and returns evaluation metrics.

    Args:
        model: A trained machine learning model.
        X_test: Test features.
        y_test: True labels for the test data.

    Returns:
        dict: A dictionary containing evaluation metrics (precision, recall, F1 score).
    """
    predictions = generate_predictions(model, X_test)
    metrics = calculate_metrics(y_test, predictions)

    return metrics
