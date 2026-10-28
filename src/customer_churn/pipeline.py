from customer_churn.config import Config
from customer_churn.data import (
    load_dataset,
    split_dataset,
    encode_target,
    get_categorical_columns,
    get_numerical_columns,
)
from customer_churn.preprocessing import (
    build_preprocessor,
    build_training_pipeline,
)
from customer_churn.train import build_model, train_model, save_model
from customer_churn.evaluate import evaluate_model


def run_training_pipeline(config: Config) -> dict:
    """
    Run the training pipeline for customer churn prediction.

    Loads the dataset, splits it, preprocesses the features, trains the model
    and evaluates it on the held-out test set.

    Args:
        config (Config): Consolidated configuration for the run.

    Returns:
        dict: Evaluation metrics (precision, recall, f1_score).
    """
    data = load_dataset(config.raw_data_path)

    X_train, _, X_test, y_train, _, y_test = split_dataset(
        data,
        target_column=config.target_column,
        validation_size=config.validation_size,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    if config.id_column in X_train.columns:
        X_train = X_train.drop(columns=[config.id_column])
        X_test = X_test.drop(columns=[config.id_column])

    y_train = encode_target(y_train)
    y_test = encode_target(y_test)

    categorical_features = get_categorical_columns(X_train)
    numerical_features = get_numerical_columns(X_train)
    preprocessor = build_preprocessor(categorical_features, numerical_features)

    model = build_model(config.model)
    pipeline = build_training_pipeline(preprocessor, model)
    pipeline = train_model(pipeline, X_train, y_train)

    eval_results = evaluate_model(pipeline, X_test, y_test)
    print("Evaluation Results:", eval_results)

    save_model(pipeline, config.model_output_path)

    return eval_results
