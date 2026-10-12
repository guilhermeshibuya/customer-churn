import pandas as pd
from pathlib import Path


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the data: {e}")


def split_dataset(
    data: pd.DataFrame,
    target_column: str,
    validation_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split

    if target_column not in data.columns:
        raise ValueError(
            f"The target column '{target_column}' is not present in the DataFrame."
        )

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(validation_size + test_size), random_state=random_state
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(test_size / (validation_size + test_size)),
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_categorical_columns(df: pd.DataFrame) -> list:
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    return categorical_columns


def get_numerical_columns(df: pd.DataFrame) -> list:
    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    return numerical_columns
