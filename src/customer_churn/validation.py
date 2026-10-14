import pandas as pd


def validate_schemma(data: pd.DataFrame, schema: dict) -> bool:
    """
    Validate the schema of a DataFrame against a given schema dictionary.

    Args:
        data (pd.DataFrame): The DataFrame to validate.
        schema (dict): A dictionary defining the expected schema, where keys are column names and values are expected data types.

    Returns:
        bool: True if the DataFrame matches the schema, False otherwise.
    """
    for column, expected_dtype in schema.items():
        if column not in data.columns:
            print(f"Column '{column}' is missing from the DataFrame.")
            return False
        actual_dtype = data[column].dtype
        if actual_dtype != expected_dtype:
            print(
                f"Column '{column}' has dtype '{actual_dtype}', expected '{expected_dtype}'."
            )
            return False

    return True
