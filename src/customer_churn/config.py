from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Config:
    raw_data_path: str | Path
    processed_data_path: str | Path
    model_output_path: str | Path
    target_column: str
    id_column: str
    test_size: float
    validation_size: float
    random_state: int
    cv_folds: int
    model: dict

    def __post_init__(self):
        if not (0 < self.test_size < 1):
            raise ValueError("test_size must be between 0 and 1.")
        if not (0 < self.validation_size < 1):
            raise ValueError("validation_size must be between 0 and 1.")
        if self.test_size + self.validation_size >= 1:
            raise ValueError("test_size + validation_size must be less than 1.")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2.")
        if not isinstance(self.model, dict) or "name" not in self.model:
            raise ValueError("model must be a mapping with a 'name' key.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"The config file at {path} was not found.")

        with open(path, 'r') as file:
            data = yaml.safe_load(file)

        if not isinstance(data, dict):
            raise ValueError(f"The config file at {path} must contain a mapping.")

        return cls(**data)
