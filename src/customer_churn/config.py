from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Config:
    raw_data_path: str | Path
    processed_data_path: str | Path
    target_column: str
    id_column: str
    test_size: float
    random_state: int
    cv_folds: int

    def __post_init__(self):
        if not (0 < self.test_size < 1):
            raise ValueError("test_size must be between 0 and 1.")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, 'r') as file:
            data = yaml.safe_load(file)

        return cls(**data)