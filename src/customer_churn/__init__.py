import argparse


def main() -> None:
    from customer_churn.config import Config
    from customer_churn.pipeline import run_training_pipeline

    parser = argparse.ArgumentParser(
        description="Run the customer churn training pipeline."
    )
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Path to the YAML config file (default: configs/train.yaml).",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    run_training_pipeline(config)
