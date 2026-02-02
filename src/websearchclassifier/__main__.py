import argparse
import sys
import traceback
from pathlib import Path
from typing import Tuple

from websearchclassifier import (
    DatasetConfig,
    FastTextSearchClassifierConfig,
    ModelType,
    Pipeline,
    SearchClassifierConfig,
    TfidfSearchClassifierConfig,
)
from websearchclassifier.utils import logger


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train web search classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--model-type",
        type=ModelType,
        choices=["tfidf", "fasttext"],
        required=True,
        help="Type of model to train",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/train.csv"),
        help="Path to training data CSV file (default: data/train.csv)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of column containing prompt texts (default: text)",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="search_required",
        help="Name of column containing boolean labels (default: search_required)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Path to save trained model (overrides config)",
    )

    return parser.parse_args()


def load_configuration(
    config_path: Path,
    model_type: ModelType,
    output_path_override: Path | None = None,
) -> Tuple[SearchClassifierConfig, Path]:
    """
    Load model configuration from YAML file or use defaults.

    Args:
        config_path: Path to YAML configuration file
        model_type: Type of model to train (tfidf or fasttext)
        output_path_override: Optional path to override config output path

    Returns:
        Tuple of (model config, output path)

    Raises:
        ValueError: If model type is unknown
        KeyError: If model type not found in config file
    """
    try:
        logger.info("Loading configuration from %s", config_path)
        config, output_path_from_config = Pipeline.load_config(config_path, model_type)
        output_path = Path(output_path_override or output_path_from_config)
        return config, output_path

    except FileNotFoundError as exception:
        logger.warning("Config file %s not found, using defaults", config_path)
        output_path = Path(output_path_override or f"models/{model_type}_classifier.pkl")

        match model_type:
            case "tfidf":
                config = TfidfSearchClassifierConfig()
            case "fasttext":
                config = FastTextSearchClassifierConfig()
            case _:
                logger.error("Unknown model type: %s", model_type)
                logger.info("Available: tfidf, fasttext")
                raise ValueError(f"Unknown model type: {model_type}") from exception

        return config, output_path


def run_pipeline(
    pipeline: Pipeline,
    config: SearchClassifierConfig,
    output_path: Path,
    model_type: ModelType,
) -> None:
    """
    Execute training pipeline and display usage instructions.

    Args:
        pipeline: Initialized pipeline instance
        config: Model configuration
        output_path: Path to save trained model
        model_type: Type of model being trained

    Raises:
        FileNotFoundError: If required files are missing
        Exception: For unexpected errors during training
    """
    try:
        model = pipeline.train_and_save(config=config, output_path=output_path)
        pipeline.test_predictions(model)
        _display_usage_instructions(model_type, output_path, config)

    except FileNotFoundError as exception:
        logger.error("File not found: %s", exception)
        raise
    except Exception as exception:
        logger.error("Unexpected error: %s", exception)
        traceback.print_exc()
        raise


def _display_usage_instructions(
    model_type: ModelType,
    output_path: Path,
    config: SearchClassifierConfig,
) -> None:
    """
    Display instructions for loading and using the trained model.

    Args:
        model_type: Type of model trained
        output_path: Path where model was saved
        config: Model configuration used
    """
    logger.info("To use the model:")

    match model_type:
        case "tfidf":
            logger.info("  from tfidf import TfidfSearchClassifier")
            logger.info("  model = TfidfSearchClassifier.load('%s')", output_path)
        case "fasttext":
            embeddings_path = (
                config.embeddings_path if isinstance(config, FastTextSearchClassifierConfig) else "cc.pl.300.bin"
            )
            logger.info("  from ftext import FastTextSearchClassifier")
            logger.info("  model = FastTextSearchClassifier.load('%s', '%s')", output_path, embeddings_path)

    logger.info("  result = model.predict('your prompt here')")


def main() -> None:
    """
    Main entry point for training pipeline.

    Orchestrates argument parsing, configuration loading, and pipeline execution.
    """
    logger.info("Web Search Classifier - Training Pipeline")
    logger.info("=" * 60)

    args = parse_arguments()

    try:
        config, output_path = load_configuration(
            config_path=args.config,
            model_type=args.model_type,
            output_path_override=args.output_path,
        )
    except (ValueError, KeyError) as exception:
        logger.error("Configuration error: %s", exception)
        sys.exit(1)

    dataset_config = DatasetConfig(
        dataset_path=args.data_path,
        prompt_column=args.text_column,
        label_column=args.label_column,
    )
    pipeline = Pipeline(dataset_config=dataset_config)

    run_pipeline(
        pipeline=pipeline,
        config=config,
        output_path=output_path,
        model_type=args.model_type,
    )
