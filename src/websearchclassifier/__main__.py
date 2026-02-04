import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

from websearchclassifier import (
    DatasetConfig,
    FastTextSearchClassifierConfig,
    ModelType,
    Pipeline,
    SearchClassifierConfig,
    TFIDFSearchClassifierConfig,
)
from websearchclassifier.config import ClassifierType, HerBERTSearchClassifierConfig
from websearchclassifier.config.classifier.implementations.logistic import LogisticRegressionConfig
from websearchclassifier.config.classifier.implementations.mlp import MLPConfig
from websearchclassifier.config.classifier.implementations.svm import SVMConfig
from websearchclassifier.utils import logger


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
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
        choices=list(ModelType),
        required=True,
        help="Type of model to train",
    )
    parser.add_argument(
        "--classifier-type",
        type=ClassifierType,
        choices=list(ClassifierType),
        required=True,
        help="Type of classifier to train",
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
    classifier_type: ClassifierType,
    output_path_override: Optional[Path] = None,
) -> Tuple[SearchClassifierConfig, Path]:
    """
    Load model configuration from YAML file or use defaults.

    Args:
        config_path: Path to YAML configuration file.
        model_type: Type of model to train (`tfidf`, `fasttext`, or `herbert`).
        classifier_type: Type of classifier to train (`logistic_regression` or `svm`).
        output_path_override: Optional path to override config output path.

    Returns:
        Tuple of (model config, output path).

    Raises:
        ValueError: If model type is unknown.
        KeyError: If model type not found in config file.
    """
    try:
        logger.info("Loading configuration from %s", config_path)
        config, output_path_from_config = Pipeline.load_config(config_path, model_type, classifier_type)
        output_path = Path(output_path_override or output_path_from_config)
        return config, output_path

    except FileNotFoundError as exception:
        logger.warning("Config file %s not found, using defaults", config_path)
        output_path = Path(output_path_override or f"models/{model_type}_classifier.pkl")

        classifier_config: Union[LogisticRegressionConfig, SVMConfig, MLPConfig]
        match classifier_type:
            case ClassifierType.LOGISTIC_REGRESSION:
                classifier_config = LogisticRegressionConfig()
            case ClassifierType.SVM:
                classifier_config = SVMConfig()
            case ClassifierType.MLP:
                classifier_config = MLPConfig()
            case _:
                logger.error("Unknown classifier type: %s", classifier_type)
                logger.info("Available: %s", ", ".join(ct.name for ct in ClassifierType))
                raise ValueError(f"Unknown classifier type: {classifier_type}") from exception

        match model_type:
            case ModelType.TFIDF:
                config = TFIDFSearchClassifierConfig(classifier_config=classifier_config)
            case ModelType.FASTTEXT:
                config = FastTextSearchClassifierConfig(classifier_config=classifier_config)
            case ModelType.HERBERT:
                config = HerBERTSearchClassifierConfig(classifier_config=classifier_config)
            case _:
                logger.error("Unknown model type: %s", model_type)
                logger.info("Available: %s", ", ".join(mt.name for mt in ModelType))
                raise ValueError(f"Unknown model type: {model_type}") from exception

        return config, output_path


def run_pipeline(
    pipeline: Pipeline,
    config: SearchClassifierConfig,
    output_path: Path,
) -> None:
    """
    Execute training pipeline and display usage instructions.

    Args:
        pipeline: Initialized pipeline instance.
        config: Model configuration.
        output_path: Path to save trained model.

    Raises:
        FileNotFoundError: If required files are missing
        Exception: For unexpected errors during training
    """
    try:
        model = pipeline.train_and_save(config=config, output_path=output_path)
        pipeline.test_predictions(model)

    except FileNotFoundError as exception:
        logger.error("File not found: %s", exception)
        raise
    except Exception as exception:
        logger.error("Unexpected error: %s", exception)
        raise


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
            classifier_type=args.classifier_type,
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
    )
