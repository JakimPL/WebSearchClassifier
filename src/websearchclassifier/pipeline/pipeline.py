"""
Training pipeline for web search classifiers.
Automates dataset loading, model training, and persistence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Optional, Tuple

import yaml

from websearchclassifier.config import (
    DatasetConfig,
    FastTextSearchClassifierConfig,
    SearchClassifierConfig,
    TfidfSearchClassifierConfig,
)
from websearchclassifier.dataset import Dataset
from websearchclassifier.model import FastTextSearchClassifier, SearchClassifier, TfidfSearchClassifier
from websearchclassifier.pipeline.models import ModelType
from websearchclassifier.utils import logger


class Pipeline:
    """
    Training pipeline for web search classifiers.

    Handles the complete workflow: data loading, model training, evaluation,
    and persistence. Supports multiple classifier types with unified interface.

    Example:
        >>> pipeline = Pipeline(data_path="data/train.csv")
        >>> model = pipeline.train_and_save(
        ...     classifier_class=TfidfSearchClassifier,
        ...     output_path="model.pkl"
        ... )
        >>> predictions = model.predict(["test prompt"])
    """

    def __init__(self, dataset_config: DatasetConfig) -> None:
        """
        Initialize the pipeline.

        Args:
            dataset_config: Configuration object for the dataset
        """
        self.dataset_config = dataset_config
        self.dataset: Optional[Dataset] = None

    def load_data(self) -> Dataset:
        """
        Load and validate training data using Dataset class.

        Returns:
            Dataset instance with prompts and labels

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If required columns are missing
        """
        logger.info(f"Loading dataset from {self.dataset_config.path}...")
        self.dataset = Dataset.load(self.dataset_config)

        num_positive = self.dataset.positive
        num_negative = self.dataset.negative

        logger.info(f"Loaded {len(self.dataset.prompts)} examples")
        logger.info(f"  - Needs search: {num_positive} ({num_positive/self.dataset.size*100:.1f}%)")
        logger.info(f"  - No search:    {num_negative} ({num_negative/self.dataset.size*100:.1f}%)")

        return self.dataset

    def train(self, config: SearchClassifierConfig) -> SearchClassifier[Any]:
        """
        Train a classifier on loaded data.

        Args:
            config: Configuration object for the classifier

        Returns:
            Trained classifier instance

        Raises:
            RuntimeError: If data hasn't been loaded yet
        """
        if self.dataset is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        classifier: SearchClassifier[Any]
        if isinstance(config, FastTextSearchClassifierConfig):
            logger.info(f"Initializing FastTextSearchClassifier...")
            classifier = FastTextSearchClassifier(config=config)

            embeddings_path = config.embeddings_path
            if not embeddings_path.exists():
                raise FileNotFoundError(
                    f"FastText embeddings not found at: {embeddings_path}\n"
                    f"Download Polish model:\n"
                    f"wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.bin.gz\n"
                    f"gunzip cc.pl.300.bin.gz"
                )
            classifier.load_embeddings(embeddings_path)
        elif isinstance(config, TfidfSearchClassifierConfig):
            logger.info(f"Initializing TfidfSearchClassifier...")
            classifier = TfidfSearchClassifier(config=config)
        else:
            raise ValueError(f"Unknown config type: {type(config)}")

        logger.info(f"Training {classifier.__class__.__name__}...")
        classifier.fit(self.dataset.prompts.tolist(), self.dataset.labels.tolist())

        return classifier

    def train_and_save(
        self,
        config: SearchClassifierConfig,
        output_path: Path,
    ) -> SearchClassifier[Any]:
        """
        Train a classifier and save it to disk.

        This is the main pipeline method combining data loading, training,
        and model persistence.

        Args:
            config: Configuration object for the classifier
            output_path: Path to save the trained model

        Returns:
            Trained classifier instance

        Example:
            >>> dataset_config = DatasetConfig(path=Path("data/train.csv"), extension="csv", prompt_column="text", label_column="search_required")
            >>> pipeline = Pipeline(dataset_config)
            >>> config = TfidfSearchClassifierConfig(max_features=3000)
            >>> model = pipeline.train_and_save(config, "tfidf_model.pkl")
        """
        if self.dataset is None:
            self.load_data()

        assert self.dataset is not None
        classifier: SearchClassifier[Any] = self.train(config)

        logger.info(f"Saving model to {output_path}...")

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        classifier.save(output_path)

        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info(f"Model saved to: {output_path}")
        logger.info(f"Model type: {classifier.__class__.__name__}")
        logger.info(f"Training samples: {len(self.dataset.prompts)}")

        return classifier

    def test_predictions(
        self,
        classifier: SearchClassifier[Any],
        test_prompts: Optional[List[str]] = None,
    ) -> None:
        """
        Test classifier on example prompts and display results.

        Args:
            classifier: Trained classifier to test
            test_prompts: List of prompts to test (uses defaults if None)
        """
        if test_prompts is None:
            test_prompts = [
                "jaka jest temperatura w Krakowie",
                "napisz esej o historii Polski",
                "aktualne notowania gieldowe",
                "co to jest rekursja",
                "ile kosztuje iPhone 15 w Polsce",
                "przetlumacz zdanie na angielski",
            ]

        logger.info("Testing predictions:")
        for prompt in test_prompts:
            prediction = classifier.predict(prompt)[0]
            probabilities = classifier.predict_proba(prompt)[0]
            confidence = max(probabilities)
            search_label = "SEARCH" if prediction else "NO SEARCH"
            logger.info(f"  '{prompt}'")
            logger.info(f"    -> {search_label} (confidence: {confidence:.1%})")

    @staticmethod
    def load_config(config_path: Path, model_type: ModelType) -> Tuple[SearchClassifierConfig, Path]:
        """
        Load configuration for specific model from YAML file.

        Args:
            config_path: Path to YAML config file
            model_type: Model type key (e.g., 'tfidf', 'fasttext')

        Returns:
            SearchClassifierConfig instance and output path

        Raises:
            KeyError: If model_type not found in config
        """
        with open(config_path, "r") as file:
            all_configs = yaml.safe_load(file)

        models_config = all_configs.get("models", {})
        if model_type not in models_config:
            available = ", ".join(models_config.keys())
            raise KeyError(f"Model type '{model_type}' not found in config. " f"Available: {available}")

        model_config_dict = models_config[model_type]
        output_directory = Path(all_configs.get("output_directory", "models/"))
        output_path = output_directory / f"{model_type}_classifier.pkl"

        config: SearchClassifierConfig
        if model_type == "tfidf":
            config = TfidfSearchClassifierConfig(**model_config_dict)
        elif model_type == "fasttext":
            config = FastTextSearchClassifierConfig(**model_config_dict)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return config, output_path


if __name__ == "__main__":
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

    args = parser.parse_args()

    logger.info("Web Search Classifier - Training Pipeline")
    logger.info("=" * 60)

    try:
        logger.info(f"Loading configuration from {args.config}")
        config, output_path_from_config = Pipeline.load_config(args.config, args.model_type)
        output_path = Path(args.output_path or output_path_from_config)
    except FileNotFoundError:
        logger.warning(f"Config file {args.config} not found, using defaults")
        output_path = Path(args.output_path or f"models/{args.model_type}_classifier.pkl")
        if args.model_type == "tfidf":
            config = TfidfSearchClassifierConfig()
        elif args.model_type == "fasttext":
            config = FastTextSearchClassifierConfig()
        else:
            logger.error(f"Unknown model type: {args.model_type}")
            logger.info("Available: tfidf, fasttext")
            raise ValueError(f"Unknown model type: {args.model_type}")
    except KeyError as exception:
        logger.error(str(exception))
        raise

    data_path = Path(args.data_path)
    dataset_config = DatasetConfig(
        path=data_path,
        extension=data_path.suffix.lstrip("."),
        prompt_column=args.text_column,
        label_column=args.label_column,
    )
    pipeline = Pipeline(dataset_config=dataset_config)

    try:
        model = pipeline.train_and_save(config=config, output_path=output_path)

        pipeline.test_predictions(model)

        logger.info("To use the model:")
        if args.model_type == "tfidf":
            logger.info(f"  from tfidf import TfidfSearchClassifier")
            logger.info(f"  model = TfidfSearchClassifier.load('{output_path}')")
        else:
            embeddings_path = (
                config.embeddings_path if isinstance(config, FastTextSearchClassifierConfig) else "cc.pl.300.bin"
            )
            logger.info(f"  from ftext import FastTextSearchClassifier")
            logger.info(f"  model = FastTextSearchClassifier.load('{output_path}', '{embeddings_path}')")
        logger.info(f"  result = model.predict('your prompt here')")

    except FileNotFoundError as exception:
        logger.error(f"File not found: {exception}")
    except Exception as exception:
        logger.error(f"Unexpected error: {exception}")
        import traceback

        traceback.print_exc()
