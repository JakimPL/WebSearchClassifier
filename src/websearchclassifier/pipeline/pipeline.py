from __future__ import annotations

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
from websearchclassifier.pipeline.types import ModelTypeLike
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
        logger.info("Loading dataset from %s...", str(self.dataset_config.dataset_path))
        self.dataset = Dataset.load(self.dataset_config)

        num_positive = self.dataset.positive
        num_negative = self.dataset.negative

        logger.info("Loaded %s examples", len(self.dataset.prompts))
        logger.info("  - Needs search: %s (%.1f%%)", num_positive, num_positive / self.dataset.size * 100)
        logger.info("  - No search:    %s (%.1f%%)", num_negative, num_negative / self.dataset.size * 100)

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
            logger.info("Initializing FastTextSearchClassifier...")
            classifier = FastTextSearchClassifier(config=config)

            if not config.embeddings_path.exists():
                raise FileNotFoundError(
                    f"FastText embeddings not found at: {config.embeddings_path}\n"
                    f"Download Polish model:\n"
                    f"wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.bin.gz\n"
                    f"gunzip cc.pl.300.bin.gz"
                )
            classifier.load_embeddings(config.embeddings_path)
        elif isinstance(config, TfidfSearchClassifierConfig):
            logger.info("Initializing TfidfSearchClassifier...")
            classifier = TfidfSearchClassifier(config=config)
        else:
            raise ValueError(f"Unknown config type: {type(config)}")

        logger.info("Training %s...", classifier.__class__.__name__)
        classifier.fit(self.dataset)
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

        logger.info("Saving model to %s...", output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        classifier.save(output_path)

        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info("Model saved to: %s", output_path)
        logger.info("Model type: %s", classifier.__class__.__name__)
        logger.info("Training samples: %s", len(self.dataset.prompts))

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

        lines = ["Testing predictions:"]
        for prompt in test_prompts:
            prediction = classifier.predict(prompt)[0]
            probabilities = classifier.predict_proba(prompt)[0]
            confidence = max(probabilities)
            search_label = "SEARCH" if prediction else "NO SEARCH"
            lines.append(f"  '{prompt}'")
            lines.append(f"    -> {search_label} (confidence: {confidence:.1%})")

        logger.info("\n".join(lines))

    @staticmethod
    def load_config(config_path: Path, model_type: ModelTypeLike) -> Tuple[SearchClassifierConfig, Path]:
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
        with open(config_path, "r", encoding="utf-8") as file:
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
