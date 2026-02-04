from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import yaml

from websearchclassifier.config import (
    ClassifierType,
    ClassifierTypeLike,
    DatasetConfig,
    FastTextSearchClassifierConfig,
    HerBERTSearchClassifierConfig,
    ModelType,
    ModelTypeLike,
    SearchClassifierConfig,
    TFIDFSearchClassifierConfig,
)
from websearchclassifier.dataset import Dataset, Prompts
from websearchclassifier.model import (
    FastTextSearchClassifier,
    HerBERTSearchClassifier,
    SearchClassifier,
    TFIDFSearchClassifier,
    get_model_config_class,
)
from websearchclassifier.utils import logger


class Pipeline:
    """
    Training pipeline for web search classifiers.

    Handles the complete workflow: data loading, model training, evaluation,
    and persistence. Supports multiple classifier types with unified interface.

    Example:
        >>> pipeline = Pipeline(data_path="data/train.csv")
        >>> model = pipeline.train_and_save(
        ...     classifier_class=TFIDFSearchClassifier,
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
        logger.info("Loading dataset from: '%s'...", str(self.dataset_config.dataset_path))
        self.dataset = Dataset.load(self.dataset_config)

        num_positive = self.dataset.positive
        num_negative = self.dataset.negative

        logger.info(
            "Loaded %s examples:\n"  #
            "    - Needs search: %s (%.1f%%)\n"  #
            "    - No search:    %s (%.1f%%)",  #
            len(self.dataset.prompts),
            num_positive,
            num_positive / self.dataset.size * 100,
            num_negative,
            num_negative / self.dataset.size * 100,
        )

        return self.dataset

    def train(self, config: SearchClassifierConfig) -> SearchClassifier[Any]:
        """
        Train a classifier on loaded data.

        Args:
            config: Configuration object for the classifier.

        Returns:
            Trained classifier instance.

        Raises:
            RuntimeError: If data hasn't been loaded yet.
        """
        if self.dataset is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        classifier: SearchClassifier[Any]
        match config:
            case FastTextSearchClassifierConfig():
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
            case TFIDFSearchClassifierConfig():
                logger.info("Initializing TFIDFSearchClassifier...")
                classifier = TFIDFSearchClassifier(config=config)
            case HerBERTSearchClassifierConfig():
                logger.info("Initializing HerBERTSearchClassifier...")
                classifier = HerBERTSearchClassifier(config=config)
            case _:
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
            config: Configuration object for the classifier.
            output_path: Path to save the trained model.

        Returns:
            Trained classifier instance.

        Example:
            >>> dataset_config = DatasetConfig(path=Path("data/train.csv"))
            >>> pipeline = Pipeline(dataset_config)
            >>> config = TFIDFSearchClassifierConfig(max_features=3000)
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
        test_prompts: Optional[Union[str, Prompts]] = None,
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
                "gdzie znajduje się Warszawa?",
                "wyszukaj mi proszę",
                "opowiesz coś o sobie?",
                "gdzie znajduje się Warszawa?",
                "wyszukaj mi proszę przepis na owsiankę",
            ]

        lines = ["Testing predictions:"]
        prompts = classifier.normalize_prompts(test_prompts)
        for prompt in prompts:
            prediction = classifier.predict(prompt)[0]
            probabilities = classifier.predict_proba(prompt)[0]
            confidence = max(probabilities)
            search_label = "   SEARCH" if prediction else "NO SEARCH"
            lines.append(f"  '{prompt}'")
            lines.append(f"    -> {search_label} (confidence: {confidence:.1%})")

        logger.info("\n".join(lines))

    @staticmethod
    def load_config(
        config_path: Path,
        model_type: ModelTypeLike,
        classifier_type: ClassifierTypeLike,
    ) -> Tuple[SearchClassifierConfig, Path]:
        """
        Load configuration for specific model from YAML file.

        Args:
            config_path: Path to YAML config file
            model_type: Model type key (e.g. `tfidf`, `fasttext`, `herbert`)
            classifier_type: Classifier type key (e.g. `logistic_regression`, `svm`)

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

        classifier_config_dict = all_configs.get("classifiers", {})
        if classifier_type not in classifier_config_dict:
            available = ", ".join(classifier_config_dict.keys())
            raise KeyError(f"Classifier type '{classifier_type}' not found in config. " f"Available: {available}")

        classifier_type = ClassifierType(classifier_type)

        config: SearchClassifierConfig
        model_type = ModelType(model_type)
        config = get_model_config_class(model_type)(
            **model_config_dict,
            classifier_config=classifier_config_dict[classifier_type],
        )
        return config, output_path
