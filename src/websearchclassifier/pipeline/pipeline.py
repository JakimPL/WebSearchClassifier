from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import yaml

from websearchclassifier.config import (
    BaselineType,
    BaselineTypeLike,
    ClassifierType,
    ClassifierTypeLike,
    DatasetConfig,
    WebSearchClassifierConfig,
    load_model_config,
)
from websearchclassifier.dataset import Dataset, Prompt, Prompts
from websearchclassifier.model import WebSearchClassifier
from websearchclassifier.utils import Kwargs, logger


class Pipeline:
    """
    Training pipeline for web search classifiers.

    Handles the complete workflow: data loading, model training, evaluation,
    and persistence. Supports multiple classifier types with unified interface.
    """

    def __init__(self, dataset_config: DatasetConfig) -> None:
        """
        Initialize the pipeline.

        Args:
            dataset_config: Configuration object for the dataset.
        """
        self.dataset_config = dataset_config
        self.dataset: Optional[Dataset] = None

    def load_data(self) -> Dataset:
        """
        Load and validate training data using Dataset class.

        Returns:
            Dataset instance with prompts and labels.

        Raises:
            FileNotFoundError: If data file doesn't exist.
            ValueError: If required columns are missing.
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

    @staticmethod
    def _load_yaml_config(config_path: Path) -> Kwargs:
        """
        Load YAML configuration file.

        Args:
            config_path: Path to YAML config file.

        Returns:
            Dictionary with configuration data.
        """
        with open(config_path, "r", encoding="utf-8") as file:
            config_dict: Kwargs = yaml.safe_load(file)

        return config_dict

    @staticmethod
    def _get_output_path(
        all_configs: Kwargs,
        baseline_type: BaselineTypeLike,
    ) -> Path:
        """
        Extract and construct output path from config.

        Args:
            all_configs: Full configuration dictionary.
            baseline_type: Baseline model type key.

        Returns:
            Path where the trained model will be saved.
        """
        output_directory = Path(all_configs.get("output_directory", "models/"))
        return output_directory / f"{baseline_type}_classifier.pkl"

    @staticmethod
    def _get_baseline_config(
        all_configs: Kwargs,
        baseline_type: BaselineTypeLike,
    ) -> Kwargs:
        """
        Extract baseline configuration from config dictionary.

        Args:
            all_configs: Full configuration dictionary.
            baseline_type: Baseline model type key.

        Returns:
            Baseline configuration dictionary.

        Raises:
            KeyError: If baseline_type not found in config.
        """
        baselines_dict: Kwargs = all_configs.get("baselines", {})
        if baseline_type not in baselines_dict:
            available = ", ".join(baselines_dict.keys())
            raise KeyError(f"Baseline type '{baseline_type}' not found in config. " f"Available: {available}")

        baseline_dict: Kwargs = baselines_dict[baseline_type]
        return baseline_dict

    @staticmethod
    def _get_classifier_config(
        all_configs: Kwargs,
        classifier_type: ClassifierTypeLike,
    ) -> Kwargs:
        """
        Extract classifier configuration from config dictionary.

        Args:
            all_configs: Full configuration dictionary.
            classifier_type: Classifier type key.

        Returns:
            Classifier configuration dictionary.

        Raises:
            KeyError: If classifier_type not found in config.
        """
        classifiers_dict: Kwargs = all_configs.get("classifiers", {})
        if classifier_type not in classifiers_dict:
            available = ", ".join(classifiers_dict.keys())
            raise KeyError(f"Classifier type '{classifier_type}' not found in config. " f"Available: {available}")

        classifier_dict: Kwargs = classifiers_dict[classifier_type]
        return classifier_dict

    def train(self, config: WebSearchClassifierConfig) -> WebSearchClassifier[Any, Any]:
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

        wbc = WebSearchClassifier[Any, Any](config=config)
        logger.info("Training [%s, %s]...", wbc.baseline.class_name, wbc.wrapper.class_name)
        return wbc.fit(self.dataset)

    def train_and_save(
        self,
        config: WebSearchClassifierConfig,
        output_path: Path,
    ) -> WebSearchClassifier[Any, Any]:
        """
        Train a classifier and save it to disk.

        This is the main pipeline method combining data loading, training,
        and model persistence.

        Args:
            config: Configuration object for the classifier.
            output_path: Path to save the trained model.

        Returns:
            Trained classifier instance.
        """
        if self.dataset is None:
            self.load_data()

        assert self.dataset is not None
        wbc: WebSearchClassifier[Any, Any] = self.train(config)

        logger.info("Saving model to %s...", output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wbc.save(output_path)

        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info("Model saved to: %s", output_path)
        logger.info("Model type: %s", wbc.__class__.__name__)
        logger.info("Training samples: %s", len(self.dataset.prompts))
        return wbc

    def test_predictions(
        self,
        wbc: WebSearchClassifier[Any, Any],
        test_prompts: Optional[Union[Prompt, Prompts]] = None,
    ) -> None:
        """
        Test classifier on example prompts and display results.

        Args:
            classifier: Trained classifier to test.
            test_prompts: List of prompts to test (uses defaults if None).
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
        prompts = wbc.normalize_prompts(test_prompts)
        for prompt in prompts:
            probabilities = wbc.predict_proba(prompt)[0]
            prediction = wbc.predict(prompt)[0]
            confidence = max(probabilities)
            search_label = "   SEARCH" if prediction else "NO SEARCH"
            lines.append(f"  '{prompt}'")
            lines.append(f"    -> {search_label} (confidence: {confidence:.1%})")

        logger.info("\n".join(lines))

    @staticmethod
    def load_config(
        baseline: BaselineTypeLike,
        classifier: ClassifierTypeLike,
        config_path: Path = Path("config.yaml"),
    ) -> Tuple[WebSearchClassifierConfig, Path]:
        """
        Load configuration for specific model from YAML file.

        Args:
            config_path: Path to YAML config file.
            baseline: Baseline model type key (e.g. `tfidf`, `fasttext`, `herbert`).
            classifier: Classifier model type key (e.g. `logistic_regression`, `svm`).

        Returns:
            WebSearchClassifierConfig instance and output path.

        Raises:
            KeyError: If baseline_type not found in config.
        """
        all_configs = Pipeline._load_yaml_config(config_path)
        output_path = Pipeline._get_output_path(all_configs, baseline)
        baseline_config_dict = Pipeline._get_baseline_config(all_configs, baseline)
        classifier_config_dict = Pipeline._get_classifier_config(all_configs, classifier)

        baseline_type_enum = BaselineType(baseline)
        classifier_type_enum = ClassifierType(classifier)

        config = load_model_config(
            baseline_type=baseline_type_enum,
            classifier_type=classifier_type_enum,
            baseline_config_dict=baseline_config_dict,
            classifier_config_dict=classifier_config_dict,
        )

        return config, output_path
