"""
FastText-based binary classifier for web search decision.
Uses pre-trained Polish FastText embeddings for semantic understanding.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, List, Optional, Self, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from websearchclassifier.config import FastTextSearchClassifierConfig
from websearchclassifier.model.base import SearchClassifier


class FastTextSearchClassifier(SearchClassifier[FastTextSearchClassifierConfig]):
    """
    Binary classifier using FastText embeddings for Polish prompts.

    Uses pre-trained Polish FastText word embeddings with averaging strategy
    to create sentence representations, then trains a classifier on top.

    Better than TF-IDF for semantic understanding but requires downloading
    pre-trained embeddings (~1-6GB depending on version).

    Example:
        >>> classifier = FastTextSearchClassifier()
        >>> classifier.load_embeddings("cc.pl.300.bin")
        >>>
        >>> train_data = [
        ...     ("jaka jest dzisiaj pogoda w Warszawie", True),
        ...     ("oblicz pierwiastek z 144", False),
        ... ]
        >>> prompts, labels = zip(*train_data)
        >>> classifier.fit(prompts, labels)
        >>>
        >>> classifier.predict(["jaka jest aktualna cena bitcoina"])
    """

    def __init__(
        self,
        config: FastTextSearchClassifierConfig,
    ):
        """
        Initialize the classifier.

        Args:
            config: Configuration object with all parameters
        """
        super().__init__(config)

        self.fasttext_model = None
        self._is_embeddings_loaded = False
        self._is_fitted = False

        if config.classifier_type == "logistic":
            self.classifier = LogisticRegression(
                C=config.regularization_strength,
                random_state=config.random_state,
                max_iter=1000,
                solver="liblinear",
            )
        elif config.classifier_type == "svm":
            self.classifier = SVC(
                C=config.regularization_strength,
                kernel="rbf",
                random_state=config.random_state,
                probability=True,
            )
        else:
            raise ValueError(f"Unknown classifier_type: {config.classifier_type}. Use 'logistic' or 'svm'")

    def load_embeddings(self, model_path: Path) -> FastTextSearchClassifier:
        """
        Load pre-trained FastText embeddings.

        Download Polish FastText model from:
        https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.bin.gz

        Args:
            model_path: Path to FastText .bin file

        Returns:
            self (for method chaining)
        """
        import fasttext

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"FastText model not found at: {model_path}\n"
                f"Download Polish model:\n"
                f"wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.bin.gz\n"
                f"gunzip cc.pl.300.bin.gz"
            )

        print(f"Loading FastText embeddings from {model_path}...")
        self.fasttext_model = fasttext.load_model(str(model_path))
        self._is_embeddings_loaded = True
        print(f"✓ Loaded embeddings (dim={self.fasttext_model.get_dimension()})")

        return self

    def load_embeddings_gensim(self, model_path: Path) -> FastTextSearchClassifier:
        """
        Load pre-trained FastText embeddings using Gensim (alternative method).

        Can load smaller/custom FastText models trained with Gensim.

        Args:
            model_path: Path to Gensim FastText model

        Returns:
            self (for method chaining)
        """
        try:
            from gensim.models import FastText
        except ImportError:
            raise ImportError("gensim library not installed. Install with: pip install gensim")

        print(f"Loading FastText embeddings (Gensim) from {model_path}...")
        self.fasttext_model = FastText.load(model_path)
        self._is_embeddings_loaded = True
        self._use_gensim = True
        print(f"✓ Loaded embeddings (dim={self.fasttext_model.wv.vector_size})")

        return self

    def _encode_text(self, text: str) -> np.ndarray:
        """
        Convert text to vector representation using FastText embeddings.

        Uses mean pooling: averages word vectors to get sentence vector.

        Args:
            text: Input text

        Returns:
            Vector representation of shape (embedding_dim,)
        """
        self._check_embeddings_loaded()

        words = text.lower().split()

        if len(words) == 0:
            return np.zeros(self.config.embeddig_dim)

        if hasattr(self.fasttext_model, "get_word_vector"):
            vectors = [self.fasttext_model.get_word_vector(word) for word in words]
        else:
            vectors = [self.fasttext_model.wv[word] for word in words if word in self.fasttext_model.wv]
            if len(vectors) == 0:
                return np.zeros(self.embedding_dim)

        return np.mean(vectors, axis=0)

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts to vector representations.

        Args:
            texts: List of input texts

        Returns:
            Array of shape (n_texts, embedding_dim)
        """
        return np.array([self._encode_text(text) for text in texts])

    def fit(self, prompts: List[str], labels: List[bool]) -> FastTextSearchClassifier:
        """
        Train the classifier on labeled data.

        Args:
            prompts: List of prompt strings
            labels: List of boolean labels (True = needs search, False = no search)

        Returns:
            self (for method chaining)
        """
        self._check_embeddings_loaded()
        self._validate_training_data(prompts, labels)

        print(f"Encoding {len(prompts)} prompts...")
        features = self._encode_batch(prompts)
        labels_array = np.array(labels, dtype=int)

        print(f"Training {self.config.classifier_type} classifier...")
        self.classifier.fit(features, labels_array)
        self._is_fitted = True
        print("✓ Model trained")

        return self

    def predict(self, prompts: Union[str, List[str]]) -> np.ndarray:
        """
        Predict whether prompts need web search.

        Args:
            prompts: Single prompt string or list of prompts

        Returns:
            Boolean array (True = needs search, False = no search)
        """
        self._check_is_fitted()
        prompts = self._normalize_input(prompts)

        features = self._encode_batch(prompts)
        predictions = self.classifier.predict(features)
        return predictions.astype(bool)

    def predict_proba(self, prompts: Union[str, List[str]]) -> np.ndarray:
        """
        Predict probability of needing web search.

        Args:
            prompts: Single prompt string or list of prompts

        Returns:
            Array of shape (n_samples, 2) with probabilities [no_search, needs_search]
        """
        self._check_is_fitted()
        unified_prompts: List[str] = self._normalize_input(prompts)

        features = self._encode_batch(unified_prompts)
        proba: np.ndarray = self.classifier.predict_proba(features)
        return proba

    def save(self, filepath: Path) -> None:
        """
        Save the trained classifier (without embeddings).

        Note: This saves only the classifier, not the FastText embeddings.
        You'll need to load embeddings separately when loading the model.

        Args:
            filepath: Path to save the classifier (e.g., 'model.pkl')
        """
        self._check_is_fitted()

        save_dict = {
            "config": self.config,
            "classifier": self.classifier,
            "_is_fitted": self._is_fitted,
        }

        with open(filepath, "wb") as file:
            pickle.dump(save_dict, file)

        print(f"✓ Classifier saved to {filepath}")
        print("⚠ Remember: FastText embeddings NOT included. Load them separately.")

    @classmethod
    def load(cls, filepath: Path, embeddings_path: Optional[Path] = None, **kwargs: Any) -> Self:
        """
        Load a trained classifier and embeddings.

        Args:
            filepath: Path to the saved classifier
            embeddings_path: Path to FastText embeddings (.bin file)
            **kwargs: Additional arguments

        Returns:
            Loaded FastTextSearchClassifier instance
        """
        with open(filepath, "rb") as file:
            save_dict = pickle.load(file)

        model = cls(config=save_dict["config"])
        model.classifier = save_dict["classifier"]
        model._is_fitted = save_dict["_is_fitted"]

        if embeddings_path:
            model.load_embeddings(embeddings_path)
        else:
            print("⚠ Warning: Embeddings not loaded. Call load_embeddings() before using the model.")

        print(f"✓ Model loaded from {filepath}")
        return model

    def _check_embeddings_loaded(self) -> None:
        """
        Check if embeddings have been loaded.

        Raises:
            RuntimeError: If embeddings haven't been loaded
        """
        if not self._is_embeddings_loaded:
            raise RuntimeError(
                "Embeddings not loaded. Call load_embeddings() first.\n"
                "Download Polish FastText: wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.bin.gz"
            )

    def __repr__(self) -> str:
        return (
            f"FastTextSearchClassifier("
            f"embedding_dim={self.config.embeddig_dim}, "
            f"classifier={self.config.classifier_type}, "
            f"regularization={self.config.regularization_strength}, "
            f"embeddings_loaded={self._is_embeddings_loaded}, "
            f"fitted={self._is_fitted}"
            f")"
        )
