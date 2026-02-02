from enum import StrEnum


class EvaluationMode(StrEnum):
    TRAIN_TEST_SPLIT = "train_test_split"
    CROSS_VALIDATION = "cross_validation"
    HOLDOUT = "holdout"
    BOOTSTRAP = "bootstrap"
