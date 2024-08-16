from dataclasses import dataclass


@dataclass
class TASKS:
    RECONSTRUCTION: str = "reconstruction"
    RECONSTRUCTION_FOR_TEST: str = "reconstruction_for_test"
    RECONSTRUCTION_FOR_TEST_ORIGINAL: str = "reconstruction_for_test_original"
    FORECASTING: str = "forecasting"
    CLASSIFICATION: str = "classification"
    EMBED: str = "embedding"
