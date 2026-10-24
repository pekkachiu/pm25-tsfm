"""Output locations per test year: TEST_YEAR keeps the plain paths, others get a suffix."""

import argparse

from pm25.config import FINETUNE_SPLITS, TEST_YEAR, TEST_YEARS


def for_year(base: str, test_year: int) -> str:
    if test_year not in TEST_YEARS:
        raise ValueError(f"test year {test_year} not configured; choose from {TEST_YEARS}")
    return base if test_year == TEST_YEAR else f"{base}_{test_year}"


def finetune_split(test_year: int) -> tuple[tuple[str, str], tuple[str, str]]:
    """(train, valid) date windows for fine-tuning, both before the test year."""
    for_year("", test_year)  # validates the year
    return FINETUNE_SPLITS[test_year]


def parse_test_year(argv: list[str] | None = None, description: str | None = None) -> int:
    """`--test-year` for the evaluation entry points (defaults to TEST_YEAR)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--test-year", type=int, choices=TEST_YEARS, default=TEST_YEAR)
    return parser.parse_args(argv).test_year
