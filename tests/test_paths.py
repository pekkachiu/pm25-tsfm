import pytest

from pm25.paths import finetune_split, for_year


def test_for_year_keeps_primary_year_paths():
    assert for_year("results/tables", 2025) == "results/tables"
    assert for_year("results/tables", 2024) == "results/tables_2024"


def test_for_year_rejects_unconfigured_year():
    with pytest.raises(ValueError, match="2023"):
        for_year("results/tables", 2023)


def test_finetune_split_stays_before_test_year():
    assert finetune_split(2025) == (("2021-01-01", "2023-12-31 23:00"),
                                    ("2024-01-01", "2024-12-31 23:00"))
    train, valid = finetune_split(2024)
    assert train == ("2021-01-01", "2022-12-31 23:00")
    assert valid == ("2023-01-01", "2023-12-31 23:00")


def test_parse_test_year():
    from pm25.paths import parse_test_year

    assert parse_test_year([]) == 2025
    assert parse_test_year(["--test-year", "2024"]) == 2024
    with pytest.raises(SystemExit):
        parse_test_year(["--test-year", "2023"])
