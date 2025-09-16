import pandas as pd
from src.data.dataset import parse_history_string


def test_parse_history_string():
    assert parse_history_string("1 2 3") == [1, 2, 3]
    assert parse_history_string("") == []
    assert parse_history_string(None) == []


