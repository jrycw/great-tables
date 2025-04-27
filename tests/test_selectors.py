import pandas as pd
import pyarrow as pa
import polars as pl
import pytest

from great_tables import GT, every_n_row
from great_tables._tbl_data import DataFrameLike

params_frames = [
    pytest.param(pd.DataFrame, id="pandas"),
    pytest.param(pl.DataFrame, id="polars"),
    # pytest.param(pa.table, id="arrow"),
]
params_series = [
    pytest.param(pd.Series, id="pandas"),
    pytest.param(pl.Series, id="polars"),
    pytest.param(pa.array, id="arrow"),
    # pytest.param(lambda a: pa.chunked_array([a]), id="arrow-chunked"),
]


@pytest.fixture(params=params_frames, scope="function")
def df(request) -> pd.DataFrame:
    return request.param({"col1": [0, 1, 2, 3, 4]})


@pytest.mark.parametrize(
    "row_selector, expected",
    [
        (every_n_row(1), [0, 1, 2, 3, 4]),
        (every_n_row(2), [0, 2, 4]),
        (every_n_row(2, 1), [1, 3]),
        (every_n_row(3), [0, 3]),
        (every_n_row(3, 1), [1, 4]),
        (every_n_row(4), [0, 4]),
    ],
)
def test_every_n_row(df: DataFrameLike, row_selector, expected):
    res = GT(df).fmt_integer(columns="col1", rows=row_selector)
    assert len(res._formats) == 1
    assert res._formats[0].cells.rows == expected


def test_every_n_row_raise(df: DataFrameLike):
    with pytest.raises(ValueError) as exc_info:
        res = GT(df).fmt_integer(columns="col1", rows=every_n_row(-1))
    assert "`n` must be a positive integer greater than 0." in exc_info.value.args[0]

    with pytest.raises(ValueError) as exc_info:
        res = GT(df).fmt_integer(columns="col1", rows=every_n_row(0))
    assert "`n` must be a positive integer greater than 0." in exc_info.value.args[0]

    with pytest.raises(ValueError) as exc_info:
        res = GT(df).fmt_integer(columns="col1", rows=every_n_row(2, -1))
    assert "`offset` must not be a negative integer." in exc_info.value.args[0]

    with pytest.raises(ValueError) as exc_info:
        res = GT(df).fmt_integer(columns="col1", rows=every_n_row(2, 2))
    assert "`offset` must be less than `n_rows`." in exc_info.value.args[0]

    with pytest.raises(ValueError) as exc_info:
        res = GT(df).fmt_integer(columns="col1", rows=every_n_row(2, 3))
    assert "`offset` must be less than `n_rows`." in exc_info.value.args[0]

    with pytest.raises(ValueError) as exc_info:
        res = GT(df).fmt_integer(columns="col1", rows=every_n_row(5))
    assert "`n` must be less than `n_rows`." in exc_info.value.args[0]

    with pytest.raises(ValueError) as exc_info:
        res = GT(df).fmt_integer(columns="col1", rows=every_n_row(100))
    assert "`n` must be less than `n_rows`." in exc_info.value.args[0]
