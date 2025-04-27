from dataclasses import dataclass
from functools import cache, singledispatch

from ._tbl_data import (
    DataFrameLike,
    PdDataFrame,
    PdSeries,
    PlDataFrame,
    PlSeries,
    PyArrowArray,
    PyArrowTable,
    _raise_not_implemented,
)


class GTSelector: ...


class GTRowSelector(GTSelector): ...


class GTColumnSelector(GTSelector): ...


@dataclass
class every_n_row(GTRowSelector):
    n: int
    offset: int = 0

    def __post_init__(self):
        self._check()

    def _check(self):
        if self.n <= 0:
            raise ValueError("`n` must be a positive integer greater than 0.")
        if self.offset < 0:
            raise ValueError("`offset` must not be a negative integer.")
        if self.offset > (self.n - 1):
            raise ValueError("`offset` must be less than `n_rows`.")

    def __call__(self, data: DataFrameLike):
        return _every_n_row(data, self.n, self.offset)


def _check_every_n_row(n_rows: int, n: int) -> None:
    if n >= n_rows:
        raise ValueError("`n` must be less than `n_rows`.")


@cache
def _get_bool_list(n_rows: int, n: int, offset: int) -> list[bool]:
    return [True if ((i % n) == offset) else False for i in range(n_rows)]


@singledispatch
def _every_n_row(data: DataFrameLike, n: int, offset: int) -> DataFrameLike:
    _raise_not_implemented(data)


@_every_n_row.register(PdDataFrame)
def _(data: PdDataFrame, n: int, offset: int) -> PdSeries:
    import pandas as pd

    n_rows = data.shape[0]
    _check_every_n_row(n_rows, n)
    bool_list = _get_bool_list(n_rows, n, offset)
    return pd.Series(bool_list)


@_every_n_row.register(PlDataFrame)
def _(data: PlDataFrame, n: int, offset: int) -> PlSeries:
    import polars as pl

    n_rows = data.height
    _check_every_n_row(n_rows, n)
    bool_list = _get_bool_list(n_rows, n, offset)
    return pl.Series(bool_list)


@_every_n_row.register(PyArrowTable)
def _(data: PyArrowTable, n: int, offset: int) -> PyArrowArray:
    import pyarrow as pa

    n_rows = data.num_rows
    _check_every_n_row(n_rows, n)
    bool_list = _get_bool_list(n_rows, n, offset)
    return pa.array(bool_list, type=pa.bool_())
