from dataclasses import dataclass
from functools import singledispatch

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


def _get_bool_list(n_rows: int, n: int, offset: int) -> list[bool]:
    if n > n_rows:
        raise ValueError("`n` must not exceed `n_rows`.")
    return [True if ((i % n) == offset) else False for i in range(n_rows)]


@singledispatch
def _every_n_row(data: DataFrameLike, n: int, offset: int) -> DataFrameLike:
    _raise_not_implemented(data)


@_every_n_row.register(PdDataFrame)
def _(data: PdDataFrame, n: int, offset: int) -> PdSeries:
    import pandas as pd

    n_rows = data.shape[0]
    return pd.Series(_get_bool_list(n_rows, n, offset))


@_every_n_row.register(PlDataFrame)
def _(data: PlDataFrame, n: int, offset: int) -> PlSeries:
    import polars as pl

    n_rows = data.height
    return pl.Series(_get_bool_list(n_rows, n, offset))


@_every_n_row.register(PyArrowTable)
def _(data: PyArrowTable, n: int, offset: int) -> PyArrowArray:
    import pyarrow as pa

    n_rows = data.num_rows
    return pa.array(_get_bool_list(n_rows, n, offset), type=pa.bool_())
