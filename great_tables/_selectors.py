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


def _check_n_lt_nrow(n_rows: int, n: int) -> None:
    if n >= n_rows:
        raise ValueError("`n` must be less than `n_rows`.")


def _check_n_le_nrow(n_rows: int, n: int) -> None:
    if n > n_rows:
        raise ValueError("`n` must be less than or equal to `n_rows`.")


@cache
def _get_bool_list_every_n_row(n_rows: int, n: int, offset: int) -> list[bool]:
    return [True if ((i % n) == offset) else False for i in range(n_rows)]


@cache
def _get_bool_list_first_n_row(n_rows: int, n: int) -> list[bool]:
    return [True if i <= n else False for i in range(1, n_rows + 1)]


@cache
def _get_bool_list_last_n_row(n_rows: int, n: int) -> list[bool]:
    return [True if i > (n_rows - n) else False for i in range(1, n_rows + 1)]


class GTSelector: ...


class GTRowSelector(GTSelector): ...


class GTColumnSelector(GTSelector): ...


# === every_n_row ===
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

    def __call__(self, data: DataFrameLike) -> PdSeries | PlSeries | PyArrowArray:
        return _every_n_row(data, self.n, self.offset)


@singledispatch
def _every_n_row(data: DataFrameLike, n: int, offset: int) -> DataFrameLike:
    _raise_not_implemented(data)


@_every_n_row.register(PdDataFrame)
def _(data: PdDataFrame, n: int, offset: int) -> PdSeries:
    import pandas as pd

    n_rows = data.shape[0]
    _check_n_lt_nrow(n_rows, n)
    bool_list = _get_bool_list_every_n_row(n_rows, n, offset)
    return pd.Series(bool_list)


@_every_n_row.register(PlDataFrame)
def _(data: PlDataFrame, n: int, offset: int) -> PlSeries:
    import polars as pl

    n_rows = data.height
    _check_n_lt_nrow(n_rows, n)
    bool_list = _get_bool_list_every_n_row(n_rows, n, offset)
    return pl.Series(bool_list)


@_every_n_row.register(PyArrowTable)
def _(data: PyArrowTable, n: int, offset: int) -> PyArrowArray:
    import pyarrow as pa

    n_rows = data.num_rows
    _check_n_lt_nrow(n_rows, n)
    bool_list = _get_bool_list_every_n_row(n_rows, n, offset)
    return pa.array(bool_list, type=pa.bool_())


# === first_n_row ===
@dataclass
class first_n_row(GTRowSelector):
    n: int

    def __post_init__(self):
        self._check()

    def _check(self):
        if self.n <= 0:
            raise ValueError("`n` must be a positive integer greater than 0.")

    def __call__(self, data: DataFrameLike) -> PdSeries | PlSeries | PyArrowArray:
        return _first_n_row(data, self.n)


@singledispatch
def _first_n_row(data: DataFrameLike, n: int) -> DataFrameLike:
    _raise_not_implemented(data)


@_first_n_row.register(PdDataFrame)
def _(data: PdDataFrame, n: int) -> PdSeries:
    import pandas as pd

    n_rows = data.shape[0]
    _check_n_le_nrow(n_rows, n)
    bool_list = _get_bool_list_first_n_row(n_rows, n)
    return pd.Series(bool_list)


@_first_n_row.register(PlDataFrame)
def _(data: PlDataFrame, n: int) -> PlSeries:
    import polars as pl

    n_rows = data.height
    _check_n_le_nrow(n_rows, n)
    bool_list = _get_bool_list_first_n_row(n_rows, n)
    return pl.Series(bool_list)


@_first_n_row.register(PyArrowTable)
def _(data: PyArrowTable, n: int) -> PyArrowArray:
    import pyarrow as pa

    n_rows = data.num_rows
    _check_n_le_nrow(n_rows, n)
    bool_list = _get_bool_list_first_n_row(n_rows, n)
    return pa.array(bool_list, type=pa.bool_())


# === last_n_row ===
@dataclass
class last_n_row(GTRowSelector):
    n: int

    def __post_init__(self):
        self._check()

    def _check(self):
        if self.n <= 0:
            raise ValueError("`n` must be a positive integer greater than 0.")

    def __call__(self, data: DataFrameLike) -> PdSeries | PlSeries | PyArrowArray:
        return _last_n_row(data, self.n)


@singledispatch
def _last_n_row(data: DataFrameLike, n: int) -> DataFrameLike:
    _raise_not_implemented(data)


@_last_n_row.register(PdDataFrame)
def _(data: PdDataFrame, n: int) -> PdSeries:
    import pandas as pd

    n_rows = data.shape[0]
    _check_n_le_nrow(n_rows, n)
    bool_list = _get_bool_list_last_n_row(n_rows, n)
    return pd.Series(bool_list)


@_last_n_row.register(PlDataFrame)
def _(data: PlDataFrame, n: int) -> PlSeries:
    import polars as pl

    n_rows = data.height
    _check_n_le_nrow(n_rows, n)
    bool_list = _get_bool_list_last_n_row(n_rows, n)
    return pl.Series(bool_list)


@_last_n_row.register(PyArrowTable)
def _(data: PyArrowTable, n: int) -> PyArrowArray:
    import pyarrow as pa

    n_rows = data.num_rows
    _check_n_le_nrow(n_rows, n)
    bool_list = _get_bool_list_last_n_row(n_rows, n)
    return pa.array(bool_list, type=pa.bool_())
