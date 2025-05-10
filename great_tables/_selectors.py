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
def _get_bool_list_every_n_row(n_rows: int, n: int, offset: int, is_revert: bool) -> list[bool]:
    if is_revert:
        return [not ((i % n) == offset) for i in range(n_rows)]
    return [(i % n) == offset for i in range(n_rows)]


@cache
def _get_bool_list_first_n_row(n_rows: int, n: int) -> list[bool]:
    return [i <= n for i in range(1, n_rows + 1)]


@cache
def _get_bool_list_last_n_row(n_rows: int, n: int) -> list[bool]:
    return [i > (n_rows - n) for i in range(1, n_rows + 1)]


class GTSelector: ...


class GTRowSelector(GTSelector): ...


class GTColumnSelector(GTSelector): ...


# === every_n_row ===
@dataclass
class every_n_row(GTRowSelector):
    """
    Utility function to select rows based on their index.

    This function can be passed to the `row=` parameter in **Great Tables** when selecting rows.
    Users can specify the number of groups via `n=` and retrieve rows from a desired group using
    `offset=`.

    Parameters
    ----------
    n
        The number of groups into which the rows are divided.
    offset
        The group index to select rows from.

    -------
    every_n_row
        An instance of `every_n_row` that can be used by **Great Tables** to produce a boolean Series,
        indicating which rows should be selected.

    Examples
    --------
    Suppose we want to style the background of a table using the `exibble` dataset,
    alternating between `lightgray` and `darkgray`. We can achieve this with two calls to `every_n_row`:
    ```{python}
    from great_tables import GT, every_n_row, loc, style
    from great_tables.data import exibble

    (
        GT(exibble)
        .tab_style(style=style.fill("lightgray"), locations=loc.body(rows=every_n_row(2)))
        .tab_style(style=style.fill("darkgray"), locations=loc.body(rows=every_n_row(2, 1)))
    )
    ```
    """

    n: int
    offset: int = 0
    is_revert: bool = False

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
        return self._call_every_n_row(data, self.is_revert)

    def __invert__(self) -> "every_n_row":
        cls = type(self)
        return cls(self.n, self.offset, ~self.is_revert)

    def _call_every_n_row(
        self, data: DataFrameLike, is_revert
    ) -> PdSeries | PlSeries | PyArrowArray:
        return _every_n_row(data, self.n, self.offset, is_revert)


@singledispatch
def _every_n_row(data: DataFrameLike, n: int, offset: int) -> DataFrameLike:
    _raise_not_implemented(data)


@_every_n_row.register(PdDataFrame)
def _(data: PdDataFrame, n: int, offset: int, is_revert: bool) -> PdSeries:
    import pandas as pd

    n_rows = data.shape[0]
    _check_n_lt_nrow(n_rows, n)
    bool_list = _get_bool_list_every_n_row(n_rows, n, offset, is_revert)
    return pd.Series(bool_list)


@_every_n_row.register(PlDataFrame)
def _(data: PlDataFrame, n: int, offset: int, is_revert: bool) -> PlSeries:
    import polars as pl

    n_rows = data.height
    _check_n_lt_nrow(n_rows, n)
    bool_list = _get_bool_list_every_n_row(n_rows, n, offset, is_revert)
    return pl.Series(bool_list)


@_every_n_row.register(PyArrowTable)
def _(data: PyArrowTable, n: int, offset: int, is_revert: bool) -> PyArrowArray:
    import pyarrow as pa

    n_rows = data.num_rows
    _check_n_lt_nrow(n_rows, n)
    bool_list = _get_bool_list_every_n_row(n_rows, n, offset, is_revert)
    return pa.array(bool_list, type=pa.bool_())


# === first_n_row ===
@dataclass
class first_n_row(GTRowSelector):
    """
    Utility function to select the first few rows based on their index.

    This function can be passed to the `row=` parameter in **Great Tables** when selecting rows.
    Users can specify the number of rows to select from the beginning.

    Parameters
    ----------
    n
        The number of rows to select starting from the beginning.

    -------
    first_n_row
        An instance of `first_n_row` that can be used by **Great Tables** to produce a boolean Series,
        indicating which rows should be selected.

    Examples
    --------
    Suppose we want to style the background of the first three rows with `darkgray` using the
    `exibble` dataset. We can do the following:
    ```{python}
    from great_tables import GT, first_n_row, style, loc
    from great_tables.data import exibble

    (
        GT(exibble)
        .tab_style(style=style.fill("darkgray"), locations=loc.body(rows=first_n_row(3)))
    )
    """

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
    """
    Utility function to select the last few rows based on their index.

    This function can be passed to the `row=` parameter in **Great Tables** when selecting rows.
    Users can specify the number of rows to select from the end.

    Parameters
    ----------
    n
        The number of rows to select starting from the end.

    -------
    last_n_row
        An instance of `last_n_row` that can be used by **Great Tables** to produce a boolean Series,
        indicating which rows should be selected.

    Examples
    --------
    Suppose we want to style the background of the last three rows with `darkgray` using the
    `exibble` dataset. We can do the following:
    ```{python}
    from great_tables import GT, last_n_row, style, loc
    from great_tables.data import exibble

    (
        GT(exibble)
        .tab_style(style=style.fill("darkgray"), locations=loc.body(rows=last_n_row(3)))
    )
    """

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
