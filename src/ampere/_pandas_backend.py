"""
Pandas/NumPy backend for Ampere — drop-in replacement for arkouda.
All public names mirror the arkouda API used by ampere/__init__.py.
"""
import re as _re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap(arr):
    """Return arr as a _PdArray view (adds .to_ndarray() and arkouda-style .astype())."""
    return np.asarray(arr).view(_PdArray)


# ---------------------------------------------------------------------------
# _PdArray — numpy subclass with arkouda-compatible extras
# ---------------------------------------------------------------------------

class _PdArray(np.ndarray):
    """numpy ndarray subclass that adds arkouda-compatible helper methods."""

    def __new__(cls, input_array, dtype=None):
        return np.asarray(input_array, dtype=dtype).view(cls)

    def __array_finalize__(self, obj):
        pass

    def to_ndarray(self):
        return np.asarray(self)

    # Override so the result stays _PdArray
    def astype(self, dtype, **kwargs):
        return _wrap(np.asarray(self).astype(dtype, **kwargs))


# ---------------------------------------------------------------------------
# PandasStrings — numpy string array with arkouda.Strings API
# ---------------------------------------------------------------------------

class _FullMatchResult:
    def __init__(self, mask):
        self._mask = _wrap(mask)

    def matched(self):
        return self._mask


class PandasStrings:
    """Mimics arkouda.Strings using a numpy str array."""

    def __init__(self, data):
        if isinstance(data, PandasStrings):
            self._data = data._data.copy()
        else:
            self._data = np.asarray(data, dtype=str)

    # ---- arkouda-compatible API ----

    def to_ndarray(self):
        return self._data.copy()

    @property
    def size(self):
        return len(self._data)

    @property
    def dtype(self):
        # Must be != np.float64 and != np.int64 to pass _get_valid_numeric_mask checks
        return str

    def fullmatch(self, pattern):
        matched = np.array(
            [bool(_re.fullmatch(pattern, s)) for s in self._data], dtype=np.bool_
        )
        return _FullMatchResult(matched)

    # ---- operators ----

    def __eq__(self, other):
        if isinstance(other, str):
            return _wrap(self._data == other)
        if isinstance(other, PandasStrings):
            return _wrap(self._data == other._data)
        return NotImplemented

    def __ne__(self, other):
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return eq
        return _wrap(~np.asarray(eq))

    def __getitem__(self, idx):
        result = self._data[idx]
        if isinstance(result, np.ndarray):
            return PandasStrings(result)
        return str(result)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"PandasStrings({self._data!r})"


# ---------------------------------------------------------------------------
# PandasGroupBy — mimics arkouda.GroupBy
# ---------------------------------------------------------------------------

class PandasGroupBy:
    """
    Mimics arkouda.GroupBy.

    Uses vectorised numpy for aggregate('sum') and aggregate('first') instead
    of the O(n × n_unique) Python loop, so it stays fast on HPC-sized traces.
    """

    def __init__(self, keys):
        raw = keys.to_ndarray() if isinstance(keys, PandasStrings) else np.asarray(keys)
        self._keys = raw
        self._unique_keys, self._inverse = np.unique(raw, return_inverse=True)
        self._is_str = raw.dtype.kind in ('U', 'S', 'O')

    def aggregate(self, values, func: str):
        is_str = isinstance(values, PandasStrings)
        vals = values.to_ndarray() if is_str else np.asarray(values)

        n = len(self._unique_keys)
        if n == 0:
            empty = np.array([], dtype=vals.dtype)
            keys_out = PandasStrings(self._unique_keys) if self._is_str else _wrap(self._unique_keys)
            return keys_out, (PandasStrings(empty) if is_str else _wrap(empty))

        if func == 'sum' and not is_str and np.issubdtype(vals.dtype, np.integer):
            # Fast path: integer keys in a known range → bincount (O(n), fully vectorised).
            # This is the hot path hit by _compute_coverage_ak on every attribution call.
            max_key = int(self._unique_keys[-1])  # unique_keys is sorted by np.unique
            sums = np.bincount(self._inverse, weights=vals.astype(np.float64),
                               minlength=n).astype(vals.dtype)
            out = sums  # one entry per unique key, already in sorted order

        elif func in ('sum', 'first') and not is_str:
            # General numeric path: sort by group, then use reduceat (O(n log n)).
            order = np.argsort(self._inverse, kind='stable')
            sorted_vals = vals[order]
            sorted_inv  = self._inverse[order]
            boundaries  = np.flatnonzero(np.diff(sorted_inv))
            group_starts = np.concatenate([[0], boundaries + 1])  # length == n

            if func == 'sum':
                out = np.add.reduceat(sorted_vals, group_starts).astype(vals.dtype)
            else:  # 'first'
                out = sorted_vals[group_starts]

        elif func == 'first':
            # String or other non-numeric 'first': still needs sort, O(n log n).
            order = np.argsort(self._inverse, kind='stable')
            sorted_vals  = vals[order]
            sorted_inv   = self._inverse[order]
            boundaries   = np.flatnonzero(np.diff(sorted_inv))
            group_starts = np.concatenate([[0], boundaries + 1])
            out = sorted_vals[group_starts]

        else:
            raise ValueError(f"Unsupported aggregate function: {func!r}")

        keys_out = PandasStrings(self._unique_keys) if self._is_str else _wrap(self._unique_keys)
        vals_out = PandasStrings(out) if is_str else _wrap(out)
        return keys_out, vals_out


# ---------------------------------------------------------------------------
# AmpereDataFrame — mimics arkouda.DataFrame
# ---------------------------------------------------------------------------

class AmpereDataFrame:
    """
    Backend DataFrame for the pandas mode.

    Stores columns as _PdArray / PandasStrings internally (arkouda-compatible),
    but delegates any unknown attribute to the underlying pandas DataFrame so
    that notebook code using sort_values(), head(), groupby(), etc. just works.
    """

    def __init__(self, data=None):
        self._data: dict = {}
        if data is None:
            return
        if isinstance(data, pd.DataFrame):
            # Construct from an existing pandas DataFrame (e.g. after a pandas op)
            for col in data.columns:
                self._store(col, data[col].values)
            return
        if not isinstance(data, dict):
            raise TypeError(f"AmpereDataFrame expects a dict or pd.DataFrame, got {type(data)}")
        for k, v in data.items():
            self._store(k, v)

    def _store(self, key, value):
        if isinstance(value, PandasStrings):
            self._data[key] = value
        elif value is None:
            self._data[key] = None
        elif isinstance(value, np.ndarray):
            if value.dtype.kind in ('U', 'S', 'O'):
                self._data[key] = PandasStrings(value)
            else:
                self._data[key] = _wrap(value)
        elif isinstance(value, pd.Series):
            self._store(key, value.values)
        else:
            self._data[key] = _wrap(np.asarray(value))

    # ---- arkouda-compatible API ----

    @staticmethod
    def concat(dfs):
        """Concatenate a list of AmpereDataFrames (or pandas DataFrames)."""
        real = [df if isinstance(df, AmpereDataFrame) else AmpereDataFrame(df) for df in dfs]
        real = [df for df in real if df.size > 0]
        if not real:
            return AmpereDataFrame()
        keys = list(real[0].keys())
        combined = {}
        for k in keys:
            parts = [df[k] for df in real]
            if any(isinstance(p, PandasStrings) for p in parts):
                arrs = [
                    p.to_ndarray() if isinstance(p, PandasStrings)
                    else np.asarray(p, dtype=str)
                    for p in parts
                ]
                combined[k] = PandasStrings(np.concatenate(arrs))
            else:
                combined[k] = _wrap(np.concatenate([np.asarray(p) for p in parts]))
        return AmpereDataFrame(combined)

    def to_pandas(self) -> pd.DataFrame:
        pdf = {}
        for k, v in self._data.items():
            if isinstance(v, PandasStrings):
                pdf[k] = v.to_ndarray()
            elif v is None:
                pdf[k] = None
            else:
                pdf[k] = np.asarray(v)
        return pd.DataFrame(pdf)

    @property
    def size(self) -> int:
        if not self._data:
            return 0
        first = next(iter(self._data.values()))
        return 0 if first is None else len(first)

    def keys(self):
        return list(self._data.keys())

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._data[key]
        # Boolean or integer array indexing — filter all columns
        mask = np.asarray(key)
        new_data = {}
        for k, v in self._data.items():
            if isinstance(v, PandasStrings):
                new_data[k] = PandasStrings(v.to_ndarray()[mask])
            elif v is None:
                new_data[k] = None
            else:
                new_data[k] = _wrap(np.asarray(v)[mask])
        return AmpereDataFrame(new_data)

    def __setitem__(self, key, value):
        self._store(key, value)

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"AmpereDataFrame(cols={list(self._data.keys())}, rows={self.size})"

    def __getattr__(self, name):
        # Delegate unknown attributes (sort_values, head, groupby, describe, …)
        # to the pandas representation. Wrap DataFrame results back into
        # AmpereDataFrame so the arkouda-compatible API stays intact.
        if name.startswith('_'):
            raise AttributeError(name)
        pdf = self.to_pandas()
        attr = getattr(pdf, name)
        if not callable(attr):
            return attr
        def _wrapper(*args, **kwargs):
            result = attr(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                return AmpereDataFrame(result)
            if isinstance(result, pd.Series):
                if result.dtype == object:
                    return PandasStrings(result.values)
                return _wrap(result.values)
            return result
        return _wrapper


# ---------------------------------------------------------------------------
# arkouda-compatible module-level functions
# ---------------------------------------------------------------------------

def _cast(arr, dtype):
    if isinstance(arr, PandasStrings):
        return _wrap(arr.to_ndarray().astype(dtype))
    return _wrap(np.asarray(arr).astype(dtype))


def _argsort(arr):
    return _wrap(np.argsort(np.asarray(arr)))


def _searchsorted(arr, vals, side='left'):
    return _wrap(np.searchsorted(np.asarray(arr), np.asarray(vals), side=side))


def _where(cond, x, y):
    return _wrap(np.where(np.asarray(cond), x, y))


def _concatenate(arrays, ordered=True):
    filtered = [a for a in arrays if a is not None]
    if not filtered:
        return None
    if any(isinstance(a, PandasStrings) for a in filtered):
        parts = [
            a.to_ndarray() if isinstance(a, PandasStrings) else np.asarray(a, dtype=str)
            for a in filtered
        ]
        return PandasStrings(np.concatenate(parts))
    return _wrap(np.concatenate([np.asarray(a) for a in filtered]))


def _unique(arr):
    if isinstance(arr, PandasStrings):
        return PandasStrings(np.unique(arr.to_ndarray()))
    return _wrap(np.unique(np.asarray(arr)))


def _sort(arr):
    return _wrap(np.sort(np.asarray(arr)))


def _cumsum(arr):
    return _wrap(np.cumsum(np.asarray(arr)))


def _zeros(size, dtype=np.float64):
    return _wrap(np.zeros(size, dtype=dtype))


def _ones(size, dtype=np.float64):
    return _wrap(np.ones(size, dtype=dtype))


def _arange(size):
    return _wrap(np.arange(size))


def _array(data):
    if isinstance(data, PandasStrings):
        return data
    if isinstance(data, (list, tuple)):
        if data and isinstance(data[0], str):
            return PandasStrings(np.array(data, dtype=str))
        return _wrap(np.array(data))
    if isinstance(data, np.ndarray):
        if data.dtype.kind in ('U', 'S', 'O'):
            return PandasStrings(data)
        return _wrap(data)
    return _wrap(np.asarray(data))


def _read_csv(path, column_delim=','):
    df = pd.read_csv(path, sep=column_delim)
    data = {}
    for col in df.columns:
        series = df[col]
        if series.dtype == object:
            data[col] = PandasStrings(series.values.astype(str))
        else:
            data[col] = _wrap(series.values)
    return AmpereDataFrame(data)


def _read_parquet(path, **kwargs):
    """Pandas-backend counterpart to arkouda.read_parquet.

    Reads one Parquet file (or a list of files, concatenated) via pandas/pyarrow and
    returns an AmpereDataFrame whose columns are _PdArray / PandasStrings, matching the
    dict-like interface (`in`, `[]`) that Ensemble.from_trace_paths_parquet expects.
    """
    paths = path if isinstance(path, (list, tuple)) else [path]
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    data = {}
    for col in df.columns:
        series = df[col]
        if series.dtype == object:
            data[col] = PandasStrings(series.values.astype(str))
        else:
            data[col] = _wrap(series.values)
    return AmpereDataFrame(data)


def _full(size, fill_value, dtype=None):
    """Pandas-backend counterpart to arkouda.full: a length-`size` array of `fill_value`.

    A string fill returns PandasStrings (so it survives the string-aware code paths);
    anything else returns a numeric _PdArray.
    """
    size = int(size)
    if isinstance(fill_value, str) or dtype is str:
        return PandasStrings(np.array([fill_value] * size, dtype=str))
    return _wrap(np.full(size, fill_value, dtype=dtype))


# ---------------------------------------------------------------------------
# PandasBackend — the namespace object returned by _backend.py
# ---------------------------------------------------------------------------

class PandasBackend:
    """Namespace that mirrors the arkouda module interface."""

    # Types used in isinstance() and dtype comparisons
    pdarray = _PdArray
    Strings = PandasStrings

    # dtype constants — must equal np.dtype comparisons in the codebase
    float64 = np.float64
    int64 = np.int64
    bool = np.bool_
    bool_ = np.bool_

    # Classes
    DataFrame = AmpereDataFrame
    GroupBy = PandasGroupBy

    # Functions
    cast = staticmethod(_cast)
    argsort = staticmethod(_argsort)
    searchsorted = staticmethod(_searchsorted)
    where = staticmethod(_where)
    concatenate = staticmethod(_concatenate)
    unique = staticmethod(_unique)
    sort = staticmethod(_sort)
    cumsum = staticmethod(_cumsum)
    zeros = staticmethod(_zeros)
    ones = staticmethod(_ones)
    arange = staticmethod(_arange)
    array = staticmethod(_array)
    read_csv = staticmethod(_read_csv)
    read_parquet = staticmethod(_read_parquet)
    full = staticmethod(_full)

    @staticmethod
    def connect(*args, **kwargs):
        pass

    @staticmethod
    def disconnect(*args, **kwargs):
        pass
