import numpy as np
from ._backend import ak, set_backend, get_backend  # noqa: F401
import os
import re
import csv
import concurrent.futures
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union, Any, Literal, Callable, Pattern
from collections import defaultdict
from tqdm import tqdm
from dataclasses import dataclass


from .session import AmpereSession, connect

# ==========================================
# 1. Configuration & Types
# ==========================================

class MetricType(Enum):
    """
    Enumeration of metric types.
    - INSTANTANEOUS: Sampled values at specific points in time (e.g., Watts).
    - CUMULATIVE: Monotonically increasing counter values (e.g., Energy Joules).
    """
    INSTANTANEOUS = 1
    CUMULATIVE = 2

@dataclass
class MetricConfig:
    """
    Configuration for a specific metric.
    
    Attributes:
        kind (MetricType): The type of the metric (INSTANTANEOUS or CUMULATIVE).
        scale_factor (float): Multiplier to apply to raw values (e.g., 1e-6 to convert microjoules to Joules).
        interpolation_kind (str): Interpolation method for instantaneous metrics ('linear' or 'previous').
    """
    kind: MetricType
    scale_factor: float = 1.0
    interpolation_kind: str = 'linear' 

TopologyResolver = Callable[[str, List['Rank']], List['Rank']]

# ==========================================
# 1b. Backend-neutral DataFrame type
# ==========================================

class _DataFrameMeta(type):
    """Metaclass so isinstance() and construction delegate to the active backend."""
    def __instancecheck__(cls, instance):
        return isinstance(instance, ak.DataFrame)

    def __call__(cls, data=None):
        return ak.DataFrame(data)


class DataFrame(metaclass=_DataFrameMeta):
    """
    Backend-neutral DataFrame type.

    Works identically regardless of whether the arkouda or pandas backend is
    active.  Import it instead of ``arkouda.DataFrame`` in notebooks:

        from ampere import DataFrame, set_backend
        set_backend('pandas')           # or 'arkouda'

        df = DataFrame({'col': arr})    # constructs the right type
        isinstance(df, DataFrame)       # True on both backends
        df = DataFrame.concat([a, b])   # concatenates on the right backend
    """

    @staticmethod
    def concat(dfs):
        """Concatenate a list of DataFrames using the active backend."""
        return ak.DataFrame.concat(dfs)


# ==========================================
# 2. Math & Logic Core
# ==========================================

def ak_interp1d(x: np.ndarray, y: np.ndarray, xi: np.ndarray, kind: str = 'linear') -> np.ndarray:
    """
    Performs one-dimensional linear interpolation on Arkouda arrays.
    
    Args:
        x (np.ndarray): X-coordinates of the data points (must be sorted).
        y (np.ndarray): Y-coordinates of the data points.
        xi (np.ndarray): X-coordinates at which to evaluate the interpolated values.
        kind (str): Interpolation type ('linear' or 'previous'). 'previous' works like step-post.

    Returns:
        np.ndarray: The interpolated values.
    """
    # 1. Find indices of xi in x
    idx = ak.searchsorted(x, xi)
    
    # 2. Clamp indices to valid range [1, n-1]
    n = x.size
    idx = ak.where(idx < 1, 1, idx)
    idx = ak.where(idx >= n, n - 1, idx)
    
    # 3. Gather surrounding points (x0, y0) and (x1, y1)
    x0 = x[idx - 1]
    x1 = x[idx]
    y0 = y[idx - 1]
    y1 = y[idx]
    
    if kind == 'previous':
        return y0
    
    # 4. Perform linear interpolation
    run = x1 - x0
    rise = y1 - y0
    fraction = (xi - x0) / run
    fraction = ak.where(run == 0, 0.0, fraction)
    return y0 + (rise * fraction)

class Metric:
    """
    Represents a time-series metric with associated values and configuration.
    
    Attributes:
        name (str): The name of the metric.
        times (np.ndarray): Array of monotonically increasing timestamps (float64).
        raw_values (np.ndarray): Array of metric values corresponding exactly to `times` (float64).
        config (MetricConfig): Configuration defining metric type (INSTANTANEOUS/CUMULATIVE) and scaling factor.
        cum_values (np.ndarray): Integrated cumulative values derived from raw values (used for delta calculations).
    """
    def __init__(self, name: str, times: np.ndarray, values: np.ndarray, config: MetricConfig):
        self.name = name
        self.kind = config.kind
        
        # Validate and cast inputs to Float64
        if times.dtype != ak.float64: times = ak.cast(times, ak.float64)
        if values.dtype != ak.float64: values = ak.cast(values, ak.float64)

        # 1. Sort
        perm = ak.argsort(times)
        self.times = times[perm]
        self.raw_values = values[perm] * config.scale_factor
        
        self.t_min = self.times[0]
        self.t_max = self.times[-1]
        self.interp_kind = config.interpolation_kind
        if self.kind == MetricType.INSTANTANEOUS and self.interp_kind == 'linear':
            self.interp_kind = 'previous'

        # 2. Integrate
        if self.kind == MetricType.INSTANTANEOUS:
            dt = self.times[1:] - self.times[:-1]
            if self.interp_kind == 'previous':
                energy_steps = self.raw_values[:-1] * dt
            else:
                avg_watts = (self.raw_values[:-1] + self.raw_values[1:]) * 0.5
                energy_steps = avg_watts * dt
            
            zeros = ak.zeros(1, dtype=ak.float64)
            self.cum_values = ak.concatenate([zeros, ak.cumsum(energy_steps)])
        else:
            self.cum_values = self.raw_values

    @property
    def values(self) -> np.ndarray:
        """Alias for raw_values to support legacy/external access."""
        return self.raw_values

    def get_delta_vectorized(self, t_starts: np.ndarray, t_ends: np.ndarray) -> np.ndarray:
        """
        Calculates the change in metric value for a set of time intervals.
        
        Args:
            t_starts (np.ndarray): Array of interval start times.
            t_ends (np.ndarray): Array of interval end times.

        Returns:
            np.ndarray: The delta (change in value) for each interval.
        """
        t_s = ak.where(t_starts < self.t_min, self.t_min, t_starts)
        t_e = ak.where(t_ends > self.t_max, self.t_max, t_ends)
        valid = t_e > t_s
        
        val_start = ak_interp1d(self.times, self.cum_values, t_s, kind='linear')
        val_end   = ak_interp1d(self.times, self.cum_values, t_e, kind='linear')
        return ak.where(valid, val_end - val_start, 0.0)

    def get_statistics_vectorized(self, t_starts: np.ndarray, t_ends: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Computes vectorized statistics for each interval [start, end].
        Returns a dict of np.ndarrays: 'min', 'max', 'mean', 'rate', 'sum'.
        """
        # Clamp intervals to the metric's time range
        t_s = ak.where(t_starts < self.t_min, self.t_min, t_starts)
        t_e = ak.where(t_ends > self.t_max, self.t_max, t_ends)
        valid = t_e > t_s
        
        # Calculate Rate and Mean via integration
        deltas = self.get_delta_vectorized(t_s, t_e)
        durations = t_e - t_s
        safe_dur = ak.where(durations == 0, 1.0, durations)
        rates = deltas / safe_dur
        rates = ak.where(valid, rates, 0.0)
        
        # Approximate Min/Max by sampling start and end points of the interval.
        # Note: True min/max over an interval would require segment reduction which is computationally expensive here.
        v_start = ak_interp1d(self.times, self.raw_values, t_s, kind='linear')
        v_end = ak_interp1d(self.times, self.raw_values, t_e, kind='linear')
        
        return {
            'mean': rates, 
            'rate': rates,
            'min': ak.where(v_start < v_end, v_start, v_end),
            'max': ak.where(v_start > v_end, v_start, v_end),
            'sum': deltas
        }

class AttributionEngine:
    """
    Engine for attributing metric values to call graph ranks based on time and depth.
    """
    @staticmethod
    def _compute_coverage_ak(starts: np.ndarray, ends: np.ndarray, breaks: np.ndarray) -> np.ndarray:
        """
        Computes the number of intervals covered by each segment [start, end] in the `breaks` timeline.
        
        The `breaks` array represents a global, sorted timeline of all unique start and end times 
        from the metric and all ranks. It divides time into discrete, non-overlapping intervals.
        For example, breaks=[0, 10, 20] defines intervals [0, 10) and [10, 20).

        Args:
            starts (np.ndarray): Start times of the segments.
            ends (np.ndarray): End times of the segments.
            breaks (np.ndarray): Global timeline of unique timestamps defining intervals.

        Returns:
            np.ndarray: An array where each element corresponds to an interval defined by `breaks` (size = breaks.size - 1).
                        The value at index `i` is the count of segments that overlap with interval `i`.
        """
        l_idx = ak.searchsorted(breaks, starts, side='right') - 1
        r_idx = ak.searchsorted(breaks, ends, side='left')
        l_idx = ak.where(l_idx < 0, 0, l_idx)
        
        valid = r_idx > l_idx
        l_valid = l_idx[valid]
        r_valid = r_idx[valid]
        
        if l_valid.size == 0:
            return ak.zeros(breaks.size - 1, dtype=ak.int64)

        idxs = ak.concatenate([l_valid, r_valid])
        ones = ak.ones(l_valid.size, dtype=ak.int64)
        vals = ak.concatenate([ones, ones * -1])
        
        g = ak.GroupBy(idxs)
        unique_idxs, summed_vals = g.aggregate(vals, 'sum')
        
        # Filter out OOB indices (e.g. ends beyond the timeline)
        mask_valid = unique_idxs < breaks.size
        unique_idxs = unique_idxs[mask_valid]
        summed_vals = summed_vals[mask_valid]
        
        diff_arr = ak.zeros(breaks.size, dtype=ak.int64)
        diff_arr[unique_idxs] += summed_vals
        return ak.cumsum(diff_arr)[:-1]

    @staticmethod
    def _exclusive_pandas(starts_np: np.ndarray, ends_np: np.ndarray, weights: np.ndarray,
                          depths_np: np.ndarray = None) -> np.ndarray:
        """
        O(K log K) exclusive attribution via the identity:
            exclusive[k] = inclusive[k] - sum(inclusive[direct_children_of_k])

        When depths_np is supplied, parents are found with pure-numpy searchsorted
        per depth level — works for consecutive (0,1,2) and gapped (0,2,5) depths.
        Falls back to the list-stack algorithm only when no depth info is available.
        """
        K = len(starts_np)
        if K == 0:
            return np.array([], dtype=np.float64)

        parent = np.full(K, -1, dtype=np.int64)

        if depths_np is not None:
            depth_vals = np.asarray(depths_np, dtype=np.int64)
            unique_d   = np.unique(depth_vals)
        else:
            unique_d   = np.array([], dtype=np.int64)

        _use_stack_walk = True
        if depths_np is not None and len(unique_d) > 1:
            # Generalised depth path: works for consecutive (0,1,2) AND gapped (0,2,5)
            # depths. For each depth level d, the parent is the last shallower call
            # (any depth < d) that started before the child AND ends after it.
            # Pure-numpy, O(K log K) total — no Python loop over calls.
            # If any non-enclosing spans are found (non-nested / unusual trace),
            # the for-else aborts and the stack walk below handles it instead.
            for d in unique_d[1:]:
                c_idxs     = np.where(depth_vals == d)[0]
                p_idxs_raw = np.where(depth_vals < d)[0]
                if len(p_idxs_raw) == 0:
                    continue
                p_ord           = np.argsort(starts_np[p_idxs_raw], kind='stable')
                p_starts_sorted = starts_np[p_idxs_raw[p_ord]]
                p_orig          = p_idxs_raw[p_ord]

                pos   = np.searchsorted(p_starts_sorted, starts_np[c_idxs], side='right') - 1
                valid = pos >= 0

                if not np.any(valid):
                    continue

                cand_p   = p_orig[pos[valid]]
                cand_c   = c_idxs[valid]
                enclosed = ends_np[cand_p] >= ends_np[cand_c]

                if not np.all(enclosed):
                    # Non-nested spans detected — abort and use stack walk
                    break

                parent[cand_c] = cand_p
            else:
                _use_stack_walk = False  # depth path succeeded for all levels

        if _use_stack_walk:
            # No depth info (or single depth): stack walk — O(K) amortised.
            # If starts are strictly increasing (typical HPC trace order) skip lexsort.
            if K > 1 and bool(np.all(starts_np[1:] > starts_np[:-1])):
                order = list(range(K))
            else:
                order = np.lexsort((-ends_np, starts_np)).tolist()
            starts_lst = starts_np.tolist()
            ends_lst   = ends_np.tolist()
            parent_lst = [-1] * K
            stack_e: list = []
            stack_i: list = []

            for k in order:
                sk = starts_lst[k]
                while stack_e and stack_e[-1] <= sk:
                    stack_e.pop()
                    stack_i.pop()
                if stack_e:
                    parent_lst[k] = stack_i[-1]
                stack_e.append(ends_lst[k])
                stack_i.append(k)

            parent = np.asarray(parent_lst, dtype=np.int64)

        valid        = parent >= 0
        children_sum = np.bincount(parent[valid].astype(np.intp),
                                   weights=weights[valid], minlength=K)
        return np.maximum(0.0, weights - children_sum)

    @staticmethod
    def compute(
        metric: Optional[Metric],
        ranks: List['Rank'],
        concurrency_mode: str = 'shared',
        strategy: str = 'inclusive',
        output_mode: str = 'quantity'
    ) -> Dict[str, Any]:
        """
        Attributes metric values to the provided ranks using the specified strategy.

        Algorithm Overview:
        1. **Global Timeline Construction**: Aggregates all start/end times from the metric and all ranks into a unique, sorted `breaks` array.
           This timeline allows us to process the entire trace as a sequence of discrete, uniform-state intervals.
        2. **Metric Delta Calculation**: Computes the change in the metric value (delta) for each interval in `breaks`.
        3. **Attribution**:
            - **Inclusive**:
                - Calculates the 'active count' of ranks for each interval.
                - If `concurrency_mode='shared'`, divides the interval's metric delta by the active count.
                - Attributes this share to *every* function active in that interval.
                - Result: Function value = Sum of shares for all intervals where it was active.
            - **Exclusive**:
                - Similar to inclusive, but attributes the metric share *only* to the deepest active function in the call stack for each rank.
                - Effectively removes the cost of children from the parent.

        DataFrame Schemas:
        - **Input Ranks**: Requires the following columns (as parallel arrays in the `Rank` object):
            - `Start Time` (float): Function start timestamp (seconds).
            - `End Time` (float): Function end timestamp (seconds).
            - `Name` (str): Function name.
            - `Depth` (int): Stack depth of the function call (0 = root).
        - **Output DataFrames**:
            - `Start Time` (float): Function start timestamp.
            - `End Time` (float): Function end timestamp.
            - `Name` (str): Function name.
            - `Depth` (int): Stack depth.
            - `Value` (float): attributed metric quantity (or rate/min/max depending on `output_mode`).

        Args:
            metric (Metric): The metric to attribute.
            ranks (List[Rank]): List of ranks (call graphs) to attribute to.
            concurrency_mode (str): 'shared' to evenly split metric usage among concurrently active ranks, 
                                    'independent' to attribute the full metric delta to each active rank (double counting).
            strategy (str): 'inclusive' (full subtree cost) or 'exclusive' (self cost only, deepest function wins).
            output_mode (str): 'quantity' (raw attributed value), 'rate' (per second), 'mean', 'min', 'max'.

        Returns:
            Dict[str, Any]: A dictionary mapping rank names to a DataFrame of attributed results.
        """

        # Fast path for pandas backend + independent concurrency.
        # inclusive[k] = metric.get_delta_vectorized(s_k, e_k) — direct interpolation, no global timeline.
        # For time profile (metric=None) inclusive[k] = duration.
        if get_backend() == 'pandas' and concurrency_mode == 'independent':
            results = {}
            for r in ranks:
                s = np.asarray(r.starts, dtype=np.float64)
                e = np.asarray(r.ends,   dtype=np.float64)
                if metric is not None:
                    inclusive = np.asarray(metric.get_delta_vectorized(r.starts, r.ends), dtype=np.float64)
                else:
                    inclusive = e - s
                if strategy == 'exclusive':
                    val = AttributionEngine._exclusive_pandas(
                        s, e, inclusive,
                        depths_np=np.asarray(r.depths, dtype=np.int64),
                    )
                else:
                    val = inclusive
                if output_mode in ('rate', 'mean') and metric is not None:
                    dur = e - s
                    safe_dur = np.where(dur == 0, 1.0, dur)
                    val = np.where(dur == 0, 0.0, val / safe_dur)
                elif output_mode in ('min', 'max') and metric is not None:
                    stats = metric.get_statistics_vectorized(r.starts, r.ends)
                    val = np.asarray(stats[output_mode], dtype=np.float64)
                results[r.name] = ak.DataFrame({
                    'Start Time': r.starts, 'End Time': r.ends,
                    'Duration':   r.ends - r.starts,
                    'Name':       r.names,  'Depth': r.depths,
                    'Value':      ak.array(val),
                    'Metadata':   r.metadata,
                })
            return results

        # Pandas fast path for shared mode:
        # np.unique(return_inverse=True) on all timestamps at once — eliminates all per-rank
        # searchsorted calls (replaced by O(1) index lookup from inverse).
        # bincount replaces GroupBy for coverage — O(N) vs O(N log N).
        if get_backend() == 'pandas':
            # --- Build merged timeline once ---
            parts: list = []
            n_mt = 0
            if metric is not None:
                mt_np = np.asarray(metric.times, dtype=np.float64)
                parts.append(mt_np)
                n_mt = len(mt_np)

            rank_s = [np.asarray(r.starts, dtype=np.float64) for r in ranks]
            rank_e = [np.asarray(r.ends,   dtype=np.float64) for r in ranks]
            for s in rank_s: parts.append(s)
            for e in rank_e: parts.append(e)

            merged_np = np.concatenate(parts)
            breaks_np, inv = np.unique(merged_np, return_inverse=True)
            N = len(breaks_np) - 1
            if N < 1:
                return {r.name: ak.DataFrame(dict()) for r in ranks}

            # --- Extract per-rank break indices from the inverse (no searchsorted) ---
            # merged layout: [metric_times | rank0_starts | rank1_starts | … | rank0_ends | rank1_ends | …]
            cur = n_mt
            rank_l = []   # inv[start position] = break index of call start
            for s in rank_s:
                rank_l.append(inv[cur:cur+len(s)])
                cur += len(s)
            rank_ei = []  # inv[end position] = break index of call end (= r_idx+1 for cumsum)
            for e in rank_e:
                rank_ei.append(inv[cur:cur+len(e)])
                cur += len(e)

            # --- Deltas: evaluate metric at all break points at once ---
            if metric is not None:
                mt_np_arr = np.asarray(metric.times,     dtype=np.float64)
                cv_np_arr = np.asarray(metric.cum_values, dtype=np.float64)
                E_at_breaks = np.interp(breaks_np, mt_np_arr, cv_np_arr)
                deltas = np.maximum(np.diff(E_at_breaks), 0.0)
            else:
                deltas = np.diff(breaks_np)

            # --- Active-rank count per interval via bincount (one pass per rank) ---
            if concurrency_mode == 'shared':
                active_counts = np.zeros(N, dtype=np.int64)
                for li, ei in zip(rank_l, rank_ei):
                    pos = np.bincount(np.clip(li, 0, N), minlength=N+1)
                    neg = np.bincount(np.clip(ei, 0, N), minlength=N+1)
                    cov = np.cumsum((pos - neg).astype(np.int64))[:N]
                    active_counts += (cov > 0).astype(np.int64)
                per_rank_resource = deltas / np.maximum(1, active_counts)
            else:
                per_rank_resource = deltas

            cum = np.concatenate([[0.0], np.cumsum(per_rank_resource)])

            # --- Per-rank attribution ---
            results = {}
            for i, r in enumerate(ranks):
                L = np.clip(rank_l[i],  0, N).astype(np.int64)
                R = np.clip(rank_ei[i], 0, N).astype(np.int64)

                inclusive = cum[R] - cum[L]
                inclusive[R <= L] = 0.0

                if strategy == 'exclusive':
                    val = AttributionEngine._exclusive_pandas(
                        rank_s[i], rank_e[i], inclusive,
                        depths_np=np.asarray(r.depths, dtype=np.int64),
                    )
                else:
                    val = inclusive

                if output_mode in ('rate', 'mean') and metric is not None:
                    dur = rank_e[i] - rank_s[i]
                    val = np.where(dur == 0, 0.0, val / np.where(dur == 0, 1.0, dur))
                elif output_mode in ('min', 'max') and metric is not None:
                    stats = metric.get_statistics_vectorized(r.starts, r.ends)
                    val = np.asarray(stats[output_mode], dtype=np.float64)

                results[r.name] = ak.DataFrame({
                    'Start Time': r.starts, 'End Time': r.ends,
                    'Duration':   r.ends - r.starts,
                    'Name':       r.names,  'Depth': r.depths,
                    'Value':      ak.array(val),
                    'Metadata':   r.metadata,
                })
            return results

        # ---- Arkouda backend: original pipeline ----

        # 1. Global Timeline
        time_arrays = []
        if metric is not None:
            time_arrays.append(metric.times)

        for r in ranks:
            if metric is not None:
                mask_overlap = (r.ends >= metric.t_min) & (r.starts <= metric.t_max)
                try:
                    if mask_overlap.any():
                        time_arrays.append(r.starts[mask_overlap])
                        time_arrays.append(r.ends[mask_overlap])
                except Exception:
                    time_arrays.append(r.starts)
                    time_arrays.append(r.ends)
            else:
                time_arrays.append(r.starts)
                time_arrays.append(r.ends)

        merged = ak.concatenate(time_arrays)
        breaks = ak.unique(merged)

        if breaks.size < 2:
            return {r.name: ak.DataFrame(dict()) for r in ranks}

        if metric is not None:
            deltas = metric.get_delta_vectorized(breaks[:-1], breaks[1:])
        else:
            deltas = breaks[1:] - breaks[:-1]


        rank_coverages = []
        if concurrency_mode == 'shared':
            active_counts = ak.zeros(breaks.size - 1, dtype=ak.int64)
            for r in ranks:
                c = AttributionEngine._compute_coverage_ak(r.starts, r.ends, breaks)
                rank_coverages.append(c)
                active_counts += ak.where(c > 0, 1, 0)
            scaling = ak.where(active_counts < 1, 1, active_counts)
            per_rank_resource = deltas / scaling.astype(ak.float64)
        else:
            per_rank_resource = deltas

        results = {}
        for i, r in enumerate(ranks):
            l_idx = ak.searchsorted(breaks, r.starts, side='right') - 1
            r_idx = ak.searchsorted(breaks, r.ends, side='left') - 1

            idx_start = ak.where(l_idx < 0, 0, l_idx)
            max_idx = breaks.size - 1
            idx_end = r_idx + 1
            idx_end = ak.where(idx_end > max_idx, max_idx, idx_end)
            mask_valid = idx_end > idx_start

            if strategy == 'exclusive':
                # Arkouda backend — original depth-loop algorithm
                unique_depths = ak.unique(r.depths)
                unique_depths = ak.sort(unique_depths)
                max_depth_per_interval = ak.zeros(breaks.size - 1, dtype=ak.int64) - 1
                for d in unique_depths.to_ndarray():
                    mask_d = r.depths == d
                    cov = AttributionEngine._compute_coverage_ak(r.starts[mask_d], r.ends[mask_d], breaks)
                    max_depth_per_interval = ak.where(cov > 0, d, max_depth_per_interval)
                cum_resources_by_depth = {}
                for d in unique_depths.to_ndarray():
                    mask_max_d = max_depth_per_interval == int(d)
                    res_d = ak.where(mask_max_d, per_rank_resource, 0.0)
                    zeros = ak.zeros(1, dtype=ak.float64)
                    cum_resources_by_depth[d] = ak.concatenate([zeros, ak.cumsum(res_d)])
                attributed = ak.zeros(r.starts.size, dtype=ak.float64)
                for d in unique_depths.to_ndarray():
                    mask_calls_at_d = r.depths == d
                    if not mask_calls_at_d.any(): continue
                    s_idx = idx_start[mask_calls_at_d]
                    e_idx = idx_end[mask_calls_at_d]
                    c_res = cum_resources_by_depth[d]
                    max_valid = c_res.size - 1
                    e_idx = ak.where(e_idx > max_valid, max_valid, e_idx)
                    s_idx = ak.where(s_idx > max_valid, max_valid, s_idx)
                    vals = c_res[e_idx] - c_res[s_idx]
                    attributed[mask_calls_at_d] = vals

            else:
                zeros = ak.zeros(1, dtype=ak.float64)
                cum_resource = ak.concatenate([zeros, ak.cumsum(per_rank_resource)])
                vals = cum_resource[idx_end] - cum_resource[idx_start]
                attributed = ak.where(mask_valid, vals, 0.0)

            final_values = attributed

            if output_mode in ['rate', 'mean'] and metric is not None:
                durations = r.ends - r.starts
                safe_dur = ak.where(durations == 0, 1.0, durations)
                final_values = attributed / safe_dur
                final_values = ak.where(durations == 0, 0.0, final_values)
            elif output_mode in ['min', 'max'] and metric is not None:
                stats = metric.get_statistics_vectorized(r.starts, r.ends)
                if output_mode == 'min': final_values = stats['min']
                if output_mode == 'max': final_values = stats['max']

            res_data = {
                'Start Time': r.starts,
                'End Time': r.ends,
                'Duration': r.ends - r.starts,
                'Name': r.names,
                'Depth': r.depths,
                'Value': final_values,
                'Metadata': r.metadata
            }
            results[r.name] = ak.DataFrame(res_data)

        return results

# ==========================================
# 3. Data Structures
# ==========================================

class Rank:
    """
    Represents the call graph execution of a specific rank (thread/process).
    Contains parallel arrays: distinct function calls with start/end times and depths.
    """
    def __init__(self, node: str, name: str, df: Any):
        self.node = node
        self.name = name
        self.starts = df['Start Time']
        self.ends = df['End Time']
        self.names = df['Name']
        self.depths = df['Depth']
        self.durations = self.ends - self.starts
        self.metadata = df.get('Metadata', None)  # Optional additional data
    def __repr__(self): return f"Rank({self.name})"

class Node:
    """
    Represents a compute node containing multiple Ranks and Metrics.
    """
    def __init__(self, name: str, metrics: List[Metric], ranks: List[Rank]):
        self.name = name
        self.ranks = ranks
        self.metrics = {m.name: m for m in metrics}

    def add_derived_metric(self, name: str, func: Callable[..., Metric], *input_names: str):
        """
        Adds a new metric derived from existing metrics in this node.
        
        Args:
            name (str): Name of the new metric.
            func (Callable): Function that returns a new Metric.
            *input_names (str): Names of metrics to pass as arguments to `func`.
                                If empty, passes the entire metrics dictionary.
        """
        try:
            if input_names:
                args = []
                missing = []
                for n in input_names:
                    if n in self.metrics:
                        args.append(self.metrics[n])
                    else:
                        missing.append(n)
                
                if missing:
                    # Fail silently or log? Let's log but not crash.
                    print(f"Cannot derive '{name}': Missing input metrics {missing} in node '{self.name}'.")
                    return

                new_metric = func(*args)
            else:
                # pass dict if no specific args requested
                new_metric = func(self.metrics)

            if new_metric:
                if new_metric.name != name:
                    new_metric.name = name # Ensure name matches
                self.metrics[name] = new_metric
        except Exception as e:
            print(f"Failed to derive metric '{name}' for node '{self.name}': {e}")

    def attribute(self, metric_name: str, topology_resolver: TopologyResolver, **kwargs) -> Any:
        """
        Attributes a specific metric to the ranks within this node.
        
        Args:
            metric_name (str): Name of the metric to attribute.
            topology_resolver (Callable): Function to filter/resolve ranks for the metric.
            **kwargs: Additional arguments passed to `AttributionEngine.compute`.

        Returns:
            Any: Combined attribution results for all participating ranks.
        """
        if metric_name not in self.metrics: 
            print(f'Did not find metric "{metric_name}" in node "{self.name}". Available metrics: {list(self.metrics.keys())}')
            return ak.DataFrame(dict())
        participating = topology_resolver(metric_name, self.ranks)
        if not participating:
            print(f"Warning: Topology resolver for '{metric_name}' returned no ranks. Available ranks: {[r.name for r in self.ranks]}")
            return ak.DataFrame(dict())
        
        res_dict = AttributionEngine.compute(self.metrics[metric_name], participating, **kwargs)
        
        dfs = []
        for r_name, df in res_dict.items():
            if df.size > 0:
                nrows = df['Start Time'].size
                df['Rank'] = ak.full(nrows, r_name)
                dfs.append(df)

        if not dfs: return ak.DataFrame(dict())

        keys = list(dfs[0].keys()) if hasattr(dfs[0], 'keys') else dfs[0].columns
        combined = {}
        for k in keys:
            parts = [d[k] for d in dfs]
            combined[k] = None if all(p is None for p in parts) else ak.concatenate(parts)
        nrows = next(v for v in combined.values() if v is not None).size
        combined['Node'] = ak.full(nrows, self.name)
        return ak.DataFrame(combined)

    def time_profile(self, topology_resolver: TopologyResolver = lambda m, r: r, strategy: str = 'inclusive') -> Any:
        """
        Profiles the time spent in functions for this node.

        Args:
            topology_resolver (Callable): Function to filter/resolve ranks. Default includes all.
            strategy (str): 'inclusive' or 'exclusive'.

        Returns:
            Any: Attributed time results.
        """
        participating = topology_resolver("time", self.ranks)
        if not participating: return ak.DataFrame(dict())

        res_dict = AttributionEngine.compute(None, participating, concurrency_mode='independent', strategy=strategy)

        dfs = []
        for r_name, df in res_dict.items():
            if df.size > 0:
                nrows = df['Start Time'].size
                df['Rank'] = ak.full(nrows, r_name)
                dfs.append(df)

        if not dfs: return ak.DataFrame(dict())

        keys = list(dfs[0].keys()) if hasattr(dfs[0], 'keys') else dfs[0].columns
        combined = {}
        for k in keys:
            parts = [d[k] for d in dfs]
            combined[k] = None if all(p is None for p in parts) else ak.concatenate(parts)
        nrows = next(v for v in combined.values() if v is not None).size
        combined['Node'] = ak.full(nrows, self.name)
        return ak.DataFrame(combined)

class Run:
    """
    Top-level container for a trace analysis session, containing multiple Nodes.
    """
    def __init__(self, path: str, nodes: List[Node]):
        self.path = path
        self.name = os.path.basename(path)
        self.nodes = nodes

    @staticmethod
    def from_trace_path(path: str, node_ranks: Dict, metric_configs: Dict = {}) -> 'Run':
        """
        Factory method to create a Run instance from a single trace file path.
        
        Args:
            path (str): Path to the trace directory or file.
            node_ranks (Dict): Mapping of node names to rank IDs.
            metric_configs (Dict): Configuration for specific metrics.

        Returns:
            Run: A populated Run instance.
        """
        return Ensemble.from_trace_paths([path], node_ranks, metric_configs).runs[0]

    def add_derived_metric(self, name: str, func: Callable[..., Metric], *input_names: str):
        """
        Attributes a new derived metric to all nodes in the run.
        """
        for node in self.nodes:
            node.add_derived_metric(name, func, *input_names)

    def attribute(self, metric_name: str, topology_resolver: TopologyResolver, **kwargs) -> Any:
        """
        Attributes a metric across all nodes in this run.

        Args:
            metric_name (str): Name of the metric.
            topology_resolver (Callable): Rank resolution logic.
            **kwargs: Arguments for attribution (strategy, mode, etc.).

        Returns:
            Any: Combined attribution results with an added 'Run' column.
        """
        dfs = [n.attribute(metric_name, topology_resolver, **kwargs) for n in self.nodes]
        dfs = [d for d in dfs if d.size > 0]
        if not dfs: return ak.DataFrame(dict())
        
        keys = list(dfs[0].keys()) if hasattr(dfs[0], 'keys') else dfs[0].columns
        combined = {}
        for k in keys:
            parts = [d[k] for d in dfs]
            combined[k] = None if all(p is None for p in parts) else ak.concatenate(parts)
        nrows = next(v for v in combined.values() if v is not None).size
        combined['Run'] = ak.full(nrows, self.name)
        return ak.DataFrame(combined)

    def time_profile(self, topology_resolver: TopologyResolver = lambda m, r: r, strategy: str = 'inclusive') -> Any:
        """
        Profiles the time spent in functions across all nodes in this run.

        Args:
            topology_resolver (Callable): Rank resolution logic.
            strategy (str): 'inclusive' or 'exclusive'.

        Returns:
            Any: Combined time attribution results.
        """
        dfs = [n.time_profile(topology_resolver, strategy=strategy) for n in self.nodes]
        dfs = [d for d in dfs if d.size > 0]
        if not dfs: return ak.DataFrame(dict())

        keys = list(dfs[0].keys()) if hasattr(dfs[0], 'keys') else dfs[0].columns
        combined = {}
        for k in keys:
            parts = [d[k] for d in dfs]
            combined[k] = None if all(p is None for p in parts) else ak.concatenate(parts)
        nrows = next(v for v in combined.values() if v is not None).size
        combined['Run'] = ak.full(nrows, self.name)
        return ak.DataFrame(combined)

# ==========================================
# 4. Infrastructure
# ==========================================

def _resolve_config(name: str, config_map: Dict) -> MetricConfig:
    if name in config_map: return config_map[name]
    for k, v in config_map.items():
        if hasattr(k, 'match') and k.match(name): return v
        if isinstance(k, str) and k.startswith('^') and re.match(k, name): return v
    return MetricConfig(kind=MetricType.INSTANTANEOUS)

class Ensemble:
    """
    Manages a collection of Runs for aggregate analysis.
    """
    def __init__(self, runs: List[Run]):
        self.runs = runs
    
    @staticmethod
    def _apply_filter_to_dict(df_dict, mask):
        """
        Helper to filter dict/DataFrame columns manually.
        
        Args:
            df_dict (Dict or Any): The dictionary or DataFrame to filter.
            mask (np.ndarray): Boolean mask for filtering.

        Returns:
            Any: A new DataFrame with the filtered data.
        """
        new_dict = {}
        keys = df_dict.keys() if hasattr(df_dict, 'keys') else df_dict.columns
        for k in keys:
            col = df_dict[k]
            if col.size == mask.size:
                if mask.dtype == ak.bool:
                    new_dict[k] = col[ak.arange(mask.size)[mask]]
                else:
                    new_dict[k] = col[mask]
            else:
                new_dict[k] = col 
        return ak.DataFrame(new_dict)

    @staticmethod
    def _get_valid_numeric_mask(arr: np.ndarray) -> np.ndarray:
        """
        Regex-based whitelist for numeric rows.
        Matches strict scientific notation or standard floats.
        Pattern handles: integers, floats, scientific notation (e.g. 1.5e-10)
        Ignores: headers ("Start Time"), empty strings, nans.
        """
        if arr.dtype == ak.float64 or arr.dtype == ak.int64:
            return ak.ones(arr.size, dtype=ak.bool)
        
        if arr.dtype == ak.Strings or isinstance(arr, ak.Strings):
            # Regex: Anchored start/end. 
            # Optional sign [-+]?
            # Number part: digits.digits OR .digits OR digits
            # Exponent part: [eE][-+]digits
            # Uses 'contains' because older Arkouda versions mapped this to RE2 search
            # The anchors ^...$ ensure it's a full match.
            regex_float = r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$'
            return arr.fullmatch(regex_float).matched()
            
        return ak.ones(arr.size, dtype=ak.bool)

    @staticmethod
    def _resolve_config(name: str, config_map: Dict) -> MetricConfig:
        if name in config_map: return config_map[name]
        for k, v in config_map.items():
            if hasattr(k, 'match') and k.match(name): return v
            if isinstance(k, str) and k.startswith('^') and re.match(k, name): return v
        return MetricConfig(kind=MetricType.INSTANTANEOUS)


    @staticmethod
    def from_trace_paths_parquet(trace_paths: List[str], node_ranks: Dict, metric_configs: Dict = {}) -> 'Ensemble':
        """
        Loads multiple trace runs in parallel and constructs an Ensemble from traces stored in Parquet format.


        Args:
            trace_paths (List[str]): List of file paths to trace directories.
            node_ranks (Dict): Dictionary mapping Node names to lists of Rank IDs (e.g., {"Node1": ["Rank0", "Rank1"]}).
            metric_configs (Dict): Dictionary mapping metric naming patterns to `MetricConfig` objects.

        """
        runs = []
        for path in tqdm(trace_paths, desc="Loading Runs"):
            abs_path = os.path.abspath(path)
            nodes = []
            for node_name, ranks in node_ranks.items():
                # IMPROVEMENT: Use Arkouda read_csv for scalable server-side loading
                # Metric loading (Keep sequential as it's small and lacks ID)
                m_path = os.path.join(abs_path, f"{ranks[0]}_metrics.parquet")
                metrics = []
                if os.path.exists(m_path):
                    try:
                        m_df = ak.read_parquet(m_path)
                        if 'metric_name' in m_df and 'time' in m_df and 'value_int' in m_df:
                            m_names = m_df['metric_name']
                            g = ak.GroupBy(m_names)
                            uk, _ = g.aggregate(m_names, 'first')
                            unique_metrics = uk.to_ndarray().tolist()
                            
                            for m_name in unique_metrics:
                                mask = (m_names == m_name)
                                times = m_df['time'][mask]
                                values = m_df['value_int'][mask]
                                if times.dtype != ak.float64: times = ak.cast(times, ak.float64)
                                if values.dtype != ak.float64: values = ak.cast(values, ak.float64)
                                cfg = Ensemble._resolve_config(m_name, metric_configs)
                                metrics.append(Metric(m_name, times, values, cfg))
                        else:
                            print(f"Warning: Metrics file {m_path} is missing required columns. Skipping metrics for node '{node_name}'.")
                    except Exception as e:
                        print(f"Error loading metrics {m_path}: {e}")
        
                valid_c_paths = []
                for r_id in ranks:
                    c_path = os.path.join(abs_path, f"{r_id}_Master_thread_callgraph.parquet")
                    if os.path.exists(c_path):
                        valid_c_paths.append(c_path)
                
                # Also discover additional callgraph files (e.g., HIP Context GPU traces)
                try:
                    for fname in os.listdir(abs_path):
                        if fname.endswith("_callgraph.parquet"):
                            extra_path = os.path.join(abs_path, fname)
                            if extra_path in valid_c_paths:
                                continue
                            if 'MPI Rank ' in fname:
                                continue
                            valid_c_paths.append(extra_path)
                except OSError:
                    pass

                if not valid_c_paths:
                    print(f"Warning: No valid callgraph files found for node '{node_name}' in path '{abs_path}'.")
                    continue
                
                loaded_ranks = []
                for valid_path in valid_c_paths:
                    data = {'Depth': [], 'Start Time': [], 'End Time': [], 'Duration': [], 'Name': [], 'Group': [], 'Metadata': []}
                    
                    try:    
                        
                        df = ak.read_parquet(valid_path)

                        data['Group'] = df['group']
                        data['Depth'] = df['depth']
                        data['Name'] = df['name']
                        data['Start Time'] = df['start_time']
                        data['End Time'] = df['end_time']
                        if 'duration' in df:
                            data['Duration'] = df['duration']
                        else:
                            data['Duration'] = df['end_time'] - df['start_time']
                        if 'metadata' in df:
                            data['Metadata'] = df['metadata']
                        else:
                            data['Metadata'] = ak.full(df['name'].size, "")
                    except Exception as e:
                        print(f"Error loading callgraph {valid_path}: {e}")

                    # Skip empty data (header-only files with no data rows)
                    if not data.get('Name'):
                        print(f"Warning: Callgraph file {valid_path} is empty or has no valid data. Skipping.")
                        continue

                    c_df = ak.DataFrame(data)
                    
                    # Filter: End > Start
                    mask = c_df['End Time'] > c_df['Start Time']
                    c_df = Ensemble._apply_filter_to_dict(c_df, mask)
                    
                    # Identify Rank ID from path or group
                    # We first check the Group column, which is reliable if consistent.
                    # Alternatively, we could assume the filename maps to a rank ID as iterated in the loop.
                    # Here, we extract the rank ID from the Group column of the first row.
                    group_val = data['Group'][0] if data['Group'] else "Unknown"

                    # For non-MPI callgraphs (e.g., HIP GPU contexts), derive
                    # a unique rank name from the filename to avoid collisions
                    # when multiple streams share the same Group value.
                    basename = os.path.basename(valid_path)
                    if basename.startswith("MPI Rank"):
                        rank_name = group_val
                    else:
                        rank_name = basename.rsplit('_callgraph', 1)[0]
                    loaded_ranks.append(Rank(node_name, rank_name, c_df))
                if loaded_ranks:
                    nodes.append(Node(node_name, metrics, loaded_ranks))
            if nodes: runs.append(Run(abs_path, nodes))
        return Ensemble(runs)

    def from_trace_paths(trace_paths: List[str], node_ranks: Dict, metric_configs: Dict = {}, max_workers: int = 32) -> 'Ensemble':
        """
        Loads multiple trace runs in parallel and constructs an Ensemble.
        
        Uses a thread pool to parse client-side CSVs efficiently before transferring 
        data to the Arkouda server. This optimization avoids slow sequential server-side parsing.

        Args:
            trace_paths (List[str]): List of file paths to trace directories.
            node_ranks (Dict): Dictionary mapping Node names to lists of Rank IDs (e.g., {"Node1": ["Rank0", "Rank1"]}).
            metric_configs (Dict): Dictionary mapping metric naming patterns to `MetricConfig` objects.
            max_workers (int): Maximum number of threads for parallel CSV parsing.

        Returns:
            Ensemble: An Ensemble object containing the loaded Runs.
        """
        runs = []
        for path in tqdm(trace_paths, desc="Loading Runs"):
            abs_path = os.path.abspath(path)
            nodes = []
            for node_name, ranks in node_ranks.items():
                # IMPROVEMENT: Use Arkouda read_csv for scalable server-side loading
                # Metric loading (Keep sequential as it's small and lacks ID)
                m_path = os.path.join(abs_path, f"{ranks[0]}_metrics.csv")
                metrics = []
                if os.path.exists(m_path):
                    try:
                        m_df = ak.read_csv(m_path, column_delim=',')
                        if 'Metric Name' in m_df and 'Time' in m_df and 'Value' in m_df:
                            m_names = m_df['Metric Name']
                            g = ak.GroupBy(m_names)
                            uk, _ = g.aggregate(m_names, 'first')
                            unique_metrics = uk.to_ndarray().tolist()
                            
                            for m_name in unique_metrics:
                                mask = (m_names == m_name)
                                times = m_df['Time'][mask]
                                values = m_df['Value'][mask]
                                if times.dtype != ak.float64: times = ak.cast(times, ak.float64)
                                if values.dtype != ak.float64: values = ak.cast(values, ak.float64)
                                cfg = Ensemble._resolve_config(m_name, metric_configs)
                                metrics.append(Metric(m_name, times, values, cfg))
                    except Exception as e:
                        print(f"Error loading metrics {m_path}: {e}")

                # Callgraph loading - PARALLEL CLIENT OPTIMIZATION
                # We use ThreadPoolExecutor to parse CSVs on client (fast) and transfer to Arkouda.
                # This bypasses the slow sequential server-side read_csv.
                
                # Helper to parse one file
                def parse_callgraph_client(path):
                    try:
                        data = {'Depth': [], 'Start Time': [], 'End Time': [], 'Duration': [], 'Name': [], 'Group': [], 'Metadata': []}
                        with open(path, 'r') as f:
                            reader = csv.reader(f, delimiter=',')
                            header = next(reader, None) # Skip header
                            for row in reader:
                                if len(row) < 7: continue # Skip malformed lines
                                # Indices: 0:Thread, 1:Group, 2:Depth, 3:Name, 4:Start, 5:End, 6:Duration
                                data['Group'].append(row[1])
                                data['Depth'].append(int(row[2]))
                                data['Name'].append(row[3])
                                data['Start Time'].append(float(row[4]))
                                data['End Time'].append(float(row[5]))
                                if len(row) > 6:
                                    data['Duration'].append(float(row[6]))
                                else:
                                    data['Duration'].append(float(row[5]) - float(row[4]))
                                if len(row) > 7:
                                    data['Metadata'].append(row[7])
                                else:
                                    data['Metadata'].append('')
                        return (path, data)
                    except Exception as e:
                        return (path, e)

                valid_c_paths = []
                for r_id in ranks:
                    c_path = os.path.join(abs_path, f"{r_id}_Master_thread_callgraph.csv")
                    if os.path.exists(c_path):
                        valid_c_paths.append(c_path)
                
                # Also discover additional callgraph files (e.g., HIP Context GPU traces)
                try:
                    for fname in os.listdir(abs_path):
                        if fname.endswith("_callgraph.csv"):
                            extra_path = os.path.join(abs_path, fname)
                            if extra_path not in valid_c_paths:
                                valid_c_paths.append(extra_path)
                except OSError:
                    pass
                
                loaded_ranks = []
                if valid_c_paths:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_path = {executor.submit(parse_callgraph_client, p): p for p in valid_c_paths}
                        
                        for future in concurrent.futures.as_completed(future_to_path):
                            path, result = future.result()
                            if isinstance(result, Exception):
                                print(f"Error parse-loading callgraph {path}: {result}")
                                continue
                            
                            # Skip empty data (header-only files with no data rows)
                            if not result.get('Name'):
                                continue
                            
                            # Transfer to Arkouda
                            try:
                                # Create dict of arrays
                                ak_dict = {}
                                ak_dict['Depth'] = ak.array(result['Depth'])
                                ak_dict['Start Time'] = ak.array(result['Start Time'])
                                ak_dict['End Time'] = ak.array(result['End Time'])
                                ak_dict['Duration'] = ak.array(result['Duration'])
                                ak_dict['Name'] = ak.array(result['Name'])
                                ak_dict['Group'] = ak.array(result['Group'])
                                ak_dict['Metadata'] = ak.array(result['Metadata'])
                                
                                c_df = ak.DataFrame(ak_dict)
                                
                                # Filter: End > Start
                                mask = c_df['End Time'] > c_df['Start Time']
                                c_df = Ensemble._apply_filter_to_dict(c_df, mask)
                                
                                # Identify Rank ID from path or group
                                # We first check the Group column, which is reliable if consistent.
                                # Alternatively, we could assume the filename maps to a rank ID as iterated in the loop.
                                # Here, we extract the rank ID from the Group column of the first row.
                                group_val = result['Group'][0] if result['Group'] else "Unknown"
                                
                                # For non-MPI callgraphs (e.g., HIP GPU contexts), derive
                                # a unique rank name from the filename to avoid collisions
                                # when multiple streams share the same Group value.
                                basename = os.path.basename(path)
                                if basename.startswith("MPI Rank"):
                                    rank_name = group_val
                                else:
                                    rank_name = basename.rsplit('_callgraph', 1)[0]
                                
                                loaded_ranks.append(Rank(node_name, rank_name, c_df))
                                
                            except Exception as e:
                                print(f"Error transferring callgraph {path}: {e}")
                
                if loaded_ranks:
                    nodes.append(Node(node_name, metrics, loaded_ranks))
            if nodes: runs.append(Run(abs_path, nodes))
        return Ensemble(runs)
    
    def add_derived_metric(self, name: str, func: Callable[..., Metric], *input_names: str):
        """
        Adds a derived metric to all runs in the ensemble.
        
        Args:
            name (str): Name of the new metric.
            func (Callable): Function that returns a new Metric.
            *input_names (str): Names of metrics to pass as arguments to `func`.
        """
        print(f"Deriving metric '{name}'...")
        for run in self.runs:
            run.add_derived_metric(name, func, *input_names)

    def attribute(self, metric_name: str, topology_resolver: TopologyResolver = lambda m, r: r, 
                  concurrency_mode: str = 'shared',
                  strategy: str = 'inclusive',
                  output_mode: str = 'quantity') -> Any:
        """
        Performs metric attribution across the entire ensemble of runs.

        Args:
            metric_name (str): The name of the metric to attribute.
            topology_resolver (Callable): Function to resolve participating ranks. Defaults to identity.
            concurrency_mode (str): 'shared' or 'independent' (see AttributionEngine).
            strategy (str): 'inclusive' or 'exclusive' (see AttributionEngine).
            output_mode (str): Output value format (quantity, rate, etc.).

        Returns:
            Any: A concatenated DataFrame containing results from all runs.
        """
        
        dfs = []
        backend = get_backend()
        print(f"Attributing '{metric_name}' ({'pandas' if backend == 'pandas' else 'Arkouda'} backend)...")
        for run in tqdm(self.runs):
            df = run.attribute(metric_name, topology_resolver, concurrency_mode=concurrency_mode, strategy=strategy, output_mode=output_mode)
            if df.size > 0: dfs.append(df)
            
        if not dfs: 
            raise KeyError(f"No data found for metric '{metric_name}'. Check metric name or topology.")
        
        keys = list(dfs[0].keys()) if hasattr(dfs[0], 'keys') else dfs[0].columns
        combined = {}
        for k in keys:
            parts = [d[k] for d in dfs]
            combined[k] = None if all(p is None for p in parts) else ak.concatenate(parts)
        return ak.DataFrame(combined)

    def time_profile(self, topology_resolver: TopologyResolver = lambda m, r: r, strategy: str = 'inclusive') -> Any:
        """
        Profiles the time spent in functions across the entire ensemble of runs.

        Args:
            topology_resolver (Callable): Function to resolve participating ranks. Default includes all.
            strategy (str): 'inclusive' or 'exclusive'.

        Returns:
            Any: A concatenated DataFrame containing results from all runs.
        """
        dfs = []
        backend = get_backend()
        print(f"Profiling time ({'pandas' if backend == 'pandas' else 'Arkouda'} backend)...")
        for run in tqdm(self.runs):
            df = run.time_profile(topology_resolver, strategy=strategy)
            if df.size > 0: dfs.append(df)

        if not dfs:
            raise KeyError("No data found for time profiling. Check topology.")

        keys = list(dfs[0].keys()) if hasattr(dfs[0], 'keys') else dfs[0].columns
        combined = {}
        for k in keys:
            parts = [d[k] for d in dfs]
            combined[k] = None if all(p is None for p in parts) else ak.concatenate(parts)
        return ak.DataFrame(combined)

if __name__ == "__main__":
    # ak.connect(server="localhost", port=5555) 
    
    configs = {re.compile(r".*energy.*"): MetricConfig(MetricType.CUMULATIVE, scale_factor=1e-6)}
    topo = {"Node0": ["Rank0", "Rank1"]}
    run = Run.from_trace_path("./examples/trace", topo, configs)
    ak_df = run.attribute("rocm_smi:::energy_count:device0", lambda m,r: r, strategy='inclusive')
    if len(ak_df) > 0:
        print(ak_df.to_pandas().head())

from .visualizer import Visualizer
