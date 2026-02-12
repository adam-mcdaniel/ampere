import arkouda as ak
import numpy as np 
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
# 2. Math & Logic Core
# ==========================================

def ak_interp1d(x: ak.pdarray, y: ak.pdarray, xi: ak.pdarray, kind: str = 'linear') -> ak.pdarray:
    """
    Performs one-dimensional linear interpolation on Arkouda arrays.
    
    Args:
        x (ak.pdarray): X-coordinates of the data points (must be sorted).
        y (ak.pdarray): Y-coordinates of the data points.
        xi (ak.pdarray): X-coordinates at which to evaluate the interpolated values.
        kind (str): Interpolation type ('linear' or 'previous'). 'previous' works like step-post.

    Returns:
        ak.pdarray: The interpolated values.
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
        times (ak.pdarray): Array of monotonically increasing timestamps (float64).
        raw_values (ak.pdarray): Array of metric values corresponding exactly to `times` (float64).
        config (MetricConfig): Configuration defining metric type (INSTANTANEOUS/CUMULATIVE) and scaling factor.
        cum_values (ak.pdarray): Integrated cumulative values derived from raw values (used for delta calculations).
    """
    def __init__(self, name: str, times: ak.pdarray, values: ak.pdarray, config: MetricConfig):
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
    def values(self) -> ak.pdarray:
        """Alias for raw_values to support legacy/external access."""
        return self.raw_values

    def get_delta_vectorized(self, t_starts: ak.pdarray, t_ends: ak.pdarray) -> ak.pdarray:
        """
        Calculates the change in metric value for a set of time intervals.
        
        Args:
            t_starts (ak.pdarray): Array of interval start times.
            t_ends (ak.pdarray): Array of interval end times.

        Returns:
            ak.pdarray: The delta (change in value) for each interval.
        """
        t_s = ak.where(t_starts < self.t_min, self.t_min, t_starts)
        t_e = ak.where(t_ends > self.t_max, self.t_max, t_ends)
        valid = t_e > t_s
        
        val_start = ak_interp1d(self.times, self.cum_values, t_s, kind='linear')
        val_end   = ak_interp1d(self.times, self.cum_values, t_e, kind='linear')
        return ak.where(valid, val_end - val_start, 0.0)

    def get_statistics_vectorized(self, t_starts: ak.pdarray, t_ends: ak.pdarray) -> Dict[str, ak.pdarray]:
        """
        Computes vectorized statistics for each interval [start, end].
        Returns a dict of ak.pdarrays: 'min', 'max', 'mean', 'rate', 'sum'.
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
    def _compute_coverage_ak(starts: ak.pdarray, ends: ak.pdarray, breaks: ak.pdarray) -> ak.pdarray:
        """
        Computes the number of intervals covered by each segment [start, end] in the `breaks` timeline.
        
        The `breaks` array represents a global, sorted timeline of all unique start and end times 
        from the metric and all ranks. It divides time into discrete, non-overlapping intervals.
        For example, breaks=[0, 10, 20] defines intervals [0, 10) and [10, 20).

        Args:
            starts (ak.pdarray): Start times of the segments.
            ends (ak.pdarray): End times of the segments.
            breaks (ak.pdarray): Global timeline of unique timestamps defining intervals.

        Returns:
            ak.pdarray: An array where each element corresponds to an interval defined by `breaks` (size = breaks.size - 1).
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
        
        diff_arr = ak.zeros(breaks.size, dtype=ak.int64)
        diff_arr[unique_idxs] += summed_vals
        return ak.cumsum(diff_arr)[:-1]

    @staticmethod
    def compute(
        metric: Metric,
        ranks: List['Rank'],
        concurrency_mode: str = 'shared',
        strategy: str = 'inclusive',
        output_mode: str = 'quantity'
    ) -> Dict[str, ak.DataFrame]:
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
            Dict[str, ak.DataFrame]: A dictionary mapping rank names to a DataFrame of attributed results.
        """
        
        # 1. Global Timeline
        time_arrays = [metric.times]
        for r in ranks:
            mask_s = (r.starts >= metric.t_min) & (r.starts <= metric.t_max)
            mask_e = (r.ends >= metric.t_min) & (r.ends <= metric.t_max)
            try:
                if mask_s.any(): time_arrays.append(r.starts[mask_s])
                if mask_e.any(): time_arrays.append(r.ends[mask_e])
            except Exception:
                # Fallback for old Arkouda versions or different array types
                time_arrays.append(r.starts)
                time_arrays.append(r.ends)
            
        merged = ak.concatenate(time_arrays)
        breaks = ak.unique(merged)
        
        if breaks.size < 2:
            return {r.name: ak.DataFrame(dict()) for r in ranks}

        # Compute Base Quantities (Deltas) for Attribution
        deltas = metric.get_delta_vectorized(breaks[:-1], breaks[1:])
        
        # Calculate active counts or max depth depending on strategy
        if strategy == 'exclusive':
            # Exclusive attribution logic handled later.
            pass 
        else:
            active_counts = ak.zeros(breaks.size - 1, dtype=ak.int64)
            for r in ranks:
                c = AttributionEngine._compute_coverage_ak(r.starts, r.ends, breaks)
                active_counts += ak.where(c > 0, 1, 0)

            if concurrency_mode == 'shared':
                scaling = ak.where(active_counts < 1, 1, active_counts)
                per_rank_resource = deltas / scaling.astype(ak.float64)
            else:
                per_rank_resource = deltas

            # Accumulate Resource
            zeros = ak.zeros(1, dtype=ak.float64)
            cum_resource = ak.concatenate([zeros, ak.cumsum(per_rank_resource)])
        
        results = {}
        
        # Optimize: Pre-compute max depth for each rank if exclusive
        # If exclusive, we don't have a single global "cum_resource" because each rank might claim different parts differently?
        # Actually, if concurrency_mode=shared, we split the metric among active RANKS first.
        # THEN within the rank, we give it to the deepest function.
        
        # Let's handle concurrency first (split metric between Ranks)
        # Then handle exclusive (split metric within Rank)
        
        # Re-calc active_counts for concurrency splitting (needed for both)
        active_counts = ak.zeros(breaks.size - 1, dtype=ak.int64)
        rank_coverages = []
        for r in ranks:
            c = AttributionEngine._compute_coverage_ak(r.starts, r.ends, breaks)
            rank_coverages.append(c)
            active_counts += ak.where(c > 0, 1, 0)
            
        if concurrency_mode == 'shared':
            scaling = ak.where(active_counts < 1, 1, active_counts)
            per_rank_resource = deltas / scaling.astype(ak.float64)
        else:
            per_rank_resource = deltas

        # Now per_rank_resource is the amount of resource available to be claimed by THIS rank in this interval.
        # Use this to build a cumulative curve? 
        # For inclusive, we just need to know if we are active.
        # For exclusive, we need to know if we are the DEEPEST active.
        
        for i, r in enumerate(ranks):
            # 1. Identify intervals where this rank is active
            # We have rank_coverages[i] which tells us overlap count (>=1 if active)
            # But for exclusive, we need to check depths.
            
            # Get start/end indices in 'breaks' for each function call
            l_idx = ak.searchsorted(breaks, r.starts, side='right') - 1
            r_idx = ak.searchsorted(breaks, r.ends, side='left') - 1
            
            idx_start = ak.where(l_idx < 0, 0, l_idx)
            # max_idx is breaks.size-1 (intervals) -> cum_resource has size intervals+1
            max_idx = breaks.size - 1
            idx_end = r_idx + 1
            idx_end = ak.where(idx_end > max_idx, max_idx, idx_end)
            mask_valid = idx_end > idx_start
            
            if strategy == 'exclusive':
                # Exclusive Attribution Strategy:
                # We need to determine the maximum active depth for each interval to strictly attribute
                # resources to the deepest active function.
                #
                # Algorithm:
                # 1. Identify all unique depths in the call graph.
                # 2. Iterate through depths (active depths update the max_depth_per_interval array).
                # 3. Create a cumulative resource array for each depth, masking out intervals where
                #    that depth is not the maximum.
                # 4. Attribute resources to function calls based on their depth and the exclusive resource pool.

                unique_depths = ak.unique(r.depths)
                unique_depths = ak.sort(unique_depths)
                
                # Initialize max active depth per interval
                max_depth_per_interval = ak.zeros(breaks.size - 1, dtype=ak.int64) - 1
                
                # Determine max active depth for each interval
                for d in unique_depths.to_ndarray():
                    mask_d = r.depths == d
                    cov = AttributionEngine._compute_coverage_ak(r.starts[mask_d], r.ends[mask_d], breaks)
                    # Update max_depth if this depth is active (higher depths overwrite lower ones)
                    max_depth_per_interval = ak.where(cov > 0, d, max_depth_per_interval)
                
                # Pre-calculate cumulative resources for each depth level
                cum_resources_by_depth = {}
                for d in unique_depths.to_ndarray():
                    mask_max_d = max_depth_per_interval == int(d)
                    res_d = ak.where(mask_max_d, per_rank_resource, 0.0)
                    zeros = ak.zeros(1, dtype=ak.float64)
                    cum_resources_by_depth[d] = ak.concatenate([zeros, ak.cumsum(res_d)])
                
                # Assign attributed values to calls
                attributed = ak.zeros(r.starts.size, dtype=ak.float64)
                
                for d in unique_depths.to_ndarray():
                    mask_calls_at_d = r.depths == d
                    if not mask_calls_at_d.any(): continue
                    
                    s_idx = idx_start[mask_calls_at_d]
                    e_idx = idx_end[mask_calls_at_d]
                    
                    c_res = cum_resources_by_depth[d]
                    vals = c_res[e_idx] - c_res[s_idx]
                    attributed[mask_calls_at_d] = vals
                
            else:
                # Inclusive
                # Resource is just per_rank_resource
                zeros = ak.zeros(1, dtype=ak.float64)
                cum_resource = ak.concatenate([zeros, ak.cumsum(per_rank_resource)])
                
                vals = cum_resource[idx_end] - cum_resource[idx_start]
                attributed = ak.where(mask_valid, vals, 0.0)

            # Post-Process Output Mode
            final_values = attributed
            
            if output_mode in ['rate', 'mean']:
                durations = r.ends - r.starts
                safe_dur = ak.where(durations == 0, 1.0, durations)
                final_values = attributed / safe_dur
                final_values = ak.where(durations == 0, 0.0, final_values)
            elif output_mode in ['min', 'max']:
                stats = metric.get_statistics_vectorized(r.starts, r.ends)
                if output_mode == 'min': final_values = stats['min']
                if output_mode == 'max': final_values = stats['max']
            
            res_data = {
                'Start Time': r.starts,
                'End Time': r.ends,
                'Name': r.names,
                'Depth': r.depths,
                'Value': final_values
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
    def __repr__(self): return f"Rank({self.name})"

class Node:
    """
    Represents a compute node containing multiple Ranks and Metrics.
    """
    def __init__(self, name: str, metrics: List[Metric], ranks: List[Rank]):
        self.name = name
        self.ranks = ranks
        self.metrics = {m.name: m for m in metrics}

    def attribute(self, metric_name: str, topology_resolver: TopologyResolver, **kwargs) -> ak.DataFrame:
        """
        Attributes a specific metric to the ranks within this node.
        
        Args:
            metric_name (str): Name of the metric to attribute.
            topology_resolver (Callable): Function to filter/resolve ranks for the metric.
            **kwargs: Additional arguments passed to `AttributionEngine.compute`.

        Returns:
            ak.DataFrame: Combined attribution results for all participating ranks.
        """
        if metric_name not in self.metrics: return ak.DataFrame(dict())
        participating = topology_resolver(metric_name, self.ranks)
        if not participating: return ak.DataFrame(dict())
        
        res_dict = AttributionEngine.compute(self.metrics[metric_name], participating, **kwargs)
        
        dfs = []
        for r_name, df in res_dict.items():
            if df.size > 0:
                nrows = df['Start Time'].size
                df['Rank'] = ak.array([r_name] * nrows)
                dfs.append(df)
        
        if not dfs: return ak.DataFrame(dict())
        
        keys = list(dfs[0].keys()) if hasattr(dfs[0], 'keys') else dfs[0].columns
        combined = {}
        for k in keys:
            combined[k] = ak.concatenate([d[k] for d in dfs])
        combined['Node'] = ak.array([self.name] * combined[keys[0]].size)
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

    def attribute(self, metric_name: str, topology_resolver: TopologyResolver, **kwargs) -> ak.DataFrame:
        """
        Attributes a metric across all nodes in this run.

        Args:
            metric_name (str): Name of the metric.
            topology_resolver (Callable): Rank resolution logic.
            **kwargs: Arguments for attribution (strategy, mode, etc.).

        Returns:
            ak.DataFrame: Combined attribution results with an added 'Run' column.
        """
        dfs = [n.attribute(metric_name, topology_resolver, **kwargs) for n in self.nodes]
        dfs = [d for d in dfs if d.size > 0]
        if not dfs: return ak.DataFrame(dict())
        
        keys = list(dfs[0].keys()) if hasattr(dfs[0], 'keys') else dfs[0].columns
        combined = {}
        for k in keys:
            combined[k] = ak.concatenate([d[k] for d in dfs])
        combined['Run'] = ak.array([self.name] * combined[keys[0]].size)
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
            df_dict (Dict or ak.DataFrame): The dictionary or DataFrame to filter.
            mask (ak.pdarray): Boolean mask for filtering.

        Returns:
            ak.DataFrame: A new DataFrame with the filtered data.
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
    def _get_valid_numeric_mask(arr: ak.pdarray) -> ak.pdarray:
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
                        data = {'Depth': [], 'Start Time': [], 'End Time': [], 'Duration': [], 'Name': [], 'Group': []}
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
                                data['Duration'].append(float(row[6]))
                        return (path, data)
                    except Exception as e:
                        return (path, e)

                valid_c_paths = []
                for r_id in ranks:
                    c_path = os.path.join(abs_path, f"{r_id}_Master_thread_callgraph.csv")
                    if os.path.exists(c_path):
                        valid_c_paths.append(c_path)
                
                loaded_ranks = []
                if valid_c_paths:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_path = {executor.submit(parse_callgraph_client, p): p for p in valid_c_paths}
                        
                        for future in concurrent.futures.as_completed(future_to_path):
                            path, result = future.result()
                            if isinstance(result, Exception):
                                print(f"Error parse-loading callgraph {path}: {result}")
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
                                
                                c_df = ak.DataFrame(ak_dict)
                                
                                # Filter: End > Start
                                mask = c_df['End Time'] > c_df['Start Time']
                                c_df = Ensemble._apply_filter_to_dict(c_df, mask)
                                
                                # Identify Rank ID from path or group
                                # We first check the Group column, which is reliable if consistent.
                                # Alternatively, we could assume the filename maps to a rank ID as iterated in the loop.
                                # Here, we extract the rank ID from the Group column of the first row.
                                group_val = result['Group'][0] if result['Group'] else "Unknown"
                                
                                loaded_ranks.append(Rank(node_name, group_val, c_df))
                                
                            except Exception as e:
                                print(f"Error transferring callgraph {path}: {e}")
                
                if loaded_ranks:
                    nodes.append(Node(node_name, metrics, loaded_ranks))
            if nodes: runs.append(Run(abs_path, nodes))
        return Ensemble(runs)
        
    def attribute(self, metric_name: str, topology_resolver: TopologyResolver = lambda m, r: r, 
                  concurrency_mode: str = 'shared',
                  strategy: str = 'inclusive',
                  output_mode: str = 'quantity') -> ak.DataFrame:
        """
        Performs metric attribution across the entire ensemble of runs.

        Args:
            metric_name (str): The name of the metric to attribute.
            topology_resolver (Callable): Function to resolve participating ranks. Defaults to identity.
            concurrency_mode (str): 'shared' or 'independent' (see AttributionEngine).
            strategy (str): 'inclusive' or 'exclusive' (see AttributionEngine).
            output_mode (str): Output value format (quantity, rate, etc.).

        Returns:
            ak.DataFrame: A concatenated DataFrame containing results from all runs.
        """
        
        dfs = []
        print(f"Attributing '{metric_name}' on Arkouda Server...")
        for run in tqdm(self.runs):
            df = run.attribute(metric_name, topology_resolver, concurrency_mode=concurrency_mode, strategy=strategy, output_mode=output_mode)
            if df.size > 0: dfs.append(df)
            
        if not dfs: 
            raise KeyError(f"No data found for metric '{metric_name}'. Check metric name or topology.")
        
        keys = list(dfs[0].keys()) if hasattr(dfs[0], 'keys') else dfs[0].columns
        combined = {}
        for k in keys:
            combined[k] = ak.concatenate([d[k] for d in dfs])
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
