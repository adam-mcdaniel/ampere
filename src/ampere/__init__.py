import arkouda as ak
import numpy as np 
import os
import re
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
    INSTANTANEOUS = 1
    CUMULATIVE = 2

@dataclass
class MetricConfig:
    kind: MetricType
    scale_factor: float = 1.0
    interpolation_kind: str = 'linear' 

TopologyResolver = Callable[[str, List['Rank']], List['Rank']]

# ==========================================
# 2. Math & Logic Core
# ==========================================

def ak_interp1d(x: ak.pdarray, y: ak.pdarray, xi: ak.pdarray, kind: str = 'linear') -> ak.pdarray:
    # 1. Indices
    idx = ak.searchsorted(x, xi)
    # 2. Clamp
    n = x.size
    idx = ak.where(idx < 1, 1, idx)
    idx = ak.where(idx >= n, n - 1, idx)
    # 3. Gather
    x0 = x[idx - 1]
    x1 = x[idx]
    y0 = y[idx - 1]
    y1 = y[idx]
    
    if kind == 'previous':
        return y0
    
    # 4. Linear
    run = x1 - x0
    rise = y1 - y0
    fraction = (xi - x0) / run
    fraction = ak.where(run == 0, 0.0, fraction)
    return y0 + (rise * fraction)

class Metric:
    def __init__(self, name: str, times: ak.pdarray, values: ak.pdarray, config: MetricConfig):
        self.name = name
        self.kind = config.kind
        
        # --- SAFE CASTING IN METRIC ---
        # Ensure we are working with Floats. If Strings came in, clean/cast them.
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
        t_s = ak.where(t_starts < self.t_min, self.t_min, t_starts)
        t_e = ak.where(t_ends > self.t_max, self.t_max, t_ends)
        valid = t_e > t_s
        
        val_start = ak_interp1d(self.times, self.cum_values, t_s, kind='linear')
        val_end   = ak_interp1d(self.times, self.cum_values, t_e, kind='linear')
        return ak.where(valid, val_end - val_start, 0.0)

    def get_statistics_vectorized(self, t_starts: ak.pdarray, t_ends: ak.pdarray) -> Dict[str, ak.pdarray]:
        """
        Computes vectorised statistics for each interval [start, end].
        Returns a dict of ak.pdarrays: 'min', 'max', 'mean', 'rate'.
        """
        # Clamp intervals
        t_s = ak.where(t_starts < self.t_min, self.t_min, t_starts)
        t_e = ak.where(t_ends > self.t_max, self.t_max, t_ends)
        valid = t_e > t_s
        
        # Rate / Mean (via Integration)
        deltas = self.get_delta_vectorized(t_s, t_e)
        durations = t_e - t_s
        safe_dur = ak.where(durations == 0, 1.0, durations)
        rates = deltas / safe_dur
        rates = ak.where(valid, rates, 0.0)
        
        # Min / Max (Approximation: Value at start, end, and internal peaks?)
        # Exact min/max over intervals in Arkouda is hard without segment reduction.
        # Approximation: Sample start and end points.
        # For true max, we'd need segment reduction on raw_values.
        # Let's populate with start/end values for now as a fast approximation.
        v_start = ak_interp1d(self.times, self.raw_values, t_s, kind='linear')
        v_end = ak_interp1d(self.times, self.raw_values, t_e, kind='linear')
        
        # If we pushed 'max' logic to server, it would be slow. 
        # For now, max = max(start, end).
        # TODO: Implement accurate segment reduction for min/max
        
        return {
            'mean': rates if self.kind == MetricType.CUMULATIVE else rates, # Mean of instantaneous IS the rate of the integral
            'rate': rates,
            'min': ak.where(v_start < v_end, v_start, v_end),
            'max': ak.where(v_start > v_end, v_start, v_end),
            'sum': deltas
        }

class AttributionEngine:
    @staticmethod
    def _compute_coverage_ak(starts: ak.pdarray, ends: ak.pdarray, breaks: ak.pdarray) -> ak.pdarray:
        l_idx = ak.searchsorted(breaks, starts, side='right') - 1
        r_idx = ak.searchsorted(breaks, ends, side='left') - 1
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
        
        # 1. Global Timeline
        time_arrays = [metric.times]
        for r in ranks:
            mask_s = (r.starts >= metric.t_min) & (r.starts <= metric.t_max)
            mask_e = (r.ends >= metric.t_min) & (r.ends <= metric.t_max)
            if mask_s.any(): time_arrays.append(r.starts[mask_s])
            if mask_e.any(): time_arrays.append(r.ends[mask_e])
            
        merged = ak.concatenate(time_arrays)
        breaks = ak.unique(merged)
        
        if breaks.size < 2:
            return {r.name: ak.DataFrame(dict()) for r in ranks}

        # Compute Base Quantities (Deltas) for Attribution
        deltas = metric.get_delta_vectorized(breaks[:-1], breaks[1:])
        
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
        max_idx = cum_resource.size - 1
        
        # Also compute stats if needed
        # We need raw metric stats for mean/min/max modes, UNWEIGHTED by concurrency?
        # Typically 'mean' implies the mean value of the metric during the interval.
        # Dealing with concurrency for 'mean' is tricky (shared mean?).
        # Let's assume output_mode != 'quantity' implies we want the raw metric property, attributed by time overlap.
        
        # However, the user request asks for "average values... during code regions".
        # This usually means "What was the average GPU power while function X was running?".
        # This is independent of how many other ranks were running.
        # BUT, if we just return the raw average, it duplicates data across ranks.
        # Let's stick to the 'Attribution' model:
        # 'quantity' = Energy (Joules) [already implemented]
        # 'rate' = Power (Watts) = Energy / Time [already implemented logic]
        # 'mean' = Same as rate for Cumulative. For Instantaneous, it's the average value.
        
        # Let's pre-compute raw stats on breaks if mode is special
        # But attribution logic splits the resource.
        # Use simpler logic: Compute attributed quantity first (Energy), then derive others.
        
        for r in ranks:
            l_idx = ak.searchsorted(breaks, r.starts, side='right') - 1
            r_idx = ak.searchsorted(breaks, r.ends, side='left') - 1
            
            idx_start = ak.where(l_idx < 0, 0, l_idx)
            idx_end = r_idx + 1
            idx_end = ak.where(idx_end > max_idx, max_idx, idx_end)
            mask_valid = idx_end > idx_start
            
            # Attributed Energy (or Integral)
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
    def __init__(self, node: str, name: str, df: Any):
        self.node = node
        self.name = name
        self.starts = df['Start Time']
        self.ends = df['End Time']
        self.names = df['Name']
        self.depths = df['Depth']
    def __repr__(self): return f"Rank({self.name})"

class Node:
    def __init__(self, name: str, metrics: List[Metric], ranks: List[Rank]):
        self.name = name
        self.ranks = ranks
        self.metrics = {m.name: m for m in metrics}

    def attribute(self, metric_name: str, topology_resolver: TopologyResolver, **kwargs) -> ak.DataFrame:
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
    def __init__(self, path: str, nodes: List[Node]):
        self.path = path
        self.name = os.path.basename(path)
        self.nodes = nodes

    @staticmethod
    def from_trace_path(path: str, node_ranks: Dict, metric_configs: Dict = {}) -> 'Run':
        return Ensemble.from_trace_paths([path], node_ranks, metric_configs).runs[0]

    def attribute(self, metric_name: str, topology_resolver: TopologyResolver, **kwargs) -> ak.DataFrame:
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
    def __init__(self, runs: List[Run]):
        self.runs = runs
    
    @staticmethod
    def _apply_filter_to_dict(df_dict, mask):
        """Helper to filter dict/DataFrame columns manually."""
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
                        import csv
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
                    import concurrent.futures
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
                                
                                # Identify Rank ID from path or group?
                                # We can use the Group column or path.
                                # The loop created paths based on ranks. 
                                # Let's extract rank ID from filename or just use Group.
                                # Using Group is safer if consistent.
                                # But we know the filename corresponds to a rank ID in the loop iteration.
                                # Wait, we lost the mapping from path -> r_id inside the future result.
                                # Let's resolve it.
                                
                                # Extract r_id from Group column (assuming homogenous file)
                                # Taking first element
                                group_val = result['Group'][0] if result['Group'] else "Unknown"
                                
                                # Or better, use the Group column from the dataframe if we want to support mixed (we don't for now)
                                loaded_ranks.append(Rank(node_name, group_val, c_df))
                                
                            except Exception as e:
                                print(f"Error transferring callgraph {path}: {e}")
                
                if loaded_ranks:
                    nodes.append(Node(node_name, metrics, loaded_ranks))
            if nodes: runs.append(Run(abs_path, nodes))
        return Ensemble(runs)

    @staticmethod
    def load_hpl_trace(dir_path: str, ranks_per_node: int = 8, metric_configs: Dict = {}) -> 'Ensemble':
        """
        Helper to load standard HPL traces where:
        - metrics are 'MPI Rank X_metrics.csv'
        - callgraphs are 'MPI Rank X_Master_thread_callgraph.csv'
        - Topology is Node0 -> Rank 0..N
        """
        ranks = [f"MPI Rank {i}" for i in range(ranks_per_node)]
        # Assuming single node for now or user maps dirs. 
        # But commonly we just want to load a directory as a "Node".
        # Let's assume the directory represents one or more nodes.
        # For simplicity, we create a basic topology.
        topo = {"Node0": ranks}
        return Ensemble.from_trace_paths([dir_path], topo, metric_configs)

        
    def attribute(self, metric_name: str, topology_resolver: TopologyResolver = lambda m, r: r, 
                  concurrency_mode: str = 'shared',
                  strategy: str = 'inclusive',
                  output_mode: str = 'quantity') -> ak.DataFrame:
        
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
