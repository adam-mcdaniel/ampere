import arkouda as ak
import numpy as np 
import os
import re
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union, Any, Literal, Callable, Pattern
from collections import defaultdict
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    def get_delta_vectorized(self, t_starts: ak.pdarray, t_ends: ak.pdarray) -> ak.pdarray:
        t_s = ak.where(t_starts < self.t_min, self.t_min, t_starts)
        t_e = ak.where(t_ends > self.t_max, self.t_max, t_ends)
        valid = t_e > t_s
        
        val_start = ak_interp1d(self.times, self.cum_values, t_s, kind='linear')
        val_end   = ak_interp1d(self.times, self.cum_values, t_e, kind='linear')
        return ak.where(valid, val_end - val_start, 0.0)

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

        deltas = metric.get_delta_vectorized(breaks[:-1], breaks[1:])
        
        all_starts = ak.concatenate([r.starts for r in ranks])
        all_ends = ak.concatenate([r.ends for r in ranks])
        active_counts = AttributionEngine._compute_coverage_ak(all_starts, all_ends, breaks)

        if concurrency_mode == 'shared':
            scaling = ak.where(active_counts < 1, 1, active_counts)
            per_rank_resource = deltas / scaling.astype(ak.float64)
        else:
            per_rank_resource = deltas

        zeros = ak.zeros(1, dtype=ak.float64)
        cum_resource = ak.concatenate([zeros, ak.cumsum(per_rank_resource)])
        
        results = {}
        max_idx = cum_resource.size - 1
        
        for r in ranks:
            l_idx = ak.searchsorted(breaks, r.starts, side='right') - 1
            r_idx = ak.searchsorted(breaks, r.ends, side='left') - 1
            
            idx_start = ak.where(l_idx < 0, 0, l_idx)
            idx_end = r_idx + 1
            idx_end = ak.where(idx_end > max_idx, max_idx, idx_end)
            mask_valid = idx_end > idx_start
            
            vals = cum_resource[idx_end] - cum_resource[idx_start]
            attributed = ak.where(mask_valid, vals, 0.0)

            if output_mode == 'rate':
                durations = r.ends - r.starts
                safe_dur = ak.where(durations == 0, 1.0, durations)
                attributed = attributed / safe_dur
                attributed = ak.where(durations == 0, 0.0, attributed)

            res_data = {
                'Start Time': r.starts,
                'End Time': r.ends,
                'Name': r.names,
                'Depth': r.depths,
                'Value': attributed
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
            if len(df) > 0:
                df['Rank'] = ak.array([r_name] * len(df))
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
        dfs = [d for d in dfs if len(d) > 0]
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
    def _load_metrics_task(args):
        path, node, mpath, config_map = args
        if not os.path.exists(mpath): return path, node, []
        
        # Load entire table with pandas engine='c' for speed
        try:
            df = pd.read_csv(mpath, engine='c', dtype={'Metric Name': 'category', 'Value': 'float64', 'Time': 'float64'}, usecols=['Metric Name', 'Time', 'Value'])
        except Exception as e:
            print(f"Error reading metrics {mpath}: {e}")
            return path, node, []
        
        metrics_data = []
        for m_name, group in df.groupby('Metric Name', observed=True):
            cfg = Ensemble._resolve_config(m_name, config_map)
            # Return numpy arrays to avoid pickling/thread issues with Arkouda objects
            metrics_data.append((m_name, group['Time'].to_numpy(), group['Value'].to_numpy(), cfg))
            
        return path, node, metrics_data

    @staticmethod
    def _load_callgraph_task(args):
        path, node, rank, cpath = args
        if not os.path.exists(cpath): return path, node, rank, None
        
        try:
            df = pd.read_csv(cpath, engine='c', dtype={
                'Name': 'string', 'Start Time': 'float64', 'End Time': 'float64', 'Depth': 'int32', 'Duration': 'float64'
            })
            df = df[df['End Time'] > df['Start Time']]
            # Convert to dict of numpy arrays for cleaner transfer
            data = {
                'Name': df['Name'].to_numpy().astype(str),
                'Start Time': df['Start Time'].to_numpy(),
                'End Time': df['End Time'].to_numpy(),
                'Depth': df['Depth'].to_numpy().astype(np.int64),
                'Duration': df['Duration'].to_numpy()
            }
            return path, node, rank, data
        except Exception as e:
            print(f"Error reading callgraph {cpath}: {e}")
            return path, node, rank, None

    @staticmethod
    def from_trace_paths(trace_paths: List[str], node_ranks: Dict, metric_configs: Dict = {}, max_workers: int = 32) -> 'Ensemble':
        m_tasks = []
        c_tasks = []
        for path in trace_paths:
            abs_path = os.path.abspath(path)
            for node, ranks in node_ranks.items():
                m_tasks.append((abs_path, node, os.path.join(abs_path, f"{ranks[0]}_metrics.csv"), metric_configs))
                for rank in ranks:
                    c_tasks.append((abs_path, node, rank, os.path.join(abs_path, f"{rank}_Master_thread_callgraph.csv")))

        print(f"Loading {len(trace_paths)} runs on {max_workers} threads...")
        m_res = defaultdict(list)
        c_res = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            # Metrics
            fut_m = [ex.submit(Ensemble._load_metrics_task, t) for t in m_tasks]
            for f in tqdm(as_completed(fut_m), total=len(fut_m), desc="Metrics", leave=False):
                p, n, ms_data = f.result()
                if ms_data: m_res[(p, n)] = ms_data
            
            # Callgraphs
            fut_c = [ex.submit(Ensemble._load_callgraph_task, t) for t in c_tasks]
            for f in tqdm(as_completed(fut_c), total=len(fut_c), desc="Callgraphs", leave=False):
                p, n, r, d = f.result()
                if d is not None: c_res[(p, n, r)] = d
                
        runs = []
        for path in trace_paths:
            abs_path = os.path.abspath(path)
            nodes = []
            for node_name, ranks in node_ranks.items():
                # Convert loaded metric data to Arkouda objects
                metrics = []
                for m_name, t_np, v_np, cfg in m_res.get((abs_path, node_name), []):
                    try:
                        metrics.append(Metric(m_name, ak.array(t_np), ak.array(v_np), cfg))
                    except Exception as e:
                        print(f"Error creating Metric {m_name}: {e}")

                loaded_ranks = []
                for r_id in ranks:
                    if (abs_path, node_name, r_id) in c_res:
                        data = c_res[(abs_path, node_name, r_id)]
                        try:
                            # Create DataFrame from dict of numpy arrays -> Arkouda handles conversion
                            ak_dict = {k: ak.array(v) for k, v in data.items()}
                            loaded_ranks.append(Rank(node_name, r_id, ak.DataFrame(ak_dict)))
                        except Exception as e:
                             print(f"Error creating Rank {r_id}: {e}")

                if loaded_ranks:
                    nodes.append(Node(node_name, metrics, loaded_ranks))
            if nodes: runs.append(Run(abs_path, nodes))
        return Ensemble(runs)

    def attribute(self, metric_name: str, topology_resolver: TopologyResolver = lambda m, r: r, 
                  concurrency_mode: str = 'shared',
                  strategy: str = 'inclusive',
                  output_mode: str = 'quantity') -> ak.DataFrame:
        
        dfs = []
        print(f"Attributing '{metric_name}' on Arkouda Server...")
        for run in tqdm(self.runs):
            df = run.attribute(metric_name, topology_resolver, concurrency_mode=concurrency_mode, strategy=strategy, output_mode=output_mode)
            if len(df) > 0: dfs.append(df)
            
        if not dfs: return ak.DataFrame(dict())
        
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