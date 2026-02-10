import pandas as pd
import numpy as np
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.interpolate import interp1d
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union, Any, Literal, Callable, Pattern
from collections import defaultdict
from tqdm import tqdm
from dataclasses import dataclass

# ==========================================
# 1. Configuration & Types
# ==========================================

class MetricType(Enum):
    INSTANTANEOUS = 1  # Watts, Frequency
    CUMULATIVE = 2     # Joules, Bytes

@dataclass
class MetricConfig:
    kind: MetricType
    scale_factor: float = 1.0
    interpolation_kind: str = 'linear' 

TopologyResolver = Callable[[str, List['Rank']], List['Rank']]

# ==========================================
# 2. Math & Logic Core (Optimized)
# ==========================================

class Metric:
    def __init__(self, name: str, times: np.ndarray, values: np.ndarray, config: MetricConfig):
        self.name = name
        self.kind = config.kind
        
        # Enforce monotonicity (Assumes mostly sorted data for speed, fixes small jitters)
        if len(times) > 1 and times[-1] < times[0]:
             idx = np.argsort(times)
             times = times[idx]
             values = values[idx]
            
        self.times = times
        self.values = values * config.scale_factor
        self.t_min = times[0]
        self.t_max = times[-1]

        # Interpolation setup
        interp_kind = config.interpolation_kind
        if self.kind == MetricType.INSTANTANEOUS and interp_kind == 'linear':
            interp_kind = 'previous'
            
        self._interp = interp1d(self.times, self.values, kind=interp_kind, fill_value="extrapolate", assume_sorted=True)

    def get_delta_vectorized(self, t_starts: np.ndarray, t_ends: np.ndarray) -> np.ndarray:
        """
        Calculates deltas for many intervals at once. Much faster than loop.
        """
        # Clamp times
        t_s = np.maximum(t_starts, self.t_min)
        t_e = np.minimum(t_ends, self.t_max)
        
        # Zero out invalid intervals
        valid = t_e > t_s
        
        results = np.zeros_like(t_s)
        
        if self.kind == MetricType.CUMULATIVE:
            # Simple difference
            results[valid] = self._interp(t_e[valid]) - self._interp(t_s[valid])
        else:
            # Approximation for instantaneous: Average value * duration
            # (Exact integration requires complex segmented trapz, usually overkill for high-res traces)
            # Using midpoint approximation for speed in vector context
            midpoints = (t_s[valid] + t_e[valid]) / 2
            vals = self._interp(midpoints)
            results[valid] = vals * (t_e[valid] - t_s[valid])
            
        return results

class AttributionEngine:
    @staticmethod
    def _compute_coverage(starts, ends, breaks):
        """Vectorized segment coverage."""
        n = len(breaks) - 1
        mask = np.zeros(n, dtype=int)
        
        l_idx = np.searchsorted(breaks, starts, side='right') - 1
        r_idx = np.searchsorted(breaks, ends, side='left') - 1
        
        valid = r_idx > l_idx
        l_idx = l_idx[valid]
        r_idx = r_idx[valid]
        
        # Fast "painting" using diff array not needed for boolean mask, loop is okay for Ranks (usually small N)
        # But for strictly massive rank counts, we'd optimize this. 
        # For now, N_ranks is usually < 100 per node.
        for l, r in zip(l_idx, r_idx):
            l = max(0, l); r = min(n, r)
            if r > l: mask[l:r] = 1
        return mask

    @staticmethod
    def compute(
        metric: Metric,
        ranks: List['Rank'],
        concurrency_mode: Literal['shared', 'full'] = 'shared',
        strategy: Literal['inclusive', 'exclusive'] = 'inclusive',
        output_mode: Literal['quantity', 'rate'] = 'quantity'
    ) -> Dict[str, pd.DataFrame]:
        
        # 1. Define Global Timeline
        # Optimization: Only collect times that are actually within metric bounds
        all_times = [metric.times]
        for r in ranks:
            # Pre-filter times to avoid huge concatenation of irrelevant data
            s = r.starts[(r.starts >= metric.t_min) & (r.starts <= metric.t_max)]
            e = r.ends[(r.ends >= metric.t_min) & (r.ends <= metric.t_max)]
            all_times.append(s)
            all_times.append(e)
        
        merged = np.concatenate(all_times)
        breaks = np.unique(merged) # O(N log N) - unavoidable but heavily optimized in numpy
        
        if len(breaks) < 2:
            return {r.name: pd.DataFrame() for r in ranks}

        # 2. Calculate Resource per Segment (Vectorized)
        seg_starts = breaks[:-1]
        seg_ends = breaks[1:]
        
        deltas = metric.get_delta_vectorized(seg_starts, seg_ends)

        # 3. Concurrency
        active_counts = np.zeros(len(deltas), dtype=int)
        rank_masks = {} # Cache coverage masks
        
        for r in ranks:
            mask = AttributionEngine._compute_coverage(r.starts, r.ends, breaks)
            rank_masks[r.name] = mask
            active_counts += mask

        if concurrency_mode == 'shared':
            scaling = np.maximum(active_counts, 1)
            per_rank_resource = deltas / scaling
        else:
            per_rank_resource = deltas

        # --- OPTIMIZATION: Prefix Sums for O(1) Range Queries ---
        # accumulated_resource[i] = total resource from start up to segment i
        cum_resource = np.concatenate(([0.0], np.cumsum(per_rank_resource)))
        
        results = {}
        for r in ranks:
            if r.call_graph.empty or not np.any(rank_masks[r.name]):
                results[r.name] = pd.DataFrame()
                continue
            
            # Find which segments correspond to each call
            # l_idx: index of break just before/at start
            # r_idx: index of break just before/at end
            l_idx = np.searchsorted(breaks, r.starts, side='right') - 1
            r_idx = np.searchsorted(breaks, r.ends, side='left') - 1
            
            # Map segment indices to cumulative array indices
            # If call covers segments L through R-1 (indexes in `deltas`),
            # The sum is cum_resource[R] - cum_resource[L]
            
            # Note on Indices:
            # breaks: [t0, t1, t2] (len 3)
            # deltas: [d0, d1] (len 2)
            # cum_resource: [0, d0, d0+d1] (len 3)
            # Call t0->t2: l=0, r=1. We want d0+d1. 
            # R index in cum_resource needs to be r_idx + 1 -> 2. 
            # L index in cum_resource needs to be l_idx -> 0.
            # Result: cum[2] - cum[0] = d0+d1. Correct.
            
            # Vectorized Indexing
            # Clip to valid range
            l_valid = np.clip(l_idx, 0, len(cum_resource)-1)
            r_valid = np.clip(r_idx + 1, 0, len(cum_resource)-1)
            
            attributed = np.zeros(len(r.starts), dtype=float)
            mask_valid = r_idx > l_idx # Filter zero-length/inverted calls
            
            if strategy == 'inclusive':
                # --- FAST PATH: O(1) per call using Prefix Sum ---
                if np.any(mask_valid):
                    attributed[mask_valid] = (
                        cum_resource[r_valid[mask_valid]] - cum_resource[l_valid[mask_valid]]
                    )
                    
            elif strategy == 'exclusive':
                # --- SLOW PATH: Timeline Painting ---
                # Requires iteration or specialized logic.
                # To optimize, we still use per_rank_resource, but we assign ownership.
                # Numba would be ideal here, but sticking to numpy:
                
                # Create an ownership map for every segment
                segment_owners = np.full(len(per_rank_resource), -1, dtype=int)
                
                # "Paint" timeline. 
                # Optimization: Iterate only over calls that actually span time.
                # Since child calls usually come *after* parents in start-sorted lists, 
                # iterating in order works for "last write wins".
                
                # Unfortunately, Python loop is the only easy way without Numba/Cython
                # for strict hierarchical ownership.
                # We minimize work inside loop.
                for i in np.where(mask_valid)[0]:
                    li, ri = l_idx[i], r_idx[i]
                    # Clamp
                    li = max(0, li)
                    ri = min(len(segment_owners), ri)
                    if ri > li:
                        segment_owners[li:ri] = i
                
                # Aggregate
                # Sum per_rank_resource grouped by segment_owners
                valid_owners = segment_owners >= 0
                if np.any(valid_owners):
                    np.add.at(attributed, segment_owners[valid_owners], per_rank_resource[valid_owners])

            # Output
            df = r.call_graph.copy()
            if output_mode == 'rate':
                durations = df['Duration'].replace(0, np.nan)
                df['Value'] = attributed / durations
                df['Value'] = df['Value'].fillna(0.0)
            else:
                df['Value'] = attributed
                
            df['Metric'] = metric.name
            results[r.name] = df

        return results

# ==========================================
# 3. Data Structures
# ==========================================

class Rank:
    def __init__(self, node: str, name: str, df: pd.DataFrame):
        self.node = node
        self.name = name
        self.call_graph = df
        self.starts = self.call_graph['Start Time'].to_numpy()
        self.ends = self.call_graph['End Time'].to_numpy()
    def __repr__(self): return f"Rank({self.name})"

class Node:
    def __init__(self, name: str, metrics: List[Metric], ranks: List[Rank]):
        self.name = name
        self.ranks = ranks
        self.metrics = {m.name: m for m in metrics}

    def attribute(self, metric_name: str, topology_resolver: TopologyResolver, **kwargs) -> pd.DataFrame:
        if metric_name not in self.metrics: return pd.DataFrame()
        metric = self.metrics[metric_name]
        participating = topology_resolver(metric_name, self.ranks)
        if not participating: return pd.DataFrame()
        
        res = AttributionEngine.compute(metric, participating, **kwargs)
        
        dfs = []
        for r_name, df in res.items():
            if not df.empty:
                df['Rank'] = r_name
                dfs.append(df)
        if not dfs: return pd.DataFrame()
        combined = pd.concat(dfs)
        combined['Node'] = self.name
        return combined

class Run:
    def __init__(self, path: str, nodes: List[Node]):
        self.path = path
        self.name = os.path.basename(path)
        self.nodes = nodes

    @staticmethod
    def from_trace_path(path: str, node_ranks: Dict, metric_configs: Dict = {}) -> 'Run':
        e = Ensemble.from_trace_paths([path], node_ranks, metric_configs, max_workers=min(4, os.cpu_count()))
        if not e.runs: raise FileNotFoundError(f"Load failed: {path}")
        return e.runs[0]

    def attribute(self, metric_name: str, topology_resolver: TopologyResolver, **kwargs) -> pd.DataFrame:
        dfs = [n.attribute(metric_name, topology_resolver, **kwargs) for n in self.nodes]
        dfs = [d for d in dfs if not d.empty]
        if not dfs: return pd.DataFrame()
        combined = pd.concat(dfs)
        combined['Run'] = self.name
        return combined

# ==========================================
# 4. Loader Infrastructure (Optimized)
# ==========================================

def _resolve_config(name: str, config_map: Dict) -> MetricConfig:
    if name in config_map: return config_map[name]
    for k, v in config_map.items():
        if hasattr(k, 'match') and k.match(name): return v
        if isinstance(k, str) and k.startswith('^') and re.match(k, name): return v
    return MetricConfig(kind=MetricType.INSTANTANEOUS)

def _load_metrics_task(args):
    path, node, mpath, config_map = args
    if not os.path.exists(mpath): return path, node, []
    
    # Load entire table
    df = pd.read_csv(mpath, engine='c', dtype={'Metric Name': 'category', 'Value': 'float64', 'Time': 'float64'}, usecols=['Metric Name', 'Time', 'Value'])
    
    metrics = []
    # OPTIMIZATION: Groupby is much faster than repeated boolean indexing
    for m_name, group in df.groupby('Metric Name', observed=True):
        cfg = _resolve_config(m_name, config_map)
        # Assuming sorted by time usually, passing directly
        m = Metric(m_name, group['Time'].to_numpy(), group['Value'].to_numpy(), cfg)
        metrics.append(m)
        
    return path, node, metrics

def _load_callgraph_task(args):
    path, node, rank, cpath = args
    if not os.path.exists(cpath): return path, node, rank, pd.DataFrame()
    
    df = pd.read_csv(cpath, engine='c', dtype={
        'Name': 'string', 'Start Time': 'float64', 'End Time': 'float64', 'Depth': 'int32', 'Duration': 'float64'
    })
    df = df[df['End Time'] > df['Start Time']]
    return path, node, rank, df

class Ensemble:
    def __init__(self, runs: List[Run]):
        self.runs = runs
        
    @staticmethod
    def from_trace_paths(trace_paths: List[str], node_ranks: Dict, metric_configs: Dict = {}, max_workers: int = 32) -> 'Ensemble':
        m_tasks = []
        c_tasks = []
        for path in trace_paths:
            for node, ranks in node_ranks.items():
                m_tasks.append((path, node, os.path.join(path, f"{ranks[0]}_metrics.csv"), metric_configs))
                for rank in ranks:
                    c_tasks.append((path, node, rank, os.path.join(path, f"{rank}_Master_thread_callgraph.csv")))

        print(f"Loading {len(trace_paths)} runs on {max_workers} threads...")
        m_res, c_res = defaultdict(list), defaultdict(dict)
        
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            # Use as_completed to avoid blocking
            fut_m = [ex.submit(_load_metrics_task, t) for t in m_tasks]
            for f in tqdm(as_completed(fut_m), total=len(fut_m), desc="Metrics", leave=False):
                p, n, ms = f.result()
                if ms: m_res[(p, n)] = ms
            
            fut_c = [ex.submit(_load_callgraph_task, t) for t in c_tasks]
            for f in tqdm(as_completed(fut_c), total=len(fut_c), desc="Callgraphs", leave=False):
                p, n, r, d = f.result()
                if not d.empty: c_res[(p, n, r)] = d
                
        runs = []
        for path in trace_paths:
            nodes = []
            for node, ranks in node_ranks.items():
                ms = m_res.get((path, node), [])
                rs = [Rank(node, r, c_res[(path, node, r)]) for r in ranks if (path, node, r) in c_res]
                if rs: nodes.append(Node(node, ms, rs))
            if nodes: runs.append(Run(path, nodes))
        return Ensemble(runs)

    def attribute(self, metric_name: str, topology_resolver: TopologyResolver = lambda m, r: r, 
                  concurrency_mode: Literal['shared', 'full'] = 'shared',
                  strategy: Literal['inclusive', 'exclusive'] = 'inclusive',
                  output_mode: Literal['quantity', 'rate'] = 'quantity') -> pd.DataFrame:
        
        results = []
        print(f"Attributing '{metric_name}'...")
        for run in tqdm(self.runs):
            df = run.attribute(metric_name, topology_resolver, concurrency_mode=concurrency_mode, strategy=strategy, output_mode=output_mode)
            if not df.empty: results.append(df)
            
        if not results: return pd.DataFrame()
        combined = pd.concat(results, ignore_index=True)
        cols = ['Run', 'Node', 'Rank', 'Metric', 'Name', 'Value']
        others = [c for c in combined.columns if c not in cols]
        return combined[cols + others]

if __name__ == "__main__":
    configs = {re.compile(r".*energy.*"): MetricConfig(MetricType.CUMULATIVE, scale_factor=1e-6)}
    topo = {"Node0": ["Rank0", "Rank1"]}
    
    # Fast Load
    run = Run.from_trace_path("./examples/trace", topo, configs)
    
    # Fast Attribute
    df = run.attribute("rocm_smi:::energy_count:device0", lambda m,r: r, strategy='inclusive')
    print(df.head())