import pytest
from ampere import Ensemble, Run, Node, Rank, MetricConfig, MetricType
import re
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

configs = {
    re.compile(r".*energy.*"): MetricConfig(MetricType.CUMULATIVE, scale_factor=1e-6),
    re.compile(r".*accel.*_power"):  MetricConfig(MetricType.INSTANTANEOUS, scale_factor=1.0)
}

# --- 2. TOPOLOGY LOGIC ---
# This is where you solve the GPU sharing problem.
# You define a function that looks at the metric name and the list of ranks
# and returns ONLY the ranks that should share that metric.
def my_hpc_topology(metric_name, ranks):
    if 'A2rocm_smi:::energy_count:device=0' in metric_name:
        return [r for r in ranks if r.name in ['MPI Rank 0', 'MPI Rank 1']]
    
    if 'A2rocm_smi:::energy_count:device=2' in metric_name:
        return [r for r in ranks if r.name in ['MPI Rank 2', 'MPI Rank 3']]
    
    if 'A2rocm_smi:::energy_count:device=4' in metric_name:
        return [r for r in ranks if r.name in ['MPI Rank 4', 'MPI Rank 5']]

    if 'A2rocm_smi:::energy_count:device=6' in metric_name:
        return [r for r in ranks if r.name in ['MPI Rank 6', 'MPI Rank 7']]
    
    
    if 'accel0_power' in metric_name:
        return [r for r in ranks if r.name in ['MPI Rank 0', 'MPI Rank 1']]
    
    if 'accel1_power' in metric_name:
        return [r for r in ranks if r.name in ['MPI Rank 2', 'MPI Rank 3']]
    
    if 'accel2_power' in metric_name:
        return [r for r in ranks if r.name in ['MPI Rank 4', 'MPI Rank 5']]

    if 'accel3_power' in metric_name:
        return [r for r in ranks if r.name in ['MPI Rank 6', 'MPI Rank 7']]
    
    return ranks

topo = {"Node0": ['MPI Rank 0', 'MPI Rank 1', 'MPI Rank 2', 'MPI Rank 3', 'MPI Rank 4', 'MPI Rank 5', 'MPI Rank 6', 'MPI Rank 7']}
ensemble = Ensemble.from_trace_paths(["./examples/hpl-trace"], topo, metric_configs=configs)
# Compute attribution using the custom topology
devices = [ensemble.attribute(
    device,
    topology_resolver=my_hpc_topology,
    concurrency_mode='shared',
    strategy='inclusive') for device in ['A2rocm_smi:::energy_count:device=0',
                 'A2rocm_smi:::energy_count:device=2',
                 'A2rocm_smi:::energy_count:device=4',
                 'A2rocm_smi:::energy_count:device=6']]

df = pd.concat(devices)
# Create Heatmap Data: Average Energy per Code Region per Rank
heatmap = df.pivot_table(index='Name', columns='Rank', values='Value', aggfunc='sum', fill_value=0)
# heatmap = heatmap[heatmap.mean().sort_values(ascending=False).index]
heatmap.sort_values(by=heatmap.columns.tolist(), ascending=False, inplace=True)
heatmap = heatmap.head(24)

print(heatmap)
sns.heatmap(heatmap, annot=True, fmt=".0f", cmap="YlGnBu")
plt.show()