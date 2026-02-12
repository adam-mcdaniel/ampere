import os
import arkouda as ak
import numpy as np
import re
from matplotlib import pyplot as plt
from ampere import Ensemble, MetricConfig, MetricType, AmpereSession

# 1. Configuración
configs = {
    re.compile(r".*rocm.*energy.*"): MetricConfig(MetricType.CUMULATIVE, scale_factor=1e-6),
    re.compile(r".*rocm.*power.*"): MetricConfig(MetricType.INSTANTANEOUS, scale_factor=1e-6),
}

def my_hpc_topology(metric_name, ranks):
    if 'device=0' in metric_name: return [r for r in ranks if r.name in ['MPI Rank 0', 'MPI Rank 1']]
    if 'device=2' in metric_name: return [r for r in ranks if r.name in ['MPI Rank 2', 'MPI Rank 3']]
    if 'device=4' in metric_name: return [r for r in ranks if r.name in ['MPI Rank 4', 'MPI Rank 5']]
    if 'device=6' in metric_name: return [r for r in ranks if r.name in ['MPI Rank 6', 'MPI Rank 7']]
    return ranks

def main():
    # Use context manager for clean connection handling
    with AmpereSession(server="localhost", port=5563):
        
        print("Loading traces...")
        # Use new helper (assuming standard structure)
        ensemble = Ensemble.load_hpl_trace("./examples/hpl-trace", ranks_per_node=8, metric_configs=configs)
        
        print("Computing attribution...")
        device_metrics = [f'A2rocm_smi:::energy_count:device={i}' for i in [0, 2, 4, 6]]
        ak_dfs = []

        for device in device_metrics:
            print(f"Attributing '{device}'...")
            df = ensemble.attribute(
                device,
                topology_resolver=my_hpc_topology,
                concurrency_mode='shared'
            )
            if len(df) > 0:
                ak_dfs.append(df)
        
        if not ak_dfs:
            print("No data found!")
            return

        # Combine results
        combined = {}
        keys = ak_dfs[0].columns
        for k in keys:
            combined[k] = ak.concatenate([df[k] for df in ak_dfs])
        
        ak_df_final = ak.DataFrame(combined)
        
        # Aggregate on Server
        print("Aggregating results...")
        g = ak.GroupBy([ak_df_final['Name'], ak_df_final['Rank']])
        keys, values = g.aggregate(ak_df_final['Value'], 'sum')
        
        # Retrieve Logic
        if hasattr(keys[0], 'to_ndarray'):
            names = keys[0].to_ndarray().tolist()
        else:
            names = keys[0].to_list()
            
        if hasattr(keys[1], 'to_ndarray'):
            ranks = keys[1].to_ndarray().tolist()
        else:
            ranks = keys[1].to_list()
            
        values = values.to_ndarray()
        
        # Basic Validation
        print("Top 5 consumers:")
        # Simple local aggregation to print top 5
        local_agg = {}
        for n, v in zip(names, values):
            local_agg[n] = local_agg.get(n, 0) + v
        
        sorted_consumers = sorted(local_agg.items(), key=lambda x: x[1], reverse=True)
        for n, v in sorted_consumers[:5]:
            print(f"{n}: {v}")

        # Plotting (Simplified)
        print("Generating heatmp...")
        # (Keeping manual plot for headless environment as Visualizer is interactive)
        all_names = sorted(list(set(names)))
        all_ranks = sorted(list(set(ranks)))
        name_to_idx = {n: i for i, n in enumerate(all_names)}
        rank_to_idx = {r: i for i, r in enumerate(all_ranks)}
        matrix = np.zeros((len(all_names), len(all_ranks)))
        
        for n, r, v in zip(names, ranks, values):
            matrix[name_to_idx[n], rank_to_idx[r]] = v
            
        # Filter Top 24
        row_sums = np.sum(matrix, axis=1)
        top_indices = np.argsort(row_sums)[::-1][:24]
        top_matrix = matrix[top_indices]
        top_names = [all_names[i] for i in top_indices]

        plt.figure(figsize=(12, 10))
        plt.imshow(top_matrix, aspect='auto', cmap="YlGnBu")
        plt.yticks(range(len(top_names)), top_names)
        plt.xticks(range(len(all_ranks)), all_ranks, rotation=45)
        plt.title("HPL Energy - Top 24")
        plt.tight_layout()
        plt.savefig("heatmap.png")
        plt.close()
        print("Done.")

if __name__ == "__main__":
    main()