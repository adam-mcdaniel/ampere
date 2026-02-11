import arkouda as ak
import seaborn as sns
from matplotlib import pyplot as plt
import re
import pandas as pd
from ampere import Ensemble, MetricConfig, MetricType

# 1. Conexión al servidor Arkouda
# Asegúrate de tener el servidor corriendo (ej: ./arkouda_server -nl 1)
ak.connect(server="localhost", port=5561)

# 2. Configuración
configs = {
    re.compile(r".*energy.*"): MetricConfig(MetricType.CUMULATIVE, scale_factor=1e-6),
    re.compile(r".*accel.*_power"): MetricConfig(MetricType.INSTANTANEOUS, scale_factor=1.0)
}

# 3. Lógica de Topología (Idéntica, opera sobre listas de objetos Rank en Python)
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

topo = {"Node0": [f'MPI Rank {i}' for i in range(8)]}

# 4. Carga (Los datos se quedan en el servidor)
print("Cargando trazas en Arkouda...")
ensemble = Ensemble.from_trace_paths(["./examples/hpl-trace"], topo, metric_configs=configs)

# 5. Atribución (Cálculo en el servidor)
print("Calculando atribución...")
device_metrics = [
    'A2rocm_smi:::energy_count:device=0',
    'A2rocm_smi:::energy_count:device=2',
    'A2rocm_smi:::energy_count:device=4',
    'A2rocm_smi:::energy_count:device=6'
]

# Obtenemos una lista de ak.DataFrames
ak_dfs = [
    ensemble.attribute(
        device,
        topology_resolver=my_hpc_topology,
        concurrency_mode='shared',
        strategy='inclusive'
    ) for device in device_metrics
]

# Filtrar resultados vacíos
ak_dfs = [d for d in ak_dfs if d.size > 0]

if not ak_dfs:
    print("No se encontraron datos para los dispositivos especificados.")
else:
    # 6. Concatenación en Arkouda
    # ak.DataFrame no tiene un 'concat' directo simple como Pandas para lista de dicts,
    # así que concatenamos las columnas manualmente.
    combined = {}
    keys = ak_dfs[0].columns
    for k in keys:
        # Concatenamos los arrays distribuidos de cada columna
        combined[k] = ak.concatenate([df[k] for df in ak_dfs])
    
    ak_df_final = ak.DataFrame(combined)

    # 7. Agregación (Group By) en el Servidor
    # Queremos pivotar, pero Arkouda no tiene pivot_table directo.
    # Hacemos la agregación (Sum Value por Name y Rank) en el servidor para reducir datos.
    # GroupBy multiclave:
    by_keys = [ak_df_final['Name'], ak_df_final['Rank']]
    g = ak.GroupBy(by_keys)
    
    # Sumar valores
    keys, values = g.aggregate(ak_df_final['Value'], 'sum')
    
    # 8. Transferencia al Cliente (Solo los datos agregados, muy pequeños)
    print("Transfiriendo datos agregados al cliente...")
    df_agg = pd.DataFrame({
        'Name': keys[0].to_ndarray(),
        'Rank': keys[1].to_ndarray(),
        'Value': values.to_ndarray()
    })

    # 9. Visualización (Pandas/Seaborn estándar)
    # Ahora 'df_agg' es un DataFrame de pandas normal y pequeño.
    heatmap = df_agg.pivot_table(index='Name', columns='Rank', values='Value', fill_value=0)
    
    # Ordenar y filtrar
    # Ordenar por la suma de la fila para ver los kernels más pesados arriba
    heatmap['total'] = heatmap.sum(axis=1)
    heatmap.sort_values('total', ascending=False, inplace=True)
    heatmap.drop(columns='total', inplace=True)
    
    heatmap = heatmap.head(24)

    print(heatmap)
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title("HPL Energy Consumption (Joules) - Top 24 Functions")
    plt.savefig("heatmap.png")
    plt.close()

# Opcional: Desconectar
ak.disconnect()
print("Done.")