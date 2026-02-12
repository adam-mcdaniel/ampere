import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import arkouda as ak
import matplotlib.pyplot as plt
import seaborn as sns
import arkouda as ak
from . import MetricType

class Visualizer:
    @staticmethod
    def _fast_downsample(subset: pd.DataFrame, width_pixels=2000):
        """
        Versión NumPy Ultra-Rápida de downsampling.
        Reemplaza a pandas groupby con np.maximum.reduceat.
        Velocidad: ~100x más rápido que Pandas para datasets grandes.
        """
        # Extraer arrays numpy (copia cero si es posible)
        starts = subset['Start Time'].values
        ends = subset['End Time'].values
        values = subset['Value'].values
        depths = subset['Depth'].values
        names = subset['Name'].values # Si es categórico, usar .codes es más rápido, pero strings ok

        if len(starts) == 0: return pd.DataFrame()

        t_min, t_max = starts.min(), ends.max()
        if t_min == t_max: t_max += 1e-9
        
        # Calcular buckets
        scale = width_pixels / (t_max - t_min)
        # Vectorized binning
        pixel_indices = ((starts - t_min) * scale).astype(np.int32)
        
        # --- ESTRATEGIA: Procesar por Profundidad (Depth) ---
        # Como 'Depth' es discreto y pequeño (ej. 0-50), iterar profundidades 
        # y usar vectorización 1D es mucho más rápido que un groupby 2D complejo.
        
        aggregated_chunks = []
        unique_depths = np.unique(depths)
        
        for d in unique_depths:
            # Máscara booleana rápida
            mask = (depths == d)
            if not np.any(mask): continue
            
            p_d = pixel_indices[mask]
            s_d = starts[mask]
            e_d = ends[mask]
            v_d = values[mask]
            n_d = names[mask]
            
            # Para usar reduceat, los datos deben estar ordenados por la clave de agrupación (pixel)
            # Las trazas suelen venir ordenadas por tiempo, así que esto suele ser redundante,
            # pero lo hacemos por seguridad. Es muy rápido en arrays pre-ordenados.
            sort_idx = np.argsort(p_d)
            p_d = p_d[sort_idx]
            s_d = s_d[sort_idx]
            e_d = e_d[sort_idx]
            v_d = v_d[sort_idx]
            n_d = n_d[sort_idx]
            
            # Encontrar dónde cambia el pixel (índices de corte)
            # np.unique devuelve los índices del PRIMER elemento de cada grupo único
            unique_pixels, jump_indices = np.unique(p_d, return_index=True)
            
            # reduceat toma slices [jump[i] : jump[i+1]]
            # np.unique ya nos da exactamente lo que reduceat necesita (casi)
            # jump_indices sirve como los 'indices' para reduceat
            
            # Agregaciones ultrarrápidas en C
            agg_s = np.minimum.reduceat(s_d, jump_indices)
            agg_e = np.maximum.reduceat(e_d, jump_indices)
            agg_v = np.maximum.reduceat(v_d, jump_indices) # Max energía en el pixel
            
            # Para nombres, tomamos el primero del bin (o el más largo si quisiéramos complicarlo)
            agg_n = n_d[jump_indices]
            
            # Construir bloque
            chunk = pd.DataFrame({
                'Start Time': agg_s,
                'End Time': agg_e,
                'Value': agg_v,
                'Name': agg_n,
                'Depth': d
            })
            aggregated_chunks.append(chunk)
            
        if not aggregated_chunks: return pd.DataFrame()
        
        # Concatenar resultados livianos
        final = pd.concat(aggregated_chunks, ignore_index=True)
        final['VisualDuration'] = final['End Time'] - final['Start Time']
        return final

    @staticmethod
    def plot_flamegraph(
        df, 
        rank_filter: str, 
        metric_name: str, 
        title: str = "Energy Flame Graph",
        min_pixel_width: float = 0.5
    ):
        """
        Flamegraph usando Heatmap (Más robusto y rápido que Scattergl lines).
        Convierte los eventos en rectángulos visuales.
        """
        # Convert to pandas if input is Arkouda DataFrame
        if isinstance(df, ak.DataFrame):
            # Warning: Converting full dataframe to client might be heavy
            # Ideally filter first
            df = df.to_pandas()
            
        # 1. Filtrar Datos
        subset = df[df['Rank'] == rank_filter].copy()
        if subset.empty:
            print(f"No data found for rank {rank_filter}")
            return
        
        if 'Duration' not in subset.columns:
            subset['Duration'] = subset['End Time'] - subset['Start Time']

        # 2. Culling (Opcional pero recomendado)
        if not subset.empty:
            total_time = subset['End Time'].max() - subset['Start Time'].min()
            if total_time == 0: total_time = 1.0 
            time_threshold = total_time * (min_pixel_width / 4000.0)
            subset = subset[subset['Duration'] >= time_threshold]
        
        N = len(subset)
        if N == 0:
            print("No events large enough to render.")
            return

        print(f"Rendering {N} events using Optimized Rectangles...")

        screen_width = 2000 # pixels aprox
        t_min = subset['Start Time'].min()
        t_max = subset['End Time'].max()
        time_per_pixel = (t_max - t_min) / screen_width
        
        # Asignar cada evento a un pixel_bucket
        subset['pixel_idx'] = ((subset['Start Time'] - t_min) / time_per_pixel).astype(int)
        
        # Agrupar por (Depth, Pixel)
        # Tomamos el valor MAX de energía en ese pixel para no perder picos
        agg = subset.groupby(['Depth', 'pixel_idx']).agg({
            'Start Time': 'min',
            'End Time': 'max', # Extendemos para cubrir huecos
            'Value': 'max',    # Conservamos hotspots
            'Name': 'first'    # Nombre representativo
        }).reset_index()
        
        agg['Duration'] = agg['End Time'] - agg['Start Time']
        
        print(f"Downsampled to {len(agg)} visual elements (Manageable for SVG)")

        # 4. Render con go.Bar (Ahora seguro porque N es pequeño)
        fig = go.Figure()
        
        # Hover Text
        hover_text = (
            "<b>" + agg['Name'].astype(str) + "</b><br>" +
            "Depth: " + agg['Depth'].astype(str) + "<br>" +
            "Val: " + agg['Value'].apply(lambda x: f"{x:.4e}")
        )

        fig.add_trace(go.Bar(
            name=rank_filter,
            x=agg['Duration'],
            y=agg['Depth'],
            base=agg['Start Time'],
            orientation='h',
            marker=dict(
                color=agg['Value'],
                colorscale='Viridis',
                colorbar=dict(title=metric_name),
                line=dict(width=0) # Sin bordes para velocidad
            ),
            hoverinfo='text',
            hovertext=hover_text
        ))

        fig.update_layout(
            title=f"{title} - {rank_filter}",
            xaxis_title="Time (s)",
            yaxis_title="Stack Depth",
            template="plotly_white",
            height=600,
            xaxis=dict(rangeslider=dict(visible=True)),
            yaxis=dict(fixedrange=True, autorange="reversed"),
            bargap=0,
            bargroupgap=0
        )

        fig.show()

    @staticmethod
    def plot_node_view(
        attributed_df, 
        ranks: list, 
        metrics_data: list = [], 
        title: str = "Node Performance View"
    ):
        # Convert to pandas if input is Arkouda DataFrame
        if isinstance(attributed_df, ak.DataFrame):
            attributed_df = attributed_df.to_pandas()
            
        n_ranks = len(ranks)
        # Ajustar altura filas: Línea métricas 15%, resto equitativo
        row_heights = [0.15] + [0.85 / n_ranks] * n_ranks
        
        fig = make_subplots(
            rows=1 + n_ranks, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.01, # Espacio mínimo
            row_heights=row_heights,
            subplot_titles=["Node Metrics"] + ranks
        )

        # --- 1. DIBUJAR LÍNEAS (Optimizado con Decimación) ---
        print(f"Plotting metrics...")
        for m in metrics_data:
            if hasattr(m, 'name'): name, times, values = m.name, m.times, m.values
            else: name, times, values = m[0], m[1], m[2]
            
            # Arkouda array to numpy
            if isinstance(times, ak.pdarray): times = times.to_ndarray()
            if isinstance(values, ak.pdarray): values = values.to_ndarray()
            
            
            values = values - values.min() # Normalizar para mejor visualización
            # if hasattr(m, 'kind') and m.kind == MetricType.CUMULATIVE:
            #     # Subtract initial value to get incremental changes
            #     values = values - values[0]
            # elif hasattr(m, 'cfg') and m.cfg.kind == MetricType.CUMULATIVE: # Handle tuple from loader
            #     values = values - values[0]
            
            # OPTIMIZACIÓN LÍNEAS:
            # Si hay > 10k puntos, el navegador sufre. Decimamos visualmente.
            # Tomamos 1 de cada N puntos para mantener la forma general.
            limit = 1000
            if len(times) > limit:
                step = len(times) // limit
                times = times[::step]
                values = values[::step]
                
            fig.add_trace(
                go.Scattergl(
                    x=times, y=values, 
                    name=name, 
                    mode='lines',
                    line=dict(width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )

        # --- 2. DIBUJAR FLAMEGRAPHS (Optimizado con NumPy) ---
        # Pre-calcular escala de color global
        if not attributed_df.empty:
            cmin, cmax = attributed_df['Value'].min(), attributed_df['Value'].max()
        else:
            cmin, cmax = 0, 1
        
        # Iterar Ranks
        for i, rank_name in enumerate(ranks):
            # Filtrado rápido usando máscaras de numpy si fuera necesario, 
            # pero pandas booleano es ok aquí si el DF no es monstruoso.
            subset = attributed_df[attributed_df['Rank'] == rank_name]
            
            if subset.empty: continue
            
            # USAR EL NUEVO DOWNSAMPLER VECTORIZADO
            agg = Visualizer._fast_downsample(subset)
            
            if agg.empty: continue

            # Hover Text Vectorizado
            # Usar comprensión de lista es a menudo más rápido que .apply en strings
            hover_text = [
                f"<b>{n}</b><br>Depth: {d}<br>Val: {v:.4e}" 
                for n, d, v in zip(agg['Name'], agg['Depth'], agg['Value'])
            ]

            fig.add_trace(
                go.Bar(
                    x=agg['VisualDuration'],
                    y=agg['Depth'],
                    base=agg['Start Time'],
                    orientation='h',
                    marker=dict(
                        color=agg['Value'],
                        colorscale='Viridis',
                        cmin=cmin, cmax=cmax,
                        line=dict(width=0)
                    ),
                    text=hover_text,
                    hoverinfo='text',
                    showlegend=False,
                    name=rank_name
                ),
                row=i + 2, col=1
            )
            
            # Configurar Ejes Y (Invertido y Fijo)
            fig.update_yaxes(
                autorange="reversed", 
                fixedrange=True, 
                showticklabels=False, # Ocultar etiquetas numéricas de profundidad para limpiar
                title_text=rank_name,
                row=i+2, col=1
            )

        # --- 3. LAYOUT ---
        fig.update_layout(
            title=title,
            template="plotly_white",
            height=300 + (150 * n_ranks), # Más compacto
            xaxis=dict(
                rangeslider=dict(visible=False),
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor="black",
                spikethickness=1
            ),
            bargap=0,
            bargroupgap=0,
            hovermode='x unified',
            margin=dict(l=100, r=20, t=50, b=20)
        )
        
        # Sincronizar zoom X
        fig.update_xaxes(matches='x')
        
        fig.show()

    @staticmethod
    def plot_heatmap(df, title: str = "Metric Heatmap", cmap: str = "viridis", save_path: str = None, annot: bool = True, fmt: str = ".1f", sort_by: str = "Value", top_n: int = 24, aggregation_func='mean'):
        """
        Plots a heatmap of the metric values (e.g., Rates) for each Rank and Function.
        Aggregates by taking the mean value if multiple entries exist for a (Rank, Name) pair.
        """

        # Convert to Pandas for plotting if needed
        if isinstance(df, ak.DataFrame):
            pdf = df.to_pandas()
        else:
            pdf = df

        if 'Rank' not in pdf.columns:
            print("Warning: 'Rank' column missing for heatmap. Plotting aggregation over all data.")
            pdf['Rank'] = 'All'

        # Pivot: Index=Rank, Columns=Name, Values=Value (Mean)
        matrix = pdf.pivot_table(index='Name', columns='Rank', values='Value', aggfunc=aggregation_func)
        
        # Sort by row-wise mean across all ranks, then take top N
        matrix['_sort_key'] = matrix.mean(axis=1)
        matrix = matrix.sort_values(by='_sort_key', ascending=False).head(top_n)
        matrix = matrix.drop(columns=['_sort_key'])
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(matrix, cmap=cmap, annot=annot, fmt=fmt, linewidths=.5)
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Heatmap saved to {save_path}")
        else:
            plt.show()

    @staticmethod
    def plot_distribution(
        attributed_df, 
        metric_name: str, 
        bins: int = 50,
        title: str = "Metric Distribution"
    ):
        """
        Plots a histogram of the metric values. Useful for analyzing 
        distributions of rates or means over function calls.
        """
        if isinstance(attributed_df, ak.DataFrame):
            attributed_df = attributed_df.to_pandas()
            
        fig = go.Figure()
        
        # Group by Rank? Or global? 
        # Let's show global distribution first, maybe colored by Rank if needed.
        # Simple Histogram:
        
        fig.add_trace(go.Histogram(
            x=attributed_df['Value'],
            name='Global',
            nbinsx=bins,
            marker_color='#3366CC',
            opacity=0.75
        ))
        
        fig.update_layout(
            title=f"{title}: {metric_name}",
            xaxis_title=metric_name,
            yaxis_title="Count (Function Calls)",
            template="plotly_white",
            bargap=0.1
        )
        
        fig.show()
