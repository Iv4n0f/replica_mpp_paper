import pickle
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

DATA_DIR = Path("data_vci")
FIGURES_DIR = Path("vci_figures")

map_size = {
    "1": "1B",
    "2": "2B",
    "4": "4B",
    "8": "8B",
    "16": "16B",
    "32": "32B",
    "64": "64B",
    "128": "128B",
    "256": "256B",
    "512": "512B",
    "1024": "1KB",
    "2048": "2KB",
    "4096": "4KB",
    "8192": "8KB",
    "16384": "16KB",
    "32768": "32KB",
    "65536": "64KB",
    "131072": "128KB",
    "262144": "256KB",
    "524288": "512KB",
    "1048576": "1MB",
    "2097152": "2MB",
    "4194304": "4MB",
    "8388608": "8MB",
    "16777216": "16MB",
    "33554432": "32MB",
    "67108864": "64MB",
    "134217728": "128MB",
    "268435456": "256MB",
    "536870912": "512MB",
    "1073741824": "1GB",
}

# Configurando o tamanho das fontes globalmente
plt.rcParams.update({
    'font.size': 12,          # Tamanho padrão da fonte
    'axes.titleweight': 'normal',  # Peso da fonte no título do gráfico
    'axes.titlesize': 12,     # Tamanho do título do gráfico
    'axes.labelsize': 12,     # Tamanho dos rótulos dos eixos
    'xtick.labelsize': 10,    # Tamanho das etiquetas no eixo X
    'ytick.labelsize': 10,    # Tamanho das etiquetas no eixo Y
    'legend.fontsize': 10,    # Tamanho da fonte na legenda
    'legend.title_fontsize': 12  # Tamanho do título da legenda
})

def load_data(bench_data: Path) -> tuple[pd.DataFrame, dict]:
    try:
        with bench_data.open("rb") as data:
            bench_dict = pickle.load(data)
            metadata = bench_dict["metadata"]
            df = pd.DataFrame(bench_dict["dataframe"]).T.drop_duplicates().T
            df.rename(columns={'lookups': 'lookup_size'}, inplace=True)
            return df, metadata
    except Exception as e:
        print(f"Error trying to open file {bench_data}: {e}")

def aggregate_data(df, by_columns:list[str]):
    # Group by 'Lookups' and calculate mean and std for 'Runtime'
    aggregated_df = df.groupby(by_columns).agg(
        runtime_mean=('runtime', 'mean'),
        runtime_std=('runtime', 'std'),
    ).reset_index()
    
    return aggregated_df

def aggregate_data_mb(df):
    # Group by 'Lookups' and calculate mean and std for 'Runtime'
    aggregated_df = df.groupby(['N', 'NumDevices']).agg(
        Runtime_mean=('Runtime', 'mean'),
        Runtime_std=('Runtime', 'std'),
    ).reset_index()
    
    return aggregated_df

@lru_cache(maxsize=2)
def generate_byte_mapping(start, end):
    byte_units = ['B', 'KB', 'MB', 'GB']
    mapping = {}

    # Gera potências de 2 de 1B até 128GB (2^37 bytes)
    for exp in range(start, end+1):
        value_in_bytes = 2 ** exp
        
        if value_in_bytes < 1024:
            # 1B até 1023B
            mapping[str(value_in_bytes)] = f"{value_in_bytes}B"
        elif value_in_bytes < 1024 ** 2:
            # 1KB até 1023KB
            value_in_kb = value_in_bytes // 1024
            mapping[str(value_in_bytes)] = f"{value_in_kb}KB"
        elif value_in_bytes < 1024 ** 3:
            # 1MB até 1023MB
            value_in_mb = value_in_bytes // (1024 ** 2)
            mapping[str(value_in_bytes)] = f"{value_in_mb}MB"
        else:
            # 1GB até 128GB
            value_in_gb = value_in_bytes // (1024 ** 3)
            mapping[str(value_in_bytes)] = f"{value_in_gb}GB"
    
    return mapping

def find_metg(group, x_data='task_granularity', y_data='efficiency', method:int = 1):
    group = group.sort_values(x_data)
    x = group[x_data].astype(float).values
    y = group[y_data].astype(float).values

    # Verifica se há cruzamento com 50%
    crosses_50 = (np.min(y) <= 50 <= np.max(y))

    if not crosses_50:
        return pd.Series({'metg': np.nan})


    if method == 1:
        # Método 1: interpolação cúbica y = f(x) + root-finding
        try:
            f = interp1d(x, y, kind='cubic', fill_value='extrapolate')

            def func(x_val):
                return float(f(x_val) - 50)

            for i in range(len(x) - 1):
                if (y[i] - 50) * (y[i + 1] - 50) < 0:
                    sol = root_scalar(func, bracket=[x[i], x[i + 1]], method='brentq')
                    if sol.converged:
                        return pd.Series({'metg': sol.root})
        except Exception as e:
            pass  # Falhou, tenta o fallback

    elif method == 2:
        # Método 2: interpolação linear invertida x = f(y)
        try:
            # Identifica dois pontos que cruzam 50%
            for i in range(len(y) - 1):
                y0, y1 = y[i], y[i + 1]
                if (y0 - 50) * (y1 - 50) < 0:
                    x0, x1 = x[i], x[i + 1]
                    # Interpolação linear entre os dois
                    metg = x0 + (50 - y0) * (x1 - x0) / (y1 - y0)
                    return pd.Series({'metg': float(metg)})
        except Exception as e:
            pass

    return pd.Series({'metg': np.nan})


@dataclass
class PlotMeta:
    x_axis: str
    y_axis: str
    hue: str
    style: str
    xlabel: str
    ylabel: str
    xticks: list
    yticks: list
    legend_title: str
    legend_labels: list


def generic_line_plot(dataset,
    title,
    output_file,
    plot_meta: PlotMeta,
    dashes: bool=False,
    markers=True,
    palette='tab10',
    size=None,
    sizes=None,
    xticks_rotation='horizontal'
):
    # Convert text width to inches (1 inch ≈ 2.54 cm)
    text_width_cm = 12  # Springer LNCS default
    width_inch = text_width_cm / 2.54  # ~4.72 inches
    height_inch = width_inch / 1.6  # ~2.95 inches (adjust ratio as needed)
    plt.figure(figsize=(width_inch, height_inch))

    line_plot = sns.lineplot(
        data=dataset,
        x=plot_meta.x_axis,
        y=plot_meta.y_axis,
        hue=plot_meta.hue,
        style=plot_meta.style,
        size=size,
        sizes=sizes,
        dashes=dashes,
        markers=markers,
        palette=palette,
    )

    line_plot.set_title(title)
    line_plot.set_xlabel(plot_meta.xlabel)
    line_plot.set_ylabel(plot_meta.ylabel)
    if len(plot_meta.xticks) > 0:
        line_plot.set_xticks(ticks=list(dataset[plot_meta.x_axis].unique()), labels=plot_meta.xticks, rotation=xticks_rotation)
    if len(plot_meta.yticks) > 0:
        line_plot.set_yticks(ticks=plot_meta.yticks)

    line_plot.grid(visible=True, linestyle='--')

    # Personalizar a legenda com labels customizadas
    handles, labels = line_plot.get_legend_handles_labels()
    if len(plot_meta.legend_labels) > 0:
        labels = plot_meta.legend_labels

    if len(plot_meta.legend_title):
        plt.legend(handles=handles, labels=labels, title=plot_meta.legend_title)

    plt.tight_layout()

    if (isinstance(output_file, Path)):
        plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi = 1200)
    else:
        plt.savefig(FIGURES_DIR/output_file, format='pdf', bbox_inches='tight', dpi = 1200)

def generic_multi_line_plot_err_bar(
    datasets,
    title,
    output_file,
    plots_meta: PlotMeta,
    dashes: bool=False,
    markers=True,
    xticks_rotation='horizontal',
    fifty_percent_line:bool = False,
    hlines: list[float] = None,
    vlines: list[float] = None,
    log_scale: bool = False,
    log_scale_y: bool = False,
    outside_legend: bool = False,
):
    # Convert text width to inches (1 inch ≈ 2.54 cm)
    # text_width_cm = 12  # Springer LNCS default
    # width_inch = text_width_cm / 2.54  # ~4.72 inches
    # height_inch = width_inch / 1.6  # ~2.95 inches (adjust ratio as needed)
    # plt.figure(figsize=(width_inch, height_inch))
    fig, ax = plt.subplots(figsize=(6, 4))

    plot_meta = plots_meta[0]
    dataset = datasets[0]

    line_plot = sns.lineplot(
        data=dataset,
        x=plot_meta.x_axis,
        y=plot_meta.y_axis,
        hue=plot_meta.hue,
        style=plot_meta.style,
        dashes=dashes,
        markers=markers,
        palette='tab10' if plot_meta.hue is not None else None,
        err_style="band",
        errorbar="ci",
        n_boot=1000,
        # err_kws={'elinewidth': 1.5, 'capsize': 5},
        ax=ax
    )

    line_plot.set_title(title)
    line_plot.set_xlabel(plot_meta.xlabel)
    line_plot.set_ylabel(plot_meta.ylabel)
    if len(plot_meta.xticks) > 0:
        line_plot.set_xticks(ticks=list(dataset[plot_meta.x_axis].unique()), labels=plot_meta.xticks, rotation=xticks_rotation)
    if len(plot_meta.yticks) > 0:
        line_plot.set_yticks(ticks=plot_meta.yticks)


    if fifty_percent_line:
        ymax = line_plot.get_ylim()[1]
        half_ymax = 50
        line_plot.axhline(y=half_ymax, color='#FF2200', linestyle='--')

    if hlines is not None:
        for hline in hlines:
            line_plot.axhline(y=hline, color='#FF2200', linestyle='--', linewidth=2, )

    if vlines is not None:
        for vline in vlines:
            line_plot.axvline(x=vline, color='#FF2200', linestyle='--', linewidth=2, )

    if log_scale:
        line_plot.set_xscale('log')
        line_plot.invert_xaxis()

    if log_scale_y:
        line_plot.set_yscale('log')

    if len(plots_meta) > 1:
        for i, dataset in enumerate(datasets[1:]):
            plot_meta_next = plots_meta[i + 1]
            ax.plot(
                dataset[plot_meta_next.x_axis],
                dataset[plot_meta_next.y_axis],
                linestyle='--',
                linewidth=2.5,
                color="#FF2200",
                label=None
            )

    line_plot.grid(visible=True, linestyle='--')

    # Personalizar a legenda com labels customizadas
    handles, labels = line_plot.get_legend_handles_labels()
    if len(plot_meta.legend_labels) > 0:
        labels = plot_meta.legend_labels

    if len(plot_meta.legend_title):
        if outside_legend:
            plt.legend(
                handles=handles, labels=labels,
                # title=plot_meta.legend_title,
                bbox_to_anchor=(1.05, 1),  # Move to the right
                loc='upper left',
                borderaxespad=0.
            )
        else:
            plt.legend(handles=handles, labels=labels, title=plot_meta.legend_title)


    plt.tight_layout()

    if (isinstance(output_file, Path)):
        plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=1200)
    else:
        plt.savefig(FIGURES_DIR/output_file, format='pdf', bbox_inches='tight', dpi=1200)

def generic_line_plot_err_bar(
    dataset,
    title,
    output_file,
    plot_meta: PlotMeta,
    dashes: bool=False,
    markers=True,
    xticks_rotation='horizontal',
    fifty_percent_line:bool = False,
    log_scale: bool = False,
    log_scale_y: bool = False,
    outside_legend: bool = False,
):
    # Convert text width to inches (1 inch ≈ 2.54 cm)
    # text_width_cm = 12  # Springer LNCS default
    # width_inch = text_width_cm / 2.54  # ~4.72 inches
    # height_inch = width_inch / 1.6  # ~2.95 inches (adjust ratio as needed)
    # plt.figure(figsize=(width_inch, height_inch))
    plt.figure(figsize=(6, 4))

    line_plot = sns.lineplot(
        data=dataset,
        x=plot_meta.x_axis,
        y=plot_meta.y_axis,
        hue=plot_meta.hue,
        style=plot_meta.style,
        dashes=dashes,
        markers=markers,
        palette='tab10' if plot_meta.hue is not None else None,
        err_style="band",
        errorbar="ci",
        n_boot=1000,
        # err_kws={'elinewidth': 1.5, 'capsize': 5},
    )

    line_plot.set_title(title)
    line_plot.set_xlabel(plot_meta.xlabel)
    line_plot.set_ylabel(plot_meta.ylabel)
    if len(plot_meta.xticks) > 0:
        line_plot.set_xticks(ticks=list(dataset[plot_meta.x_axis].unique()), labels=plot_meta.xticks, rotation=xticks_rotation)
    if len(plot_meta.yticks) > 0:
        line_plot.set_yticks(ticks=plot_meta.yticks)


    if fifty_percent_line:
        ymax = line_plot.get_ylim()[1]
        half_ymax = 50
        line_plot.axhline(y=half_ymax, color='red', linestyle='--')

    if log_scale:
        line_plot.set_xscale('log')
        line_plot.invert_xaxis()

    if log_scale_y:
        line_plot.set_yscale('log')

    line_plot.grid(visible=True, linestyle='--')

    # Personalizar a legenda com labels customizadas
    handles, labels = line_plot.get_legend_handles_labels()
    if len(plot_meta.legend_labels) > 0:
        labels = plot_meta.legend_labels

    if len(plot_meta.legend_title):
        if outside_legend:
            plt.legend(
                handles=handles, labels=labels,
                title=plot_meta.legend_title,
                bbox_to_anchor=(1.05, 1),  # Move to the right
                loc='upper left',
                borderaxespad=0.
            )
        else:
            plt.legend(handles=handles, labels=labels, title=plot_meta.legend_title)


    plt.tight_layout()

    if (isinstance(output_file, Path)):
        plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=1200)
    else:
        plt.savefig(FIGURES_DIR/output_file, format='pdf', bbox_inches='tight', dpi=1200)

def generic_bar_plot(dataset, title, output_file, plot_meta: PlotMeta):
    # Criar o gráfico de linhas
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(
        data=dataset,
        x=plot_meta.x_axis,
        y=plot_meta.y_axis,
        hue=plot_meta.hue,
        # style=plot_meta.style,
        palette='colorblind',
        errorbar=None,
        edgecolor='black'
    )

    hatches = ['x', '*', '+', 'o']

    rectangles = [patch for patch in bar_plot.patches if isinstance(patch, Rectangle)]

    num_bars_per_group = len(dataset['NumGPUs'].unique())
    # Itera e aplica hatches
    for i, thisbar in enumerate(rectangles):
        if (i >= len(dataset)):
            break
        thisbar.set_edgecolor('black')
        thisbar.set_linewidth(1.5)
        # Itera sobre o tamanho de grupos únicos no hue (Space Order)
        thisbar.set_hatch(hatches[i // num_bars_per_group])

    # Corrige a legenda com hatches corretos
    handles, labels = bar_plot.get_legend_handles_labels()
    for handle, hatch in zip(handles, hatches):
        handle.set_hatch(hatch)
    bar_plot.legend_.remove()


    bar_plot.grid(visible=True, linestyle='--')
    bar_plot.set_title(title)
    bar_plot.set_xlabel(plot_meta.xlabel)
    bar_plot.set_ylabel(plot_meta.ylabel)
    # bar_plot.set_xticks(ticks=plot_meta.xticks)
    if len(plot_meta.yticks) > 0:
        bar_plot.set_yticks(ticks=plot_meta.yticks)

    legend = plt.legend(handles=handles, labels=labels, title=plot_meta.legend_title)
    plt.setp(legend.get_title(), fontweight='bold')
    plt.savefig(FIGURES_DIR/output_file, bbox_inches='tight')


def plot_microbenchmark_vci(bench_df, title, output_file, x, y, xlabel, ylabel, legend_title):
    plot_meta = PlotMeta(
        x_axis=x,
        y_axis=y,
        hue='config',
        style='config',
        xlabel=xlabel,
        ylabel=ylabel,
        xticks=bench_df[x].unique(),
        yticks=[],
        legend_title=legend_title,
        legend_labels=[]
    )

    generic_line_plot(bench_df, title, output_file, plot_meta)

def plot_microbenchmark_vci_err_bar(bench_df, title, output_file, x, y, xlabel, ylabel, legend_title):
    plot_meta = PlotMeta(
        x_axis=x,
        y_axis=y,
        hue='config',
        style='config',
        xlabel=xlabel,
        ylabel=ylabel,
        xticks=bench_df[x].unique(),
        yticks=[],
        legend_title=legend_title,
        legend_labels=[]
    )

    generic_line_plot_err_bar(bench_df, title, output_file, plot_meta)

def plot_microbenchmark_err_bar(bench_df, title, output_file, x, y, xlabel, ylabel, legend_title=None):
    plot_meta = PlotMeta(
        x_axis=x,
        y_axis=y,
        hue='config',
        style='config',
        xlabel=xlabel,
        ylabel=ylabel,
        xticks=bench_df[x].unique(),
        yticks=[],
        legend_title=(legend_title or 'MPP Backend'),
        legend_labels=[]
    )

    generic_line_plot_err_bar(bench_df, title, output_file, plot_meta)

def plot_microbenchmark(bench_df, title, output_file, x, y, xlabel, ylabel, legend_title=None):
    plot_meta = PlotMeta(
        x_axis=x,
        y_axis=y,
        hue='config',
        style='config',
        xlabel=xlabel,
        ylabel=ylabel,
        xticks=bench_df[x].unique(),
        yticks=[],
        legend_title=(legend_title or 'MPP Backend'),
        legend_labels=[]
    )

    generic_line_plot(bench_df, title, output_file, plot_meta)

def plot_weak_scaling(bench_df, title, output_file):

    plot_meta = PlotMeta(
        x_axis='num_devices',
        y_axis='runtime_mean',
        hue='size',
        style='size',
        xlabel='Number of GPUs',
        ylabel='Execution Time (s)',
        xticks=bench_df['num_devices'].unique(),
        yticks=[],
        legend_title='Problem Size',
        # legend_labels=['small', 'large']
        legend_labels=[],
    )

    generic_line_plot(bench_df, title, output_file, plot_meta)

def plot_weak_scaling_norm(bench_df, title, output_file, norm_fact):
    df_runtime_norm_fact = bench_df[bench_df['num_devices'] == norm_fact].copy()
    df_runtime_norm_fact = df_runtime_norm_fact.rename(columns={'runtime_mean': 'runtime_norm_fact'})
    df_runtime_norm_fact = df_runtime_norm_fact.drop(columns=['name', 'num_devices', 'runtime_std'])
    df = pd.merge(bench_df, df_runtime_norm_fact, on=['lookup_size', 'size'], how='right')
    df['runtime_norm'] = df['runtime_mean']/df['runtime_norm_fact']

    y_max = int(np.ceil(df['runtime_norm'].max()))
    # yticks = [y for y in range(y_max+1)]

    yticks = np.linspace(0, y_max, 6)

    plot_meta = PlotMeta(
        x_axis='num_devices',
        y_axis='runtime_norm',
        hue='size',
        style='size',
        xlabel='Number of GPUs',
        ylabel='Normalized Runtime',
        xticks=bench_df['num_devices'].unique(),
        yticks=yticks,
        legend_title='Problem Size',
        # legend_labels=['small', 'large'],
        legend_labels=[],
    )

    generic_line_plot(df, title, output_file, plot_meta)

def plot_strong_scaling(bench_df, title, output_file):
    df_runtime_1dev = bench_df[bench_df['num_devices'] == 1].copy()
    df_runtime_1dev = df_runtime_1dev.rename(columns={'runtime_mean': 'runtime_1dev'})
    df_runtime_1dev = df_runtime_1dev.drop(columns=['name', 'num_devices', 'runtime_std'])
    df = pd.merge(bench_df, df_runtime_1dev, on=['lookup_size', 'size'], how='right')
    df['runtime_norm_1d'] = df['runtime_mean']/df['runtime_1dev']

    plot_meta = PlotMeta(
        x_axis='num_devices',
        y_axis='runtime_norm_1d',
        hue='lookup_size',
        style='lookup_size',
        xlabel='Number of GPUs',
        ylabel='Normalized Runtime',
        xticks=bench_df['num_devices'].unique(),
        yticks=[],
        legend_title='Lookups',
        legend_labels=['$10^5$', '$10^6$', '$10^7$', '$10^8$', '$10^9$', '$10^{10}$']
    )

    generic_line_plot(df, title, output_file, plot_meta)

def plot_strong_scaling_comp(bench_df, title, output_file):
    plot_meta = PlotMeta(
        x_axis='num_devices',
        y_axis='runtime_mean',
        hue='config',
        style='config',
        xlabel='Number of GPUs',
        ylabel='Execution Time (s)',
        xticks=bench_df['num_devices'].unique(),
        yticks=[],
        legend_title='Plugin',
        legend_labels=[],
    )

    generic_line_plot(bench_df, title, output_file, plot_meta)

def plot_miniwave_scaling(bench_df, title, output_file):
    plot_meta = PlotMeta(
        x_axis='NumGPUs',
        y_axis='runtime_mean',
        hue='space_order',
        style='space_order',
        xlabel='Number of GPUs',
        ylabel='Execution Time (s)',
        xticks=bench_df['NumGPUs'].unique(),
        yticks=[],
        legend_title='Space Order',
        legend_labels=[],
    )

    generic_bar_plot(bench_df, title, output_file, plot_meta)


def plot_massccs_scaling(bench_df, title, output_file):
    plot_meta = PlotMeta(
        x_axis='workers',
        y_axis='runtime_mean',
        hue='molecule',
        style='molecule',
        xlabel='Number of Nodes',
        ylabel='Execution Time (s)',
        xticks=bench_df['workers'].unique(),
        yticks=[],
        legend_title='Protein',
        legend_labels=[],
    )

    generic_line_plot(bench_df, title, output_file, plot_meta)

def plot_massccs_scaling_norm(bench_df, title, output_file):
    df_runtime_1dev = bench_df[bench_df['workers'] == 1].copy()
    df_runtime_1dev = df_runtime_1dev.rename(columns={'runtime_mean': 'runtime_1dev'})
    df_runtime_1dev = df_runtime_1dev.drop(columns=['gas', 'workers', 'runtime_std'])
    df = pd.merge(bench_df, df_runtime_1dev, on=['molecule'], how='right')
    df['runtime_norm_1d'] = df['runtime_mean']/df['runtime_1dev']

    plot_meta = PlotMeta(
        x_axis='workers',
        y_axis='runtime_norm_1d',
        hue='molecule',
        style='molecule',
        xlabel='Number of Nodes',
        ylabel='Normalized Runtime',
        xticks=bench_df['workers'].unique(),
        yticks=[],
        legend_title='Protein',
        legend_labels=[],
    )

    generic_line_plot(df, title, output_file, plot_meta)

def plot_massccs_speedup(bench_df, title, output_file):
    df_runtime_1dev = bench_df[bench_df['workers'] == 1].copy()
    df_runtime_1dev = df_runtime_1dev.rename(columns={'runtime_mean': 'runtime_1dev'})
    df_runtime_1dev = df_runtime_1dev.drop(columns=['gas', 'workers', 'runtime_std'])
    df = pd.merge(bench_df, df_runtime_1dev, on=['molecule'], how='right')
    df['speedup'] = df['runtime_1dev']/df['runtime_mean']

    plot_meta = PlotMeta(
        x_axis='workers',
        y_axis='speedup',
        hue='molecule',
        style='molecule',
        xlabel='Number of Nodes',
        ylabel='Speedup',
        xticks=bench_df['workers'].unique(),
        # xticks=[],
        yticks=[],
        legend_title='Protein',
        legend_labels=[],
    )

    generic_line_plot(df, title, output_file, plot_meta)

def plot_taskbench_scaling(bench_df, title, output_file):
    df_runtime_1dev = bench_df[bench_df['workers'] == 2].copy()
    df_runtime_1dev = df_runtime_1dev.rename(columns={'runtime_mean': 'runtime_1dev'})
    df_runtime_1dev = df_runtime_1dev.drop(columns=['name', 'workers', 'runtime_std'])
    df = pd.merge(bench_df, df_runtime_1dev, on=['type'], how='right')
    df['runtime_norm_1d'] = df['runtime_mean']/df['runtime_1dev']

    plot_meta = PlotMeta(
        x_axis='workers',
        y_axis='runtime_norm_1d',
        hue='type',
        style='type',
        xlabel='Number of Nodes',
        ylabel='Normalized Runtime',
        xticks=bench_df['workers'].unique(),
        yticks=[],
        legend_title='Dependence',
        legend_labels=[],
        # legend_labels=['$10^5$', '$10^6$', '$10^7$', '$10^8$', '$10^9$', '$10^{10}$']
    )

    generic_line_plot(df, title, output_file, plot_meta)

def plot_taskbench_scaling_comp(bench_df, title, output_file):
    plot_meta = PlotMeta(
        x_axis='ccr',
        y_axis='runtime_mean',
        hue='type',
        style='type',
        xlabel='ccr',
        ylabel='Execution Time (s)',
        xticks=bench_df['ccr'].unique(),
        yticks=[],
        legend_title='Dependence',
        legend_labels=[],
    )

    generic_line_plot(bench_df, title, output_file, plot_meta)

def plot_miniwave_scaling_comp(bench_df, title, output_file):
    plot_meta = PlotMeta(
        x_axis='num_devices',
        y_axis='runtime_mean',
        hue='space_order',
        style='space_order',
        xlabel='Number of GPUs',
        ylabel='Execution Time (s)',
        # xticks=bench_df['num_devices'].unique(),
        xticks=[],
        yticks=[],
        legend_title='Space Order',
        legend_labels=[],
    )

    generic_line_plot(bench_df, title, output_file, plot_meta)

def plot_miniwave_speedup(bench_df, title, output_file):
    df_runtime_1dev = bench_df[bench_df['num_devices'] == 1].copy()
    df_runtime_1dev = df_runtime_1dev.rename(columns={'runtime_mean': 'runtime_1dev'})
    df_runtime_1dev = df_runtime_1dev.drop(columns=['num_devices', 'runtime_std'])
    df = pd.merge(bench_df, df_runtime_1dev, on=['space_order'], how='right')
    df['speedup'] = df['runtime_1dev']/df['runtime_mean']

    plot_meta = PlotMeta(
        x_axis='num_devices',
        y_axis='speedup',
        hue='space_order',
        style='space_order',
        xlabel='Number of GPUs',
        ylabel='Speedup',
        xticks=bench_df['num_devices'].unique(),
        # xticks=[],
        yticks=[],
        legend_title='Space Order',
        legend_labels=[],
    )

    generic_line_plot(df, title, output_file, plot_meta)