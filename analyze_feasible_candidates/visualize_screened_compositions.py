import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rc
import matplotlib.patches as mpatches


from concurrent.futures import ProcessPoolExecutor, as_completed
import os

    # IMPORTANT: set backend inside worker (or before importing pyplot in main file)

sort_based_on = 'Nominal'

def percentage_formatter_x(x, pos):
    return f'{x * 100:.0f}'

def add_colorbox_legend_span_axes(fig, ax_ref, labels, colors,
                                  y_offset=0.01, fontsize=18,
                                  box_pad=0.15, lw=0.5):
    """
    Draw colored label boxes uniformly across the width of ax_ref.
    Call AFTER layout is finalized (after subplots_adjust / tight_layout).
    """
    pos = ax_ref.get_position()          # in figure coords
    left, right = pos.x0, pos.x1
    y = pos.y1 + y_offset
    n = len(labels)

    for i, (lab, col) in enumerate(zip(labels, colors)):
        x = left + (i + 0.5) / n * (right - left)
        fig.text(
            x, y, lab,
            transform=fig.transFigure,
            ha="center", va="bottom",
            fontsize=fontsize, # fontweight="bold",
            color='white', #_contrast_text(col),
            bbox=dict(
                boxstyle=f"square,pad={box_pad}",
                facecolor=col, edgecolor="k", linewidth=lw
            ),
        )

def add_colorbox_legend(ax, labels, colors, *,
                        y=1.06, pad_x=0.02, fontsize=12,
                        text_color='white', edgecolor="k", lw=2):
    """
    Draw a row of colored boxes with element labels (like your screenshot).

    y: y-position in axes coords (>1 puts it above the axis)
    pad_x: horizontal spacing in axes coords
    """

    n = len(labels)
    xs = np.linspace(0.03, 0.97, n)  # evenly spaced across axis

    for x, lab, col in zip(xs, labels, colors):
        ax.text(
            x, y, lab,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold",
            color=text_color,
            bbox=dict(boxstyle="square,pad=0.35", facecolor=col,
                      edgecolor=edgecolor, linewidth=lw),
            clip_on=False,
        )

rc('font', **{'family': 'times'})#, 'serif': ['Computer Modern']})
plt.rcParams['agg.path.chunksize'] = 200

input_ending = "nominal"


phase_labels_in_data = [ "LIQUID","A2", "B2", "OTHER_PHASES"]

phase_labels = [ "Liquid","A2", "B2", "Others"]

element_labels     = ['Cr',       "Co"      , "Al",       "Fe",       "Mn",       'V',       "Ni",       "Ti",       "Mo",       "W",       "Si"      ]
matrix_labels      = ['Cr_in_A2', 'Co_in_A2', 'Al_in_A2', 'Fe_in_A2', 'Mn_in_A2', 'V_in_A2', 'Ni_in_A2', 'Ti_in_A2', 'Mo_in_A2', 'W_in_A2', 'Si_in_A2']
precipitate_labels = ['Cr_in_B2', 'Co_in_B2', 'Al_in_B2', 'Fe_in_B2', 'Mn_in_B2', 'V_in_B2', 'Ni_in_B2', 'Ti_in_B2', 'Mo_in_B2', 'W_in_B2', 'Si_in_B2']


def process_one_nclu(NCLU, input_ending):

    fn = f"feasible_compositions_in_{NCLU}_alloy_clusters.xlsx"
    print(f"[pid {os.getpid()}] Processing {fn}")

    df = pd.read_excel(fn, engine='openpyxl')

    cols_needed = (
        element_labels
        + matrix_labels
        + precipitate_labels
        + phase_labels_in_data
        + ['kmeans_cluster']
    )

    df = df.dropna(subset=cols_needed)
    # Optional: also drop rows that are basically empty (sum ~ 0)
    df = df[
         df[element_labels].sum(axis=1) > 1e-2
    ]
    # Print the column names
    print(df.columns)
    
    x_n, x_m, x_p, phases = [], [], [], []
    cluster_sizes, cluster_names = [], []
    
    secondary = 'Al'  # choose
    
    step = 0.5
    nom_cols = element_labels  # your nominal element columns (Cr, Al, Co, ...)
    
    # round to nearest 0.5 at.%
    df[nom_cols] = (df[nom_cols] / step).round() * step
    
    groups = {int(k)-1: g for k, g in df.groupby('kmeans_cluster', sort=False)}
        
    for i in range(NCLU):
        cluster_names.append(f'C{i+1}')
        df_cluster = groups[i]   # O(1) lookup, no scan
        cluster_sizes.append(len(df_cluster))
        
        df_cluster_sorted_n = df_cluster.sort_values(by=['Cr', 'Co', 'Al', 'Fe', 'Mn', 'V'], ascending=[True,True,True,True, True, True], 
     	  														kind='mergesort')
        x_n.append(   df_cluster_sorted_n[element_labels      ].values)
        x_m.append(   df_cluster_sorted_n[matrix_labels       ].values)
        x_p.append(   df_cluster_sorted_n[precipitate_labels  ].values)
        phases.append(df_cluster_sorted_n[phase_labels_in_data].values)
                    
    x_n = np.vstack(x_n) # * 100
    x_m = np.vstack(x_m) * 100
    x_p = np.vstack(x_p) * 100
    phases = np.vstack(phases) * 100
    
    def visualize_stacked_barplot(compositions, element_labels, group_sizes, group_names, ylim=None, ylabel=None, ax=None, show_xticklabels=False, ysteps=10, show_minor_y=True):
        major_ticks = np.arange(ylim[0], ylim[1]+1, ysteps)
        if show_minor_y:
            minor_ticks = np.arange(ylim[0], ylim[1]+1, 5)
        element_colors = [c for i, c in enumerate(mcolors.TABLEAU_COLORS)]
        element_colors.append('silver')
        # x_positions = [x + 0.5 for x in range(compositions.shape[0])]
        x_positions = np.arange(compositions.shape[0]) + 0.5
        width = 1
        group_edge_idx = []
        group_middle_idx = []
        group_start_idx = 0
        for i in range(len(group_sizes)):
            groud_end_idx = group_start_idx + group_sizes[i]
            group_edge_idx.append(groud_end_idx + 0.5)
            group_middle_idx.append((groud_end_idx + group_start_idx) / 2.0)
            group_start_idx = groud_end_idx
        bottom_prev = 0
        for i in range(len(element_labels)):
            BAR_KW = dict(width=width, linewidth=0, edgecolor='none', antialiased=False)
            if i == 0:
                ax.bar(x_positions, compositions[:, i], width, label=element_labels[i], linewidth=0.1, color=element_colors[i])
            else:
                bottom = bottom_prev + compositions[:, i - 1]
                bottom_prev = bottom
                ax.bar(x_positions, compositions[:, i], bottom=bottom, label=element_labels[i], color=element_colors[i], **BAR_KW)
        for group_edge in group_edge_idx:
            linewidth = 1.0 # 0.5 # Black border separating clusters
            ax.axvline(x=group_edge - 0.5, linewidth=linewidth, color='k')
        
        ax.set_xticks(group_middle_idx)
        if show_xticklabels:
            ax.set_xticklabels(group_names)
        else:
            ax.set_xticklabels([])
            
        ax.set_yticks(major_ticks)
        if show_minor_y:
            ax.set_yticks(minor_ticks, minor=True)
        ax.set_ylabel(ylabel, labelpad=1.0) # default is 4.0
        ax.set_xlim(0, groud_end_idx)
        ax.set_ylim(ylim[0], ylim[1])

    fig_width = 3.5
    
    fig, axs = plt.subplots(nrows=3, ncols=1, 
                figsize=(fig_width, 4),
               )
    visualize_stacked_barplot(x_n, element_labels, cluster_sizes, cluster_names, ylim=[50, 100], ylabel='Nominal (at%)', ax=axs[0])
    visualize_stacked_barplot(x_m, element_labels, cluster_sizes, cluster_names, ylim=[50, 100],  ylabel='A2 comp. (at%)', ax=axs[1])
    visualize_stacked_barplot(x_p, element_labels, cluster_sizes, cluster_names, ylim=[0, 100],  ylabel='B2 comp. (at%)', ax=axs[2], show_xticklabels=True, ysteps=20) #, ysteps=10, show_minor_y=True)


    # Visualize elements
    element_colors = list(mcolors.TABLEAU_COLORS.values())
    element_colors.append("silver")
    colors = element_colors[:len(element_labels)]
    
    
    #y = 1.08 # slightly above boudnary
    y = 1.07
    fontsize=10

    colors = element_colors[:len(element_labels)]

    adjust_left = 0.14
    adjust_right = 0.98
    fig.subplots_adjust(hspace=0.15, top=0.90,left=adjust_left, right=adjust_right)   # try 0.25–0.40


    add_colorbox_legend_span_axes(fig, axs[0], element_labels, colors,
                                  y_offset=0.012, fontsize=fontsize, box_pad=0.2)

    plt.savefig(f'clustered_{NCLU}_compositions.png', dpi=400)
    plt.close('all')
    
    fig, axs = plt.subplots(nrows=1, ncols=1,
                 figsize=(fig_width, 0.9),
              )
    visualize_stacked_barplot(phases, phase_labels, cluster_sizes, cluster_names, ylim=[70, 100], ylabel='Phases (%)', ax=axs, show_xticklabels=True, ysteps=10, show_minor_y=True)
    fig.subplots_adjust(bottom=0.25,left=adjust_left, right=adjust_right)   # try 0.25–0.40

    fig.subplots_adjust(hspace=0.5)   # try 0.25–0.40
    plt.savefig(f'clustered_{NCLU}_phase_fractions.png', dpi=400)
    plt.close('all')


    return NCLU  # or return output paths if you want


if __name__ == "__main__":

    nclus_list = [3]

    max_workers = 1

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_one_nclu, nclu, input_ending) for nclu in nclus_list]
        for fut in as_completed(futures):
            nclu_done = fut.result()
            print(f"Finished NCLU={nclu_done}")

