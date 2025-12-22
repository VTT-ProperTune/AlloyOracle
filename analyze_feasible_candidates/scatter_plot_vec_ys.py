import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 11 

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from scipy.stats import gaussian_kde

import sys
sys.path.insert(1, '../helper_functions')
import calculate_ys

def add_hist_above(ax, x, bins=25, height="18%", pad=0.0, #pad=0.05,
                   show_mean=True, mean_fmt="{:.1f}", fontsize=9):
    """
    Adds a small histogram axis ABOVE `ax` (outside the plotting area),
    sharing x-limits with `ax`. Returns the new hist axis.
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None

    # Create a new axes above the main one
    divider = make_axes_locatable(ax)
    axh = divider.append_axes("top", size=height, pad=pad, sharex=ax)

    # Histogram (uses same x-range as the scatter axis)
    axh.hist(x, bins=bins)
    # Keep the limits locked (in case autoscale tries anything)
    xmin, xmax = ax.get_xlim()
    axh.set_xlim(xmin, xmax)

    # Cosmetics: no x tick labels (main axis shows them)
    axh.tick_params(axis="x", labelbottom=False, bottom=False)
    axh.tick_params(axis="y", left=False, labelleft=False)
    axh.set_ylabel("")  # keep it clean

    if show_mean:
        mu = float(x.mean())
        axh.axvline(mu, lw=1.2, color='black')
        t = axh.text( 0.98, 0.85, r"$\mu$=" + mean_fmt.format(mu),
                      transform=axh.transAxes, ha="right", va="top",
                      fontsize=fontsize,
                      bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', pad=0.01),
				    	  )

    return axh

# Load the Excel file
NCLU = 3 

fn = f'feasible_compositions_in_{NCLU}_alloy_clusters.xlsx'

df = pd.read_excel(fn)
strength_label = 'ys_298K'
misfit_label   = 'misfit_delta'
vec_label = 'VEC_m'


x = [df["Cr_in_A2"],
     df["Al_in_A2"],
     df["Fe_in_A2"],
     df["Ni_in_A2"],
     df["Ti_in_A2"],
     df[ "V_in_A2"],
     df["Mo_in_A2"],
     df[ "W_in_A2"],
     df["Mn_in_A2"],
     df["Si_in_A2"],
     df["Co_in_A2"]]


x = np.asarray(x).T 
if np.sum(x[0,:]) < 1.01:
   x *= 100. # convert to percentages

ys_ref, ys_data = calculate_ys.CalculateYieldStrength(x)

df[vec_label]        = ys_data['VEC_m']
df[strength_label] = ys_data[strength_label]
df[misfit_label]   = ys_data['misfit_delta']

mpl.rcParams['font.size'] = 14 

fig, ax = plt.subplots(figsize=(4.5,3.))#(4,3))
vec      = df[vec_label]
strength = df[strength_label]

field = df['kmeans_cluster']

n_levels = NCLU
cmap = plt.get_cmap('viridis', n_levels)
bounds = np.linspace(np.min(field) + 0.5, np.max(field) + 0.5, n_levels)
vmin = np.min(field)
vmax = np.max(field)


indices = np.argsort(field)
ax.set_xlabel(r'A2 VEC (e$^–$/atom)')
ax.set_ylabel(r'A2 strength (GPa)')


# Create bounds for 16 discrete values (1 to 16)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Create opaque colorbar (by copying colormap and setting alpha=1)
opaque_cmap = cmap(np.linspace(0, 1, cmap.N))
opaque_cmap[:, -1] = 1.0  # Remove transparency
cmap_no_alpha = mpl.colors.ListedColormap(opaque_cmap)

field_name = 'Cluster ID'

s = 60 # 13
alpha = 0.9 # 0.5
marker = '.'

if 'Cluster' in field_name:
  num_colors = NCLU #
  bounds = np.arange(1, num_colors + 2)  # 1 to 17
  norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=num_colors)
  sc = ax.scatter( vec[indices], strength[indices], c=field[indices], s=s, alpha=alpha, marker=marker, cmap=cmap, edgecolors='none', norm=norm  )
  sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_no_alpha)

ticks_cbar = np.linspace(1.5,NCLU+0.5,NCLU)

ticklabels_cbar = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10',
                   'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20']

ticklabels_cbar = ticklabels_cbar[:NCLU]

cbar = fig.colorbar(
       sm, ax=ax, ticks=ticks_cbar,
       orientation='horizontal', location='top', #xticklabels=ticklabels_cbar,
       pad=0.02,        # gap between axes and colorbar (in figure fraction units)
       fraction=0.03,   # thickness of the bar (relative to ax height)
       )

cbar.ax.set_xticklabels(ticklabels_cbar)

cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')

plt.subplots_adjust(left=0.1, bottom=0.23, right=0.99, top=0.87)
fname = 'strength_vs_VEC_vs_ID.png'
print(f"Saving the figure for A2 VEC vs. A2 yield strength, colored by cluster ID: {fname}!")
fig.savefig( fname, dpi=300)
plt.show()
mpl.rcParams['font.size'] = 11 


panel_names = [       'a',             'b',             'c',            'd',           'e',           'f']

fields = ['Al_in_A2', 'Co_in_A2', 'Fe_in_A2', 'V_in_A2','Mn_in_A2', misfit_label]

fig_vec = 0
axarr_vec = 0
fig_strength = 0
axarr_strength = 0


fig, axarr = plt.subplots(ncols=2, nrows=3, figsize=(4,6), sharex=True, sharey=True)
axes_loop = axarr

i_ax = -1
for ax_loop in axes_loop.ravel():
    i_ax += 1
    if i_ax >= len(fields):
       break

    j_ax = 0
    if i_ax > 3:
      j_ax = 2
    elif i_ax > 1:
      j_ax = 1
    ax = axarr[j_ax, i_ax%2]
       

    field_name = fields[i_ax]
    if field_name == 'blank':
       continue
    field = df[field_name]

    cmap = plt.get_cmap('viridis') #, n_levels)


    # Define normalization with boundaries
    bounds = np.linspace(np.min(field), np.max(field)) #, n_levels + 1)

    vmin = np.min(field)
    vmax = np.max(field)

    if "A2" == field_name:
       field *= 100 # scale to 100%
       field_name = 'BCC (vol%)'
       bounds = np.linspace(np.min(field), np.max(field)) #, n_levels + 1)
    elif "B2" == field_name:
       field *= 100 # scale to 100%
       field_name = 'B2 fraction (%)'
       vmin = np.min(field)
       vmax = np.max(field)
       bounds = np.linspace(np.min(field), np.max(field)) #, n_levels + 1)
    elif '_in_A2' in field_name:
       field_name = field_name.split('_')[0] + ' in A2 (at%)'
       field *= 100 # scale to 100%
       vmin = np.min(field)
       vmax = np.max(field)
       bounds = np.linspace(np.min(field), np.max(field)) #, n_levels + 1)
    elif field_name in ['Al', 'Co',  'Fe', 'V','Mn', 'W', 
                        'Ni', 'Mo',  'Si', 'Ti', 'Cr',   ]:
       field_name = f'Nom. {field_name} (at%)'
    elif 'cluster' in field_name:
       alpha = 0.1
       field_name = 'Cluster ID'
       cluster_ids = field
       n_levels = NCLU
       cmap = plt.get_cmap('viridis', n_levels)
       bounds = np.linspace(np.min(field), np.max(field), n_levels + 1)
       vmin = np.min(field)
       vmax = np.max(field)
    elif 'misfit' in field_name:
       field_name = r'A2 misfit $\delta$ (%)'

    print(np.min(field), "<=", field_name, "<=", np.max(field), "; average", np.mean(field))
    indices = np.argsort(field)

    s = 25 # 13
    alpha = 0.5#0.6
    marker = '.'

    # Create bounds for 16 discrete values (1 to 16)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Create opaque colorbar (by copying colormap and setting alpha=1)
    opaque_cmap = cmap(np.linspace(0, 1, cmap.N))
    opaque_cmap[:, -1] = 1.0  # Remove transparency
    cmap_no_alpha = mpl.colors.ListedColormap(opaque_cmap)

   
    if i_ax < len(panel_names): #else:
       if 'Cluster' in field_name:
         num_colors = NCLU# 16
         bounds = np.arange(1, num_colors + 2)  # 1 to 17
         norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=num_colors)
         sc = ax.scatter( vec[indices], strength[indices], c=field[indices], s=s, alpha=alpha, marker=marker, cmap=cmap, edgecolors='none', norm=norm  )
         sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_no_alpha)
       else:
         norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
         sc = ax.scatter(vec[indices], strength[indices], c=field[indices],
                         s=s, alpha=alpha, marker=marker,
                         cmap=cmap, edgecolors='none', norm=norm)
         sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_no_alpha)
   
         # Add colorbar above each subplot
         if i_ax < len(panel_names):
            cbar = fig.colorbar(
                sm, ax=ax, orientation='horizontal', location='top',
                fraction=0.20,   # thickness of the bar (relative to ax height)
                pad=0.03,        # gap between axes and colorbar (in figure fraction units)
            )

       cbar.ax.xaxis.set_ticks_position('top')
       cbar.ax.xaxis.set_label_position('top')
       cbar.ax.tick_params(axis='x', which='both', pad=0)   # try 0, -1, -2, -3
       cbar.set_label(field_name, labelpad=4)

       ax.text(0.015, 0.97, # left corner
               f"{panel_names[i_ax]})",
               transform=ax.transAxes, ha='left', va='top',
               fontsize=12) #, fontweight='bold')

    
for ax in axarr[:,0]:
    ax.set_ylabel('A2 strength (GPa)')

for ax in axarr[-1,:]:
    ax.set_xlabel(r'A2 VEC (e$^-$/atom)')

fig.tight_layout()

fig.subplots_adjust(wspace=0.14) #bottom=0.1, right=0.8, top=0.9)

fname = 'vec_strength_property_scatter.png'
print(f"Saving the figure for A2 VEC vs. A2 yield strength, colored by key elements and misfit: {fname}!")
fig.savefig( fname, dpi=300)


plt.show()
    
