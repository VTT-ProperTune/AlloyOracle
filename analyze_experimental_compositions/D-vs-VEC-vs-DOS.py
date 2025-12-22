import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rc('font', family='serif', serif='Times New Roman')
plt.rc('mathtext', fontset='dejavuserif')

plt.rcParams.update({
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.titlesize": 8,
})

label_fontsize = 9

# --- Data ---
td_DOS_realized = {'C2.1': 0.6784, 'C4.1': 0.9415, 'C3.2': 0.9275,
                   'C3.1': 1.1152, 'C1.2': 1.1293, 'C1.1': 1.2420}
VEC_realized = {'C2.1': 5.708, 'C4.1': 6.069, 'C3.2': 6.264,
                'C3.1': 6.403, 'C1.2': 6.403, 'C1.1': 6.625}
D_realized = {'C2.1': 2.183, 'C4.1': 2.406, 'C3.2': 2.541,
              'C3.1': 2.731, 'C1.2': 3.007, 'C1.1': 3.152}

# Cr-X binary data
RiceD_binary = {'0-Cr': 2.137, '1-Al': 1.904, '2-Si': 1.938,
                '3-Ti': 1.904, '4-V': 2.081, '5-W': 2.118,
                '6-Mo': 2.053, '7-Mn': 2.198, '8-Fe': 2.331,
                '9-Co': 2.482, '10-Ni': 2.519}
valence_X = {"Al": 3, "Si": 4, "Ti": 4, "V": 5,
             "Cr": 6, "W": 6, "Mo": 6, "Mn": 7,
             "Fe": 8, "Co": 9, "Ni": 10}

# Sort keys
keys_realized = sorted(td_DOS_realized.keys())
td_dos_vals = [td_DOS_realized[k] for k in keys_realized]
D_vals = [D_realized[k] for k in keys_realized]
VEC_vals = [VEC_realized[k] for k in keys_realized]

figsize_X=3*0.92#0.88
figsize_Y=3*1.08
# --- Create figure: 2 rows, compact ---
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(figsize_X,figsize_Y), dpi=300,
                                        constrained_layout=False)

# =========================
# Top panel: Realized alloys
# =========================
sc = ax_top.scatter(VEC_vals, D_vals, c=td_dos_vals, cmap='viridis', s=25)
ax_top.set_xlabel(r"VEC (e$^-$/atom)", fontsize=8)
#ax_top.grid(True, ls='--', alpha=0.5)
ax_top.set_xlim(5.65, 6.68)
ax_top.set_ylim(2.0, 3.3)

# Pure Cr marker
#ax_top.scatter(6.0, 2.1, facecolors='none', edgecolors='blue', s=25, linewidths=1.0)
ax_top.scatter(6.0, 2.1, c=0.57,s=25) #facecolors='none', edgecolors='blue', s=25, linewidths=1.0)
ax_top.text(6.03, 2.1, "Pure Cr", color='k', fontsize=7, va='center', ha='left')

# Annotate alloys
offsets = {"C1.1": (-11, -2), "C1.2": (11, -2), "C3.1": (11, -2),
           "C3.2": (0, 5), "C4.1": (0, 5), "C2.1": (0, 5)}


# Remap to the new cluster names
remap_clusters = {
    'C2':  "C3.2",
    'C11': "C2.1",
    'C1':  "C3.4",
    'C9':  "C3.5",
    'C7':  "C1.1",
    'C6':  "C2.2"
}

new_labels =[ 
    "C3.2",
    "C2.1",
    "C3.4",
    "C3.5",
    "C1.1",
    "C2.2"
]

old_new = {
    "C2.1": "C3.2",
    "C4.1": "C2.1",
    "C3.2": "C3.4",
    "C3.1": "C3.5",
    "C1.2": "C1.1",
    "C1.1": "C2.2"
}

for k in keys_realized:
    dx, dy = offsets.get(k, (0, 6))
    ax_top.annotate( old_new[k],  #k, 
                    (VEC_realized[k], D_realized[k]),
                    textcoords="offset points", xytext=(dx, dy),
                    ha='center', fontsize=7, color='k')

# Colorbar above top plot
divider = make_axes_locatable(ax_top)
cax = divider.append_axes("top", size="8%", pad=0.05)
cbar = plt.colorbar(sc, cax=cax, orientation='horizontal')
cbar.set_label(r"$d$-DOS at $E_F$ (states/eV/atom)", fontsize=8)
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=8)

ax_top.tick_params(axis='both', labelsize=8)

# Panel label a)
ax_top.text(0.02, 0.92, "a)", transform=ax_top.transAxes,
            fontsize=10, va='top', ha='left')

# =========================
# Bottom panel: Cr-X binaries
# =========================
for key, value in RiceD_binary.items():
    element = key.split("-")[1]
    x_val = valence_X[element]
    ax_bottom.plot(x_val, value, marker='o', color='r',
                   markerfacecolor='none', markersize=4)
    # Annotate
    offset_dict = {"Al": (2, 4), "Si": (2, 4), "Ti": (6, -3), "V": (6, -4),
                   "W": (6, -6), "Mo": (8, -8), "Mn": (2, 4), "Fe": (2, 4),
                   "Co": (2, -9), "Ni": (2, -9), "Cr": (6, 4)}
    dx, dy = offset_dict.get(element, (0, 4))
    ax_bottom.annotate(element, (x_val, value),
                       textcoords="offset points", xytext=(dx, dy),
                       ha='center', fontsize=7)

ax_bottom.set_xlabel("Valence electrons of alloying element (e$^-$)")#, fontsize=9)
ax_bottom.set_xlim(2.5, 10.5)
ax_bottom.set_ylim(1.85, 2.57)
ax_bottom.tick_params(axis='both', labelsize=8)
ax_bottom.set_xticks(list(range(3, 11)))

for ax in [ax_top, ax_bottom]:
    ax.set_ylabel(r"Ductility $D$")

# Reference lines
ax_bottom.axhline(RiceD_binary["0-Cr"], color="k", linestyle="--", linewidth=0.8)
ax_bottom.axvline(valence_X["Cr"], color="k", linestyle="--", linewidth=0.8)

# Panel label b)
ax_bottom.text(0.02, 0.92, "b)", transform=ax_bottom.transAxes,
               fontsize=10, va='top', ha='left')

ax_bottom.set_yticks([1.9, 2.1, 2.3, 2.5])

# Shared ylabel
#fig.supylabel(r'Ductility parameter $D$', fontsize=8)
plt.tight_layout()

fname = 'D_vs_VEC_vs_DOS.pdf'
plt.savefig(fname, bbox_inches="tight", pad_inches=0.02)
print(f"Saved Rice ductiliy D parameter vs. VEC plot: {fname}!")

plt.show()
