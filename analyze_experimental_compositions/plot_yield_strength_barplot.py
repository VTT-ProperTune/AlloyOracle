#!/usr/bin/env python3

# - SQS YS, Binary Cr-X D parameter, and SQS D.

# %% code cell 2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 10 

# ##### TC Nominals with old experimental sample labels 
# ```
# C2 = "55.0 15.0 0.0 10.0 15.0 5.0" # C10
# C11 = "60.0 5.0 0.0 20.0 0.0 15.0" # C7
# C1 = "77.5 5.0 0.0 0.0 5.0 12.5" # C5
# C9 = "65.0 7.5 7.5 0.0 5.0 15.0" # C8 
# C7 = "50.0 12.5 25.0 0.0 2.5 10.0" # C11
# C6 = "50.0 5.0 7.5 12.5 0.0 25.0" C4
# ```

# Ys the nominal compositions
ys_rom = {
    "C2": 2.542,
    "C11": 1.472,
    "C1": 1.459,
    "C9": 1.856,
    "C7": 2.333,
    "C6": 1.506,
}

ys_sqs = {
    "C2": 3.190,
    "C11": 1.770,
    "C1": 1.742,
    "C9": 2.337,
    "C7": 2.833,
    "C6": 1.605,
}

ys_expt = {
    'C2': 1.6833333333333333,
    'C11': 2.566666666666667,
    'C1': 2.4,
    'C9': 2.283333333333333,
    'C7': 2.2333333333333334,
    'C6': 2.2666666666666666,
}
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

mpl.rcParams['font.size'] = 12


keys = list(ys_expt.keys())   # ["C2","C11","C1","C9","C7","C6"]

def sort_key(new_label: str):
    # "C3.4" -> (3, 4)
    x_str, y_str = new_label[1:].split(".")
    return (int(x_str), int(y_str))

old_keys_sorted = sorted(remap_clusters.keys(), key=lambda k: sort_key(remap_clusters[k]))

def sort_key(new_label: str):
    x_str, y_str = new_label[1:].split(".")
    return (int(x_str), int(y_str))

# 1) sort old keys by ys_sqs (low -> high)
keys_sorted = sorted(ys_sqs, key=ys_sqs.get)   # e.g. ["C1","C6","C9","C11","C7","C2"] depending on ys_sqs

# 2) build x labels in the same order (new names)
labels_sorted = [remap_clusters[k] for k in keys_sorted]

# 3) build y arrays in the same order
y_expt   = np.array([ys_expt[k]   for k in keys_sorted], dtype=float)
y_sqs    = np.array([ys_sqs[k]    for k in keys_sorted], dtype=float)
y_rom    = np.array([ys_rom[k]    for k in keys_sorted], dtype=float)

x = np.arange(len(keys))

fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300, constrained_layout=True)
mpl.rcParams['font.size'] = 10.5

w = 0.18  # bar width (thin); try 0.14–0.22
ax.bar(x - 1.0*w, y_expt,   width=w, label="Expt.",         color="k")
ax.bar(x - 0.0*w, y_sqs,    width=w, label="SQS",           color="tab:blue")
ax.bar(x + 1.0*w, y_rom,    width=w, label="ROM",           color="tab:green")

ax.set_xticks(x)
ax.set_xticklabels(labels_sorted, rotation=0)
ax.set_ylabel("Yield strength (GPa)")
ax.set_xlabel("Sample")

labels_short = ["Expt.", "SQS", "ROM"]
handles = [
    plt.Rectangle((0,0),1,1, color="k"),
    plt.Rectangle((0,0),1,1, color="tab:blue"),
    plt.Rectangle((0,0),1,1, color="tab:green"),
]

ax.legend(handles, labels_short, ncol=3, frameon=False,
          loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=10.0,
          columnspacing=1.88, handlelength=1.5, handletextpad=0.2)

# leave room for the legend
fname = 'YS_vs_alloy.pdf'
print("Saving the experimental vs. predicted hardness bar plot figure as {fname}!")
fig.savefig(fname)
plt.show()
