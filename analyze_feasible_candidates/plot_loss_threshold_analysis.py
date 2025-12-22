import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
from matplotlib.ticker import NullFormatter

font = {'family' : 'times',
        'size'   : 11 }
rc('font', **font)


surrogate_loss            = np.asarray([0.2,    0.5,     1,     2,       5])
surrogate_feasible        = np.asarray([8092, 25160, 61274, 137821, 828227])
calphad_feasible_min_loss = np.asarray([6397, 15326, 14898, 13882,    8148])

nrows = 2
fig, axarr = plt.subplots( ncols=1,nrows=nrows, 
                           #figsize=(4,2.7*nrows), 
                           #figsize=(4,2.1*nrows), 
                           figsize=(4,1.9*nrows), 
                           height_ratios=[1.6,1],
                           sharex=True)

ax = 0 
if nrows > 1:
   ax = axarr[1]
   ax.set_ylabel('False positives (%)')
   ax.scatter(surrogate_loss, 100 - 100*calphad_feasible_min_loss/np.minimum(surrogate_feasible,2e4), c='k')
   ax.set_xlabel(r'Surrogate loss threshold $\tau_\text{loss}$ (%)')
   ax = axarr[0]
else:
   ax = axarr
   ax.set_xlabel(r'Surrogate loss threshold $\tau_\text{loss}$ (%)')



ax.scatter(surrogate_loss, surrogate_feasible,        c='b',             label='Surrogate screened')
ax.scatter(surrogate_loss, np.minimum(surrogate_feasible,2e4),                 label='Pre-clustered', facecolors='none',edgecolors='b')
ax.scatter(surrogate_loss, calphad_feasible_min_loss, c='g', marker='^', label='CALPHAD validated')
ax.set_yscale('log')
ax.set_ylabel('Number of compositions')
leg = ax.legend( #loc='lower right', 
                 loc='upper center', 
					  fontsize=10.5, 
                 #ncols=2, labelspacing=0.2, columnspacing=0.2, handletextpad=0.05, borderpad=0.2)
                 ncols=1, labelspacing=0.2, columnspacing=0.2, handletextpad=0.05, borderpad=0.2)

plt.draw() # Draw the figure so you can find the positon of the legend. 

# Get the bounding box of the original legend
bb = leg.get_bbox_to_anchor().transformed(ax.transAxes.inverted()) 

# Change to location of the legend. 
yOffset = 0.0#-0.03
yOffset = 0.01
bb.y0 += yOffset
bb.y1 += yOffset

xOffset = -0.15#16
bb.x0 += xOffset
bb.x1 += xOffset
leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

ax.set_ylim(top=1e6)

axarr_loop = axarr
if nrows == 1:
   axarr_loop = [axarr]

for ax in axarr_loop:
    ax.set_xticks([0,1,2,3,4,5])

ax.set_xlim(left=-0.05,right=5.2)

# --- arrows from solid -> hollow (cap at 20k) on TOP panel only ---
preclustered = np.minimum(surrogate_feasible, 2e4)
idxs = np.where(surrogate_loss > 0.6)[0]  # which points are capped (last two)

for i in idxs:
    x  = surrogate_loss[i]
    y0 = surrogate_feasible[i]     # solid blue (original)
    y1 = preclustered[i]           # hollow blue (capped)
    offset_x = 0.0 # 0.03
    offset_y = 3000
    axarr[0].annotate(
        '', xy=(x+offset_x, y1 + offset_y), xytext=(x+offset_x, y0),   # small x-offset to avoid marker overlap
        arrowprops=dict(
            arrowstyle='-|>',
            color='b',
            lw=1.5,
            shrinkA=0, shrinkB=0,
            mutation_scale=12,
            connectionstyle='arc3'
        ),
        zorder=4
    )



# --- add panel labels (a), (b) ---
axes = axarr if nrows > 1 else [axarr]
panel_labels = ['a)', 'b)', '(c)', '(d)'][:len(axes)]  # extend if needed
for i, a in enumerate(axes):
    a.text(0.015, 0.98, 
           panel_labels[i],
           transform=a.transAxes, ha='left', va='top',
           fontsize=14,# fontweight='bold',
           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))


xticks = axarr[1].get_xticks()           # or your own list
minor = (xticks[:-1] + xticks[1:]) / 2   # midpoints

for a in (axarr if nrows > 1 else [axarr]):
    a.set_xticks(minor, minor=True)
    a.xaxis.set_minor_formatter(NullFormatter())
    a.tick_params(axis='x', which='minor', bottom=True, length=3)

plt.tight_layout()
plt.subplots_adjust(hspace=0.1,left=0.14)

plt.savefig('loss_threshold_analysis.pdf')
plt.show()
