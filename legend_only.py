import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# === Create legend handles ===
patch_abc   = mpatches.Patch(color='#b0c4ff', label='B < 0 (ABC-DS)', alpha=0.7)
patch_nnds  = mpatches.Patch(color='#cdebc5', label='B < 0 (S2-NNDS)', alpha=0.7)
patch_inter = mpatches.Patch(color='#b074ff', label='B < 0 (Intersection)', alpha=0.7)
patch_unsafe= mpatches.Patch(color='red', label='Unsafe Set', alpha=0.5)
line_abc    = mlines.Line2D([], [], color='#ffff00', label='ABC-DS Trajectory', linewidth=2)
line_nnds   = mlines.Line2D([], [], color='#ff0000', label=r"$S^2$-NNDS Trajectory", linewidth=2)
patch_init  = mpatches.Patch(color='cyan', label='Initial Set', alpha=0.7)

legend_handles = [
    patch_abc, patch_nnds, patch_inter, patch_unsafe,
    patch_init,
    line_abc, line_nnds
]
legend_labels = [
    'B < 0 (ABC-DS)', 'B < 0 (S2-NNDS)', 'B < 0 (Intersection)', 'Unsafe Set',
    'Initial Set',
    'ABC-DS Trajectory', r"$S^2$-NNDS Trajectory"
]

# === Create a blank figure and save only the legend ===
fig_legend = plt.figure(figsize=(3, 2))
ax = fig_legend.add_subplot(111)
ax.axis('off')
legend = ax.legend(
    legend_handles, legend_labels,
    fontsize=6,
    loc='center',
    frameon=True,
    facecolor='white',
    framealpha=1.0,
    borderpad=1.2,
    ncol=1  # vertical; use ncol=2 for horizontal if preferred
)
fig_legend.savefig("only_legend.svg", bbox_inches='tight', pad_inches=0.1, transparent=True, format='svg')
plt.close(fig_legend)
