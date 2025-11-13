from env.modules import *
from optimization.optimize_GB import trials

# --- Build arrays 
weights_arr = []
accuracy_arr = []

for t in trials.trials:
    vals = t['misc']['vals']
    weights = [
        vals['w_gb'][0],
        vals['w_et'][0],
        vals['w_lgbm'][0],
        vals['w_rf'][0],
    ]
    weights_arr.append(weights)
    accuracy_arr.append(-t['result']['loss'])

weights_arr = np.array(weights_arr)
accuracy_arr = np.array(accuracy_arr)

# Labels shown on the x-axis (one per trial) — each label is 4 lines (GB, ET, LGBM, RF). Trim to one decimal for readability
weight_labels = ["\n".join([f"{w:.1f}" for w in ws]) for ws in weights_arr]
x = np.arange(len(weights_arr))

# --- Highlight color for highest-performing bar(s)
max_acc = accuracy_arr.max()
colors = np.array(['C0'] * len(x))                 # default color
colors[np.isclose(accuracy_arr, max_acc)] = 'C1'   # highlight best

# --- Figure & axes: keep 16:9 aspect ratio
fig, ax = plt.subplots(figsize=(16, 9), dpi=120)

# Bars (color array to highlight the best bar)
bars = ax.bar(x, accuracy_arr, width=0.85, color=colors)

# --- Y-limits: tighten to reduce empty space above/below the bars
pad = 0.002
ymin = float(np.floor(1000 * (accuracy_arr.min() - pad)) / 1000)
ymax = float(np.ceil(1000 * (accuracy_arr.max() + pad)) / 1000)
if np.isclose(ymin, ymax):  # safety in degenerate cases
    ymin -= 0.001
    ymax += 0.001
ax.set_ylim(ymin, ymax)

# Reduce horizontal margins (no big gaps at plot edges)
ax.margins(x=0.01)

# --- Value annotations on bars (3 decimals, vertical)
# Place near the middle of each bar; make the BEST bar's text bigger and higher-contrast.
for bar in bars:
    is_best = np.isclose(bar.get_height(), max_acc)
    y_mid = ymin + 0.5 * (bar.get_height() - ymin)

    txt_size = 12 if is_best else 10
    txt_color = "black" if is_best else "white"
    halo = pe.withStroke(linewidth=2.0, foreground="white") if is_best else pe.withStroke(linewidth=1.5, foreground="black")

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y_mid,
        f"{bar.get_height():.3f}",
        ha="center",
        va="center",
        rotation=90,
        rotation_mode="anchor",
        fontsize=txt_size,
        fontweight="normal",
        color=txt_color,
        path_effects=[halo],
        clip_on=True
    )

# --- Axis labels and title (readable but compact)
ax.set_xticks(x)
tick_texts = ax.set_xticklabels(weight_labels, fontsize=8, rotation=0, ha="center")
# Improve readability of the 4-line weight labels
for tt in tick_texts:
    try:
        tt.set_linespacing(0.9)
    except Exception:
        pass

ax.set_xlabel("Trial Weights (top to bottom): GB, ET, LGBM, RF", fontsize=11, labelpad=10)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title(
    "Ensemble Test Accuracy (Soft Voting Trials) — best trial highlighted",
    fontsize=13,
    pad=12
)

# Horizontal gridlines for easier reading
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# --- Legend explaining the highlight
handles = [
    Patch(facecolor='C1', label='Highest accuracy'),
    Patch(facecolor='C0', label='Other trials')
]
ax.legend(handles=handles, loc='upper left', fontsize=9, frameon=False)

# Tight layout with a bit more bottom space for the multi-line tick labels
plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.96])
plt.show()