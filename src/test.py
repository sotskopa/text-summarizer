import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from itertools import chain

# Constants
x = ["1-gram", "2-gram", "3-gram", "4-gram"]
categories = ["Candidate", "Reference", "Test"]  
hatches = ["/", "\\", "-", "o", ".", "x"] * 10  
color_names = ["green", "blue", "red", "yellow"] * 10  
alpha = 0.4
bar_position_adjustment = 0.05

# Generate data
heights = [[np.random.random() for _ in x] for _ in categories]
data = pd.DataFrame({
    "n-gram": x * len(categories),
    "Novel n-gram ratio": list(chain(*heights)),
    "Category": np.repeat(categories, len(x))
})

# Set the seaborn style
sns.set(style="whitegrid", font="serif", rc={"axes.linewidth": 0.5, "axes.labelsize": 15, "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15})

# Adjust color transparency
palette = [to_rgba(color, alpha=alpha) for color in color_names[: len(categories)]]

# Plot
plt.figure(figsize=(6, 5))
ax = sns.barplot(x="n-gram", y="Novel n-gram ratio", hue="Category", data=data, linewidth=2, palette=palette, zorder=2, alpha=alpha, width=0.4)
plt.gca().xaxis.set_tick_params(pad=2)  # Add horizontal space between bars in the same group

# Create the legend
legend_handles = [Patch(facecolor=palette[i], edgecolor=(palette[i][:3] + (0.5,)), hatch=hatches[i]) for i in range(len(categories))]
plt.legend(handles=legend_handles, labels=categories)

# Apply hatch and adjust position of bars
for i, bar in enumerate(ax.patches):
    bar.set_hatch(hatches[i // len(x)])
    bar.set_x(bar.get_x() + bar_position_adjustment * (i // len(x)))
    bar.set_edgecolor(bar.get_facecolor()[:3] + (0.5,))

# Set labels and rotate x-axis labels
plt.xlabel("")
plt.ylabel("Novel n-gram ratio", fontsize=15)
plt.xticks(rotation=30)

plt.show()
