import matplotlib.pyplot as plt

# Setup canvas
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor('white')

# Define box styles
box_fwd = dict(boxstyle="round,pad=0.5", facecolor="#e1f5fe", edgecolor="#0277bd", linewidth=2)
box_loss = dict(boxstyle="round,pad=0.5", facecolor="#ffebee", edgecolor="#c62828", linewidth=2)
box_bwd = dict(boxstyle="round,pad=0.5", facecolor="#fff3e0", edgecolor="#ef6c00", linewidth=2)
box_upd = dict(boxstyle="round,pad=0.5", facecolor="#e8f5e9", edgecolor="#2e7d32", linewidth=2)

# Draw Boxes (Text nodes)
ax.text(0.1, 0.75, "Input X", size=12, ha="center", va="center", bbox=box_fwd, fontweight='bold')
ax.text(0.5, 0.75, "Hidden Layer A1", size=12, ha="center", va="center", bbox=box_fwd, fontweight='bold')
ax.text(0.9, 0.75, "Output ŷ", size=12, ha="center", va="center", bbox=box_fwd, fontweight='bold')

ax.text(0.9, 0.25, "Loss (y - ŷ)", size=12, ha="center", va="center", bbox=box_loss, fontweight='bold')
ax.text(0.65, 0.25, "Backpropagation", size=12, ha="center", va="center", bbox=box_bwd, fontweight='bold')
ax.text(0.35, 0.25, "Gradient Descent", size=12, ha="center", va="center", bbox=box_bwd, fontweight='bold')
ax.text(0.1, 0.25, "Update W, b", size=12, ha="center", va="center", bbox=box_upd, fontweight='bold')

# Arrow properties
arrow_props = dict(facecolor='#333333', shrink=0.05, width=2, headwidth=10, edgecolor='none')

# Forward Arrows
ax.annotate("", xy=(0.35, 0.75), xytext=(0.17, 0.75), arrowprops=arrow_props)
ax.text(0.26, 0.78, "W1, b1", ha="center", fontweight='bold', color="#0277bd")

ax.annotate("", xy=(0.82, 0.75), xytext=(0.63, 0.75), arrowprops=arrow_props)
ax.text(0.725, 0.78, "W2, b2", ha="center", fontweight='bold', color="#0277bd")

# Down Arrow (Output to Loss)
ax.annotate("", xy=(0.9, 0.32), xytext=(0.9, 0.68), arrowprops=arrow_props)

# Backward Arrows
ax.annotate("", xy=(0.78, 0.25), xytext=(0.82, 0.25), arrowprops=arrow_props)
ax.annotate("", xy=(0.48, 0.25), xytext=(0.51, 0.25), arrowprops=arrow_props)
ax.annotate("", xy=(0.20, 0.25), xytext=(0.23, 0.25), arrowprops=arrow_props)

# Apply Updates Arrow (Dashed)
dashed_arrow = dict(facecolor='#2e7d32', shrink=0.05, width=2, headwidth=10, edgecolor='none', linestyle='dashed')
ax.annotate("", xy=(0.1, 0.68), xytext=(0.1, 0.32), arrowprops=dashed_arrow)
ax.text(0.08, 0.5, "Apply Updates", rotation=90, va='center', ha='center', fontweight='bold', color="#2e7d32")

# Final adjustments
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.title("Neural Network Training Process (Forward & Backward Propagation)", fontsize=16, fontweight='bold', pad=10)

# Save image
plt.savefig("nn_diagram.png", dpi=300, bbox_inches='tight')
print("Diagram generated successfully: nn_diagram.png")
