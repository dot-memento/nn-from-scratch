import pandas as pd
import matplotlib.pyplot as plt

loss_file_path = "loss.csv"
scatter_file_path = "scatter.csv"
loss_df = pd.read_csv(loss_file_path)
scatter_df = pd.read_csv(scatter_file_path)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,8))

ax1.plot(loss_df['epoch'], loss_df['loss'], linestyle='-', color='blue')
ax1.set_title('Loss Evolution during Training')
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Average Loss")
ax1.grid(True)

ax2.scatter(scatter_df['input'], scatter_df['predicted'], alpha=0.6, edgecolors="k")
ax2.scatter(scatter_df['input'], scatter_df['expected'], alpha=0.6, edgecolors="k")
ax2.set_title("Estimated function plot")
ax2.set_xlabel("x")
ax2.set_ylabel("Predicted f(x)")
ax2.grid(True)

fig.tight_layout()
fig.savefig('graph.png')
