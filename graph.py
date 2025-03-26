#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# Load data
loss_file_path = "loss.csv"
loss_df = pd.read_csv(loss_file_path)

scatter_file_path = "scatter.csv"
scatter_df = pd.read_csv(scatter_file_path)

fig = plt.figure(figsize=(16, 8))

# Loss plot
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(loss_df['epoch'], loss_df['loss'], linestyle='-')
ax1.set_title('Loss Evolution during Training')
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Average Loss")
ax1.set_yscale('log')
ax1.grid(True)

# Extract data
x = scatter_df.iloc[:, 0].values
y = scatter_df.iloc[:, 1].values
z = scatter_df.iloc[:, 2].values
z_pred = scatter_df.iloc[:, 3].values

# 3D scatter plot
"""ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.scatter(x, y, z, alpha=0.6, label="Ground Truth")
ax2.scatter(x, y, z_pred, alpha=0.6, label="Predictions")
ax2.set_title("3D Estimated Function Plot")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("f(x, y)")
ax2.legend()
ax2.grid(True)"""

# Surface plot with proper interpolation
ax3 = fig.add_subplot(1, 3, 3, projection='3d')

# Create a grid for interpolation
xi = np.linspace(x.min(), x.max(), 50)
yi = np.linspace(y.min(), y.max(), 50)
X, Y = np.meshgrid(xi, yi)
Z = griddata((x, y), z, (X, Y), method='cubic')
Z_pred = griddata((x, y), z_pred, (X, Y), method='cubic')

# Plot the surface
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax2.set_title("3D Surface of Actual Function")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("f(x, y)")
ax2.grid(True)

# Plot the surface
ax3.plot_surface(X, Y, Z_pred, cmap='viridis', alpha=0.7)
ax3.set_title("3D Surface of Predicted Function")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("f(x, y)")
ax3.grid(True)

fig.tight_layout()
plt.show()
