#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

# Load data
loss_file_path = "loss.csv"
loss_df = pd.read_csv(loss_file_path)

fig = plt.figure(figsize=(6, 4))

# Loss plot
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title('Loss Evolution during Training')
ax1.plot(loss_df['epoch'], loss_df['loss'], linestyle='-')
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Average Loss")
ax1.set_yscale('log')
ax1.grid(True)
ax1.legend(['Loss'], loc='upper right', fontsize='small')

# Accuracy plot
ax2 = ax1.twinx()
ax2.plot(loss_df['epoch'], loss_df['accuracy'], color='orange', linestyle='--')
ax2.set_ylabel("Accuracy")
ax2.grid(False)
ax2.legend(['Accuracy'], loc='upper left', fontsize='small')
ax2.set_ylim([0, 1])

fig.tight_layout()
plt.show()
