import pandas as pd
import matplotlib.pyplot as plt

file_path = "data.csv"
df = pd.read_csv(file_path)

fig, ax1 = plt.subplots()
ax1.set_title('Loss/Accuracy Evolution during Training')
ax1.grid(True)

ax1.set_xlabel("Epochs")
ax1.set_ylabel("Average Loss")
ax1.plot(df['epoch'], df['loss'], linestyle='-', color='blue')

ax2 = ax1.twinx()
ax2.set_ylabel("Accuracy")
ax2.plot(df['epoch'], df['accuracy'], linestyle='-', color='red')

fig.tight_layout()
fig.savefig('graph.png')
