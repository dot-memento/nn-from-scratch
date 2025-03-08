#! /bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 7),
            nn.Tanh(),
            nn.Linear(7, 6),
            nn.Tanh(),
            nn.Linear(6, 7),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(7, 8),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Generate dataset: all 8-bit binary numbers (0 to 255)
binary_data = np.array([[int(b) for b in format(i, '08b')] for i in range(256)], dtype=np.float32)
dataset = torch.tensor(binary_data)

# Initialize model, loss function, and optimizer
model = Autoencoder()
criterion = nn.BCELoss()  # Reconstruction loss
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.005,           # Lower than default (try 0.001 - 0.01)
    betas=(0.85, 0.98), # Less aggressive momentum update
    weight_decay=0.005, # Regularization (default: 0.01, try lower for small networks)
    amsgrad=True        # Helps with stability in long training
)

# Training loop (SGD-style: process one sample at a time)
epochs = 200
losses = []

for epoch in range(epochs):
    total_loss = 0.0
    for i in range(len(dataset)):  # Iterate through all samples one by one
        optimizer.zero_grad()
        input_sample = dataset[i].unsqueeze(0)  # Add batch dimension (1, 8)
        output = model(input_sample)
        loss = criterion(output, input_sample)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / 256
    losses.append(avg_loss)  # Store average loss for each epoch
    print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.6f}")

# Plot the loss evolution
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.title('Loss Evolution during Training')
plt.grid(True)
plt.savefig("torch_plot")

# Test the model
with torch.no_grad():
    test_output = model(dataset).round()  # Round output to get binary values
    print("\nOriginal vs. Reconstructed:")
    for i in range(10):  # Show first 10 examples
        print(f"Input: {dataset[i].numpy()} -> Output: {test_output[i].numpy()}")
