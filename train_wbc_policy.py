import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# 1. Load the Expert Data
print("📂 Loading Centaur Expert Data...")
dataset = np.load("dataset/centaur_expert_data.npz")
X = dataset['states']
Y = dataset['actions']

input_dim = X.shape[1]
output_dim = Y.shape[1]
print(f"✅ Loaded {X.shape[0]} samples. Input dim: {input_dim}, Output dim: {output_dim}")

# Convert to PyTorch Tensors
X_tensor = torch.FloatTensor(X)
Y_tensor = torch.FloatTensor(Y)

train_dataset = TensorDataset(X_tensor, Y_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


# 2. Define the Whole-Body Control Neural Network
class CentaurWBCPolicy(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # A robust multi-layer perceptron (MLP) for Loco-Manipulation
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)


model = CentaurWBCPolicy(input_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 3. Train the Model
epochs = 50
print(f"🚀 Training WBC Policy for {epochs} Epochs...")

for epoch in range(1, epochs + 1):
    epoch_loss = 0.0
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss / len(train_loader):.6f}")

# 4. Save the Weights
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/wbc_policy.pth")
print("🎉 Training Complete! Policy weights saved to: models/wbc_policy.pth")