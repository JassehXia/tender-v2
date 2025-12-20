import json
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn

# Load exported JSON
with open("prisma/food_sample.json") as f:
    food_data = json.load(f)

df = pd.DataFrame(food_data)

# One-hot encode ingredients
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['tags'])

# Simulate user likes (for testing)
y = [1 if i % 2 == 0 else 0 for i in range(len(df))]

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Neural network
class SimpleRecommender(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = SimpleRecommender(X.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(50):
    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Make predictions
with torch.no_grad():
    probs = model(X_tensor).squeeze().numpy()

for name, prob in zip(df['name'], probs):
    print(f"{name}: {prob:.2f}")

print("-----TOP CHOICES-----")
top_indices = probs.argsort()[-5:][::-1]
for i in top_indices:
    print(f"{df['name'][i]}: {probs[i]:.2f}")