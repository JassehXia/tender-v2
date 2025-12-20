import json
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn

# Load exported JSON
with open("data/food_sample.json") as f:
    food_data = json.load(f)

df = pd.DataFrame(food_data)

# -----------------------------
# CLEAN TAGS
# -----------------------------

STOP_TAGS = {
    "homemade food",
    "restaurant food"
}

def clean_tag(tag):
    if not isinstance(tag, str):
        return None

    tag = (
        tag.replace("[", "")
           .replace("]", "")
           .replace('"', "")
           .replace("'", "")
           .strip()
           .lower()
    )

    if tag in STOP_TAGS or tag == "":
        return None

    return tag


def clean_tag_list(tags):
    cleaned = []
    for t in tags:
        ct = clean_tag(t)
        if ct:
            cleaned.append(ct)
    return cleaned


df["tags"] = df["tags"].apply(clean_tag_list)

import random
import numpy as np
# -----------------------------
# SYNTHETIC USER TASTE PROFILE
# -----------------------------

STRONG_LIKES = {
    "chicken": 3,
    "noodles": 3,
    "rice": 2,
    "beef": 2,
    "stir-frying": 3,
    "fried": 2,
    "savory": 2,
}

WEAK_LIKES = {
    "vegetables": 1,
    "green onions": 1,
    "sauce": 1,
    "spices": 1,
}

STRONG_DISLIKES = {
    "sweet": -3,
    "dessert": -3,
    "fruit": -2,
    "cake": -3,
    "milk": -2,
}

WEAK_DISLIKES = {
    "boiling": -1,
    "steamed": -1,
    "raw": -1,
}



# One-hot encode ingredients
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['tags'])



# Simulate user likes (for testing)
def score_food(tags):
    score = 0

    for t in tags:
        if t in STRONG_LIKES:
            score += STRONG_LIKES[t]
        elif t in WEAK_LIKES:
            score += WEAK_LIKES[t]
        elif t in STRONG_DISLIKES:
            score += STRONG_DISLIKES[t]
        elif t in WEAK_DISLIKES:
            score += WEAK_DISLIKES[t]

    # Add noise to simulate imperfect decisions
    score += np.random.normal(0, 0.5)

    return score


y = []
raw_scores = []

for tags in df["tags"]:
    s = score_food(tags)
    raw_scores.append(s)

    # Decision boundary
    y.append(1 if s > 0 else 0)


from collections import Counter

liked_tags = Counter()
disliked_tags = Counter()

for i, tags in enumerate(df['tags']):
    if y[i] == 1:
        liked_tags.update(tags)
    else:
        disliked_tags.update(tags)

print("\n===== TOP 10 LIKED TAGS =====")
for tag, count in liked_tags.most_common(10):
    print(f"{tag}: {count}")

print("\n===== TOP 10 DISLIKED TAGS =====")
for tag, count in disliked_tags.most_common(10):
    print(f"{tag}: {count}")




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

# -------------------------------
# Model predictions
# -------------------------------
results = pd.DataFrame({
    "name": df["name"],
    "prob": probs
})

# Top 10 likes (highest probabilities)
print("\n===== TOP 10 PREDICTED LIKES =====")
top_likes = results.sort_values("prob", ascending=False).head(10)
for _, row in top_likes.iterrows():
    print(f"{row['name']}: {row['prob']:.2f}")

# Top 10 dislikes (lowest probabilities)
print("\n===== TOP 10 PREDICTED DISLIKES =====")
top_dislikes = results.sort_values("prob").head(10)
for _, row in top_dislikes.iterrows():
    print(f"{row['name']}: {row['prob']:.2f}")
