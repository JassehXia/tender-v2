import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer

# -----------------------------
# LOAD DATA
# -----------------------------
with open("data/food_sample.json") as f:
    food_data = json.load(f)

df = pd.DataFrame(food_data)

STOP_TAGS = {"homemade food", "restaurant food"}

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
    return [ct for t in tags if (ct := clean_tag(t))]

df["tags"] = df["tags"].apply(clean_tag_list)

# -----------------------------
# SYNTHETIC USER PROFILE
# -----------------------------
STRONG_LIKES = {
    "chicken": 3, "noodles": 3, "rice": 2,
    "beef": 2, "stir-frying": 3, "fried": 2, "savory": 2
}
WEAK_LIKES = {"vegetables": 1, "green onions": 1, "sauce": 1, "spices": 1}
STRONG_DISLIKES = {"sweet": -3, "dessert": -3, "fruit": -2, "cake": -3, "milk": -2}
WEAK_DISLIKES = {"boiling": -1, "steamed": -1, "raw": -1}

def score_food(tags):
    score = 0
    for t in tags:
        score += STRONG_LIKES.get(t, 0)
        score += WEAK_LIKES.get(t, 0)
        score += STRONG_DISLIKES.get(t, 0)
        score += WEAK_DISLIKES.get(t, 0)
    score += np.random.normal(0, 0.5)
    return score

# -----------------------------
# TRAIN MODEL (ONCE)
# -----------------------------
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["tags"])

y = [(1 if score_food(tags) > 0 else 0) for tags in df["tags"]]

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

for _ in range(50):
    loss = criterion(model(X_tensor), y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()

# -----------------------------
# INFERENCE FUNCTION
# -----------------------------
def recommend(food_ids: list[int]):
    X_sub = X_tensor[food_ids]
    with torch.no_grad():
        probs = model(X_sub).squeeze().numpy()

    results = [
        {"id": int(fid), "prob": float(p)}
        for fid, p in zip(food_ids, probs)
    ]

    return sorted(results, key=lambda x: x["prob"], reverse=True)
