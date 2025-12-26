import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
from typing import List, Dict
import pickle
import os

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

class FoodRecommender:
    def __init__(self):
        self.model = None
        self.mlb = None
        self.is_trained = False
    
    def train_model(self, food_data: List[Dict], user_interactions: List[Dict]):
        """
        Train the model based on user interactions
        
        Args:
            food_data: List of food items with tags
            user_interactions: List of user interactions (like/dislike)
        """
        df = pd.DataFrame(food_data)
        df["tags"] = df["tags"].apply(clean_tag_list)
        
        # One-hot encode tags
        self.mlb = MultiLabelBinarizer()
        X = self.mlb.fit_transform(df['tags'])
        
        # Create labels based on user interactions
        y = self._create_labels(df, user_interactions)
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        # Initialize and train model
        self.model = SimpleRecommender(X.shape[1])
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Training loop
        for epoch in range(50):
            y_pred = self.model(X_tensor)
            loss = criterion(y_pred, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        self.is_trained = True
        
        return {"status": "trained", "num_foods": len(df), "num_interactions": len(user_interactions)}
    
    def _create_labels(self, df: pd.DataFrame, interactions: List[Dict]) -> List[int]:
        """
        Create binary labels based on user interactions
        """
        # Create a mapping of food_id to label
        interaction_map = {}
        for interaction in interactions:
            food_id = interaction.get('foodId')
            action = interaction.get('action')
            
            if action == 'LIKE':
                interaction_map[food_id] = 1
            elif action == 'DISLIKE':
                interaction_map[food_id] = 0
        
        # Create labels array
        labels = []
        for idx, row in df.iterrows():
            food_id = row.get('id')
            # Default to 0 if no interaction
            labels.append(interaction_map.get(food_id, 0))
        
        return labels
    
    def predict(self, food_items: List[Dict]) -> List[Dict]:
        """
        Predict recommendations for given food items
        
        Args:
            food_items: List of food items with tags
            
        Returns:
            List of food items with prediction scores
        """
        if not self.is_trained or self.model is None or self.mlb is None:
            raise ValueError("Model not trained yet")
        
        df = pd.DataFrame(food_items)
        df["tags"] = df["tags"].apply(clean_tag_list)
        
        # Transform tags using existing MLBinarizer
        X = self.mlb.transform(df['tags'])
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Make predictions
        with torch.no_grad():
            probs = self.model(X_tensor).squeeze().numpy()
        
        # Combine with original data
        results = []
        for i, row in df.iterrows():
            results.append({
                "id": row.get('id'),
                "name": row.get('name'),
                "tags": row.get('tags'),
                "imageUrl": row.get('imageUrl'),
                "score": float(probs[i]) if probs.ndim > 0 else float(probs)
            })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def save_model(self, path: str = "model_checkpoint.pkl"):
        """Save model and preprocessor"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        checkpoint = {
            'model_state': self.model.state_dict(),
            'mlb': self.mlb,
            'input_dim': self.model.net[0].in_features
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_model(self, path: str = "model_checkpoint.pkl"):
        """Load model and preprocessor"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.mlb = checkpoint['mlb']
        self.model = SimpleRecommender(checkpoint['input_dim'])
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        self.is_trained = True

# Global instance
recommender = FoodRecommender()