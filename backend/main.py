from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from model import recommender
import json
import os

app = FastAPI(title="Food Recommendation API")

# CORS configuration - update with your Vercel URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://*.vercel.app",
        os.getenv("FRONTEND_URL", "http://localhost:3000")
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class FoodItem(BaseModel):
    id: str
    name: str
    tags: List[str]
    imageUrl: Optional[str] = None

class UserInteraction(BaseModel):
    foodId: str
    action: str  # LIKE, DISLIKE, SAVE, SKIP

class TrainRequest(BaseModel):
    foods: List[FoodItem]
    interactions: List[UserInteraction]

class PredictRequest(BaseModel):
    foods: List[FoodItem]

class PredictionResponse(BaseModel):
    id: str
    name: str
    tags: List[str]
    imageUrl: Optional[str]
    score: float

@app.get("/")
def read_root():
    return {
        "status": "Food Recommendation API is running",
        "model_trained": recommender.is_trained
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": recommender.is_trained
    }

@app.post("/train")
def train_model(request: TrainRequest):
    """
    Train the recommendation model based on user interactions
    """
    try:
        # Convert Pydantic models to dicts
        food_data = [food.model_dump() for food in request.foods]
        interactions = [interaction.model_dump() for interaction in request.interactions]
        
        # Check if we have enough data
        if len(interactions) < 5:
            raise HTTPException(
                status_code=400,
                detail="Need at least 5 interactions to train the model"
            )
        
        result = recommender.train_model(food_data, interactions)
        
        return {
            "message": "Model trained successfully",
            "details": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=List[PredictionResponse])
def predict(request: PredictRequest):
    """
    Get predictions for food items based on trained model
    """
    try:
        if not recommender.is_trained:
            raise HTTPException(
                status_code=400,
                detail="Model not trained yet. Please train the model first."
            )
        
        # Convert Pydantic models to dicts
        food_data = [food.model_dump() for food in request.foods]
        
        predictions = recommender.predict(food_data)
        
        return predictions
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend")
def get_recommendations(request: PredictRequest, top_n: int = 10):
    """
    Get top N recommended food items
    """
    try:
        predictions = predict(request)
        
        # Return top N recommendations
        return {
            "recommendations": predictions[:top_n],
            "total_evaluated": len(predictions)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-model")
def save_model():
    """
    Save the current model to disk
    """
    try:
        if not recommender.is_trained:
            raise HTTPException(status_code=400, detail="No model to save")
        
        recommender.save_model()
        return {"message": "Model saved successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-model")
def load_model():
    """
    Load a previously saved model
    """
    try:
        recommender.load_model()
        return {"message": "Model loaded successfully"}
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)