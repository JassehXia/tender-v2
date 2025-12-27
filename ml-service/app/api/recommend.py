from fastapi import APIRouter
from pydantic import BaseModel
from app.services.recommender import recommend

router = APIRouter()

class RecommendRequest(BaseModel):
    food_ids: list[int]

class RecommendResponse(BaseModel):
    ranked_food_ids: list[int]

@router.post("/recommend",response_model=RecommendRequest)
def recommend_foods(payload: RecommendRequest):
    ranked = recommend(payload.food_ids)
    return {"ranked_food_ids":ranked}