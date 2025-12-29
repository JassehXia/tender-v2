from fastapi import APIRouter
from pydantic import BaseModel
from app.services.model import recommend

router = APIRouter()

class RecommendRequest(BaseModel):
    food_ids: list[int]

class RankedFood(BaseModel):
    id: int
    prob: float

class RecommendResponse(BaseModel):
    ranked_foods: list[RankedFood]

@router.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(payload: RecommendRequest):
    ranked = recommend(payload.food_ids)
    return {"ranked_foods": ranked}
