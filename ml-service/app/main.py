from fastapi import FastAPI
from app.api.health import router as health_router
from app.api.recommend import router as recommend_router

app = FastAPI(
    title= "Tender ML Service",
    version = "0.1.0"
)

app.include_router(health_router)
app.include_router(recommend_router)