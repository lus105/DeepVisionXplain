from fastapi import FastAPI
from src.api.training import router


app = FastAPI(title="NN Training API", description="API for managing training processes.")
# Include API routes
app.include_router(router.router, prefix="/training", tags=["Training"])