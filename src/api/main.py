import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from src.api.training import router

# Get the directory where this file is located
API_DIR = Path(__file__).parent
STATIC_DIR = API_DIR / "static"


app = FastAPI(
    title='NN Training API', description='API for managing training processes.'
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],  # Only allow same origin
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only allow needed methods
    allow_headers=["*"],  # Allows all headers
)

# Serve the training interface
@app.get("/", response_class=FileResponse)
async def get_training_interface():
    """Serve the training interface HTML page"""
    html_file = STATIC_DIR / "training_interface.html"
    return FileResponse(html_file)

app.include_router(router.router, prefix='/training', tags=['Training'])

if __name__ == '__main__':
    uvicorn.run(app)
