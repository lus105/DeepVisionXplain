import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.training import router


app = FastAPI(
    title='NN Training API', description='API for managing training processes.'
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'http://localhost:8000',
        'http://127.0.0.1:8000',
    ],  # Only allow same origin
    allow_credentials=True,
    allow_methods=['GET', 'POST'],  # Only allow needed methods
    allow_headers=['*'],  # Allows all headers
)

app.include_router(router.router, prefix='/training', tags=['Training'])

if __name__ == '__main__':
    uvicorn.run(app)
