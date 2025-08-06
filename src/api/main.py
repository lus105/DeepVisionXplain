import uvicorn
from fastapi import FastAPI
from src.api.training import router

app = FastAPI(
    title='NN Training API', description='API for managing training processes.'
)

@app.get('/ping')
async def ping():
    """Health check endpoint"""
    return {'status': 'ok'}

app.include_router(router.router, prefix='/training', tags=['Training'])

if __name__ == '__main__':
    uvicorn.run(app)
