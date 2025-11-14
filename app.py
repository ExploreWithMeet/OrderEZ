from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import routers
from routes import dp_routes

# Create FastAPI app
app = FastAPI(
    title="OrderEZ ML API",
    description="Machine Learning backend for OrderEZ - Dynamic Pricing & Recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(dp_routes.router)

# Root endpoint
@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={
            "message": "OrderEZ - ML Backend API",
            "status": "running",
            "version": "1.0.0",
            "endpoints": {
                "dynamic_pricing": {
                    "train": {
                        "method": "POST",
                        "url": "/dp/train",
                        "description": "Train LSTM model for a branch"
                    },
                    "predict": {
                        "method": "POST",
                        "url": "/dp/predict",
                        "description": "Generate price predictions"
                    },
                    "predictions": {
                        "method": "GET",
                        "url": "/dp/predictions/{branch_id}",
                        "description": "Get stored predictions"
                    },
                    "model": {
                        "method": "GET",
                        "url": "/dp/model/{branch_id}",
                        "description": "Get model information"
                    },
                    "branches": {
                        "method": "GET",
                        "url": "/dp/branches",
                        "description": "Get all branches"
                    },
                    "apply_price": {
                        "method": "POST",
                        "url": "/dp/apply-price",
                        "description": "Apply predicted price"
                    },
                    "health": {
                        "method": "GET",
                        "url": "/dp/health",
                        "description": "Health check"
                    }
                }
            }
        },
    )

# Health check
@app.get("/health")
async def health_check():
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "OrderEZ ML API",
            "version": "1.0.0"
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 Starting OrderEZ ML API Server")
    print("="*70)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )