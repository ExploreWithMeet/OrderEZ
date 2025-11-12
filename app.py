from fastapi import FastAPI
from fastapi.responses import JSONResponse
from routes import recommendation_routes

app = FastAPI(title="OrderEZ ML API")

app.include_router(recommendation_routes.router)
# History Based Food Recommendation
# Cart Based Food Recommendation
# Dynamic pricing on some criteria
# In Restaurant ordered category based recommendation
# In Restaurant Recent Orders recommendation

# ---- optional ----
# Demand Forecasting
# Delivery Time Prediction - Random Forest
# Restaurant Recommendations
# Customer Lifetime Value for Restaurants


@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={
            "message": "OrderEZ - ML Backend API",
            "status": "running",
            "endpoints": {
                "recommendation": {
                    "method": "GET",
                    "url": "/recommend/{user_id}/{restaurant_id}",
                },
            },
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="localhost", port=5000, reload=True)
