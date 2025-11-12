from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from schema.convex_schema import ConvexRequest
from utils.convex import call_convex

router = APIRouter(prefix="/recommendation")


@router.get("/history/{branch_id}")
async def history_route(
    branch_id: str,
    user_id: str = Query(..., description="User ID"),
    n: int = Query(..., description="N Dishes"),
):
    """Get dish recommendations for a restaurant"""

    try:
        convex_response = await call_convex(
            ConvexRequest(
                module="orders",
                func="getOrdersByBranchAndUser",
                isQuery=True,
                args={
                    "branchId": "jh73qhmd8592jr1aff209jxa0d7stqf7",
                    "userName": "Manish",
                },
                returnDf=False,
            )
        )

        if convex_response.get("type") == "error":
            return JSONResponse(
                status_code=500, content={"error": convex_response.get("error")}
            )

        dishes = convex_response.get("data")

        if not dishes or len(dishes) == 0:
            return JSONResponse(
                status_code=200,
                content={"message": "No Dishes to Recommend", "dishes": []},
            )

        return JSONResponse(
            status_code=200,
            content={"message": "Success", "dishes": dishes},
        )

    except Exception as e:
        error = f"Error Getting dishes: {str(e)}"
        print(error)
        return JSONResponse(status_code=500, content={"error": error})
