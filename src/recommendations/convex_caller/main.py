from app.schema.convex_schema import ConvexRequest
from app.utils.convex import call_convex


async def call_new_orders():
    try:
        orders = call_convex(ConvexRequest())
    except Exception as e:
        error = f"Error Occured getting new orders: {str(e)}"
        print(error)
        return {"type": "error", "message": error}


async def call_old_rules():
    pass
