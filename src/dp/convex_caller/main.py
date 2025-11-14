from datetime import datetime, timedelta
from typing import Dict, List, Optional

from schema.convex_schema import ConvexRequest
from utils.convex import call_convex
from utils.returnFormat import returnFormat


async def fetch_branch_items(branch_id: str) -> dict:
    """
    Fetch all items for a branch

    Args:
        branch_id: Branch ID

    Returns:
        Response dict with items data (JSON list)
    """
    req = ConvexRequest(
        module="items",
        func="getByBranch",
        isQuery=True,
        args={"branchId": branch_id},
        returnDf=False,  # Changed to False - returns JSON
    )
    return await call_convex(req)


async def fetch_price_history(item_id: str, days: int = 30) -> dict:
    """
    Fetch price history for an item

    Args:
        item_id: Item ID
        days: Number of days of history

    Returns:
        Response dict with price history (JSON list)
    """
    end_time = datetime.now().timestamp() * 1000
    start_time = (datetime.now() - timedelta(days=days)).timestamp() * 1000

    req = ConvexRequest(
        module="prices",
        func="getHistoryByItem",
        isQuery=True,
        args={"itemId": item_id, "startTime": start_time, "endTime": end_time},
        returnDf=False,  # Changed to False - returns JSON
    )
    return await call_convex(req)


async def fetch_item_metrics(item_id: str, days: int = 7) -> dict:
    """
    Fetch aggregated metrics for an item

    Args:
        item_id: Item ID
        days: Number of days to aggregate

    Returns:
        Response dict with metrics (JSON object)
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    req = ConvexRequest(
        module="itemMetrics",
        func="getMetrics",
        isQuery=True,
        args={"itemId": item_id, "startDate": start_date, "endDate": end_date},
        returnDf=False,
    )
    return await call_convex(req)


async def fetch_orders_by_branch(branch_id: str, days: int = 30) -> dict:
    """
    Fetch recent orders for a branch

    Args:
        branch_id: Branch ID
        days: Number of days of orders

    Returns:
        Response dict with orders (JSON list)
    """
    end_time = datetime.now().timestamp() * 1000
    start_time = (datetime.now() - timedelta(days=days)).timestamp() * 1000

    req = ConvexRequest(
        module="orders",
        func="getByBranchTimeRange",
        isQuery=True,
        args={"branchId": branch_id, "startTime": start_time, "endTime": end_time},
        returnDf=False,  # Changed to False - returns JSON
    )
    return await call_convex(req)


async def fetch_training_data(branch_id: str, days: int = 90) -> dict:
    """
    Fetch comprehensive training data for a branch

    Args:
        branch_id: Branch ID
        days: Days of historical data

    Returns:
        Response dict with training data as list of records
    """
    # print(f"Fetching training data for branch {branch_id}...")

    # Fetch all necessary data
    items_response = await fetch_branch_items(branch_id)
    if items_response["type"] == "error":
        return items_response
    
    items_list = items_response["data"]  # Now it's a JSON list
    if not items_list:
        return returnFormat("error", f"No items found for branch {branch_id}")
    
    print(f"Found {len(items_list)} items")

    all_data = []

    for item in items_list:
        item_id = item["_id"]

        # Fetch price history
        prices_response = await fetch_price_history(item_id, days)
        if prices_response["type"] == "error":
            continue

        prices_list = prices_response["data"]  # Now it's a JSON list
        if not prices_list:
            continue

        # Fetch metrics
        metrics_response = await fetch_item_metrics(item_id, 7)
        metrics = (
            metrics_response["data"] if metrics_response["type"] == "success" else {}
        )

        # Process each price record
        for price_row in prices_list:
            timestamp = price_row["updatedAt"]
            dt = datetime.fromtimestamp(timestamp / 1000)

            record = {
                "item_id": item_id,
                "branch_id": branch_id,
                "current_price": price_row["value"],
                "base_price": item.get("basePrice", price_row["value"]),
                "timestamp": timestamp,
                "dt": dt.isoformat(),  # Convert to ISO string for JSON compatibility
                "demand_7d": price_row.get("demand", "MEDIUM"),
                "rating_7d": 4,  # Default, update from actual ratings if available
                "orders_7d": metrics.get("totalOrders", 0),
                "revenue_7d": metrics.get("totalRevenue", 0),
                "avg_quantity": metrics.get("totalQuantity", 0)
                / max(metrics.get("totalOrders", 1), 1),
                "time_of_day": (
                    "NOON"
                    if 12 <= dt.hour < 15
                    else (
                        "MORNING"
                        if 5 <= dt.hour < 12
                        else "AFTERNOON" if 15 <= dt.hour < 20 else "NIGHT"
                    )
                ),
                "season": (
                    "WINTER"
                    if dt.month in [12, 1, 2]
                    else "SUMMER" if dt.month in [3, 4, 5, 6] else "MONSOON"
                ),
                "day_of_week": dt.weekday(),
                "is_weekend": 1 if dt.weekday() >= 5 else 0,
                "is_holiday": price_row.get("isEvent", False),
                "is_event": price_row.get("isEvent", False),
                "event_name": "",
            }
            all_data.append(record)

    return returnFormat("success", f"Training data fetched: {len(all_data)} records", all_data)


async def store_prediction(prediction: Dict) -> dict:
    """
    Store price prediction in database

    Args:
        prediction: Prediction dictionary

    Returns:
        Response dict with stored prediction ID
    """
    req = ConvexRequest(
        module="pricingPredictions",
        func="create",
        isQuery=False,
        args={
            "itemId": prediction["item_id"],
            "branchId": prediction["branch_id"],
            "timestamp": prediction["predicted_at"],
            "predictedChangePercent": prediction["predicted_change_percent"],
            "confidence": prediction["confidence"],
            "currentPrice": prediction["current_price"],
            "suggestedPrice": prediction["suggested_price"],
            "demandCategory": prediction["demand_category"],
            "isApplied": False,
            "recommendation": prediction["recommendation"],
        },
        returnDf=False,
    )
    return await call_convex(req)


async def store_predictions_batch(predictions: List[Dict]) -> dict:
    """
    Store multiple predictions

    Args:
        predictions: List of prediction dictionaries

    Returns:
        Response dict with stored count
    """
    print(f"Storing {len(predictions)} predictions...")

    successful = 0
    failed = 0

    for pred in predictions:
        result = await store_prediction(pred)
        if result["type"] == "success":
            successful += 1
        else:
            failed += 1

    return returnFormat(
        "success" if failed == 0 else "error",
        f"Stored {successful} predictions, {failed} failed",
        {"successful": successful, "failed": failed},
    )


async def update_item_price(
    item_id: str, new_price: float, prediction_id: Optional[str] = None
) -> dict:
    """
    Update item price (apply dynamic pricing)

    Args:
        item_id: Item ID
        new_price: New price to set
        prediction_id: Optional prediction ID that triggered this

    Returns:
        Response dict
    """
    # Update item's current price
    req1 = ConvexRequest(
        module="items",
        func="updatePrice",
        isQuery=False,
        args={"itemId": item_id, "currentPrice": new_price},
        returnDf=False,
    )
    result1 = await call_convex(req1)

    if result1["type"] == "error":
        return result1

    # Add to price history
    req2 = ConvexRequest(
        module="prices",
        func="create",
        isQuery=False,
        args={
            "itemId": item_id,
            "value": new_price,
            "updatedAt": datetime.now().timestamp() * 1000,
        },
        returnDf=False,
    )
    result2 = await call_convex(req2)

    # Mark prediction as applied
    if prediction_id:
        req3 = ConvexRequest(
            module="pricingPredictions",
            func="markAsApplied",
            isQuery=False,
            args={
                "predictionId": prediction_id,
                "appliedAt": datetime.now().timestamp() * 1000,
            },
            returnDf=False,
        )
        await call_convex(req3)

    return returnFormat("success", f"Price updated to ₹{new_price:.2f}")


async def store_model_metadata(branch_id: str, metadata: Dict) -> dict:
    """
    Store ML model metadata

    Args:
        branch_id: Branch ID
        metadata: Model metadata dictionary

    Returns:
        Response dict
    """
    req = ConvexRequest(
        module="mlModels",
        func="create",
        isQuery=False,
        args={
            "branchId": branch_id,
            "modelVersion": metadata.get("model_version", "v1.0"),
            "trainedAt": metadata["trained_at_ms"],
            "accuracy": metadata.get("accuracy", 0),
            "mae": metadata.get("mae", 0),
            "rmse": metadata.get("rmse", 0),
            "totalSamples": metadata.get("total_samples", 0),
            "modelPath": metadata.get("model_path", ""),
            "preprocessorPath": metadata.get("preprocessor_path", ""),
            "isActive": True,
            "sequenceLength": metadata.get("sequence_length"),
            "maxPriceChange": metadata.get("max_price_change"),
            "minPriceChange": metadata.get("min_price_change"),
        },
        returnDf=False,
    )
    return await call_convex(req)


async def get_all_branches() -> dict:
    """
    Get list of all active branches

    Returns:
        Response dict with branch list (JSON)
    """
    req = ConvexRequest(
        module="branches", func="getAllActive", isQuery=True, args={}, returnDf=False
    )
    return await call_convex(req)


async def create_owner_alert(
    branch_id: str,
    item_id: str,
    alert_type: str,
    severity: str,
    message: str,
    predicted_change_percent: Optional[float] = None,
    confidence: Optional[float] = None,
) -> dict:
    """
    Create an owner alert for low demand items or other issues

    Args:
        branch_id: Branch ID
        item_id: Item ID
        alert_type: Type of alert (LOW_DEMAND, PRICE_CHANGE, STOCK_LOW)
        severity: Severity level (info, warning, danger)
        message: Alert message
        predicted_change_percent: Optional predicted change
        confidence: Optional confidence score

    Returns:
        Response dict
    """
    req = ConvexRequest(
        module="ownerAlerts",
        func="create",
        isQuery=False,
        args={
            "branchId": branch_id,
            "itemId": item_id,
            "alertType": alert_type,
            "severity": severity,
            "message": message,
            "predictedChangePercent": predicted_change_percent,
            "confidence": confidence,
            "timestamp": datetime.now().timestamp() * 1000,
            "isRead": False,
        },
        returnDf=False,
    )
    return await call_convex(req)


async def get_unread_alerts(branch_id: str) -> dict:
    """
    Get unread alerts for a branch

    Args:
        branch_id: Branch ID

    Returns:
        Response dict with alerts list (JSON)
    """
    req = ConvexRequest(
        module="ownerAlerts",
        func="getUnread",
        isQuery=True,
        args={"branchId": branch_id},
        returnDf=False,
    )
    return await call_convex(req)


async def get_active_model(branch_id: str) -> dict:
    """
    Get the active ML model for a branch

    Args:
        branch_id: Branch ID

    Returns:
        Response dict with model metadata (JSON)
    """
    req = ConvexRequest(
        module="mlModels",
        func="getActive",
        isQuery=True,
        args={"branchId": branch_id},
        returnDf=False,
    )
    return await call_convex(req)


async def get_latest_predictions(branch_id: str, limit: int = 20) -> dict:
    """
    Get latest predictions for a branch

    Args:
        branch_id: Branch ID
        limit: Number of predictions to fetch

    Returns:
        Response dict with predictions list (JSON)
    """
    req = ConvexRequest(
        module="pricingPredictions",
        func="getByBranch",
        isQuery=True,
        args={"branchId": branch_id, "limit": limit},
        returnDf=False,
    )
    return await call_convex(req)


async def get_predictions_by_demand(branch_id: str, demand_category: str) -> dict:
    """
    Get predictions filtered by demand category

    Args:
        branch_id: Branch ID
        demand_category: Demand category (HIGH, NORMAL, LOW)

    Returns:
        Response dict with predictions list (JSON)
    """
    req = ConvexRequest(
        module="pricingPredictions",
        func="getByDemandCategory",
        isQuery=True,
        args={"branchId": branch_id, "demandCategory": demand_category},
        returnDf=False,
    )
    return await call_convex(req)