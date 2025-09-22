import json
from langchain_core.tools import tool
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from core.session_manager import SessionManager

@tool
async def search_order(order_id: str, user_id: str, db: AsyncSession) -> str:
    """
    Search for previous orders by order ID.
    Args:
        order_id: The order ID to search for.
    Returns:
        Order details if found, or not found message.
    """
    orders = await SessionManager.get_user_orders(db, UUID(user_id))
    for order in orders:
        if order.order_number == order_id:
            return json.dumps({"status": "success", "order": order.dict()})
    return json.dumps({"status": "error", "message": "Order not found."})
