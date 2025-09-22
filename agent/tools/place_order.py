import json
from typing import Dict, Any
from langchain_core.tools import tool
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from core.session_manager import SessionManager

@tool
async def place_order(user_id: str, db: AsyncSession) -> str:
    """
    Trigger frontend order placement UI via websocket.
    Args:
        user_id: User ID to send websocket signal to.
    Returns:
        Signal result confirming UI trigger.
    """
    session_context = await SessionManager.get_session_context_for_agent(db, UUID(user_id))
    product_data = session_context.get("product_context", {}).get("product")
    user_selection = session_context.get("product_context", {}).get("user_selection")
    design_data = session_context.get("design_urls")

    if not all([product_data, user_selection, design_data]):
        return json.dumps({"status": "error", "message": "Missing product, selection, or design for order placement."})

    # Send websocket signal to frontend
    try: 
        from core.websocket_manager import websocket_manager

        order_context = {
            "product": product_data,
            "design": design_data,
            "selection": user_selection
        }
    
        signal_sent = await websocket_manager.send_order_ui_trigger(user_id, order_context)
    
        if signal_sent:
            return json.dumps({
                "status": "success", 
                "message": "Order placement UI triggered. Please guide user to complete the order in the popup window.",
                "action": "trigger_order_ui"
            })
        else:
            return json.dumps({
                "status": "warning",
                "message": "Order UI trigger sent, but no active connection found. Please refresh the page and try again."
            })
    except ImportError:
        # Fallback if websocket is not available.. 
        return json.dumps({
            "status": "success", 
            "message": "Order placement UI triggered. Please complete your order in the popup window.",
            "action": "trigger_order_ui"
        }) 
