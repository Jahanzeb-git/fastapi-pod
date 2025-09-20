from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage
import logging

from app.database import get_db, User
from auth.dependencies import get_current_user
from core.session_manager import SessionManager
from agent.prompts import AgentState, get_system_prompt
from agent.tools import AVAILABLE_TOOLS
from agent.graph import app_graph
from app.config import settings
from core.ui_event_manager import UIEventType

logger = logging.getLogger(__name__)

# API Router
agent_router = APIRouter()

# Pydantic Models for API
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)

class ChatResponse(BaseModel):
    response: str
    agent_state: str
    has_product: bool
    has_design: bool
    has_upscaled: bool

class ProductSelectionRequest(BaseModel):
    product: Dict[str, Any]
    user_selection: Dict[str, Any]

class AgentStatusResponse(BaseModel):
    agent_state: str
    has_product: bool
    product_name: Optional[str] = None
    has_design: bool
    has_upscaled: bool
    orders_count: int
    conversation_length: int

class OrderPlaceRequest(BaseModel):
    product_data: Dict[str, Any]
    design_data: Dict[str, Any] 
    user_selection: Dict[str, Any]
    customer_info: Dict[str, Any]
    pricing: Dict[str, Any]

class OrderPlaceResponse(BaseModel):
    status: str
    message: str
    order_id: str

# API Endpoints
@agent_router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
): 
    await SessionManager.add_conversation_message(db, current_user.id, "user", request.message)
    
    inputs = {"messages": [HumanMessage(content=request.message)], "user_id": str(current_user.id), "db": db}
    result = await app_graph.ainvoke(inputs)
    
    response_message = result["messages"][-1].content
    await SessionManager.add_conversation_message(db, current_user.id, "assistant", response_message)
    
    session_context = await SessionManager.get_session_context_for_agent(db, current_user.id)
    return ChatResponse(
        response=response_message,
        agent_state=session_context["agent_state"],
        has_product=session_context["product_context"] is not None,
        has_design=session_context["design_urls"] is not None and session_context["design_urls"].get("design_url") is not None,
        has_upscaled=session_context["design_urls"] is not None and session_context["design_urls"].get("upscaled_url") is not None,
    )

@agent_router.post("/product/select")
async def select_product(
    request: ProductSelectionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
): 
    await SessionManager.update_product_context(db, current_user.id, {"product": request.product, "user_selection": request.user_selection})
    return {"status": "success", "message": "Product selected successfully"}

@agent_router.get("/status", response_model=AgentStatusResponse)
async def get_agent_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
): 
    session_context = await SessionManager.get_session_context_for_agent(db, current_user.id)
    return AgentStatusResponse(
        agent_state=session_context["agent_state"],
        has_product=session_context["product_context"] is not None,
        product_name=session_context["product_context"].get("product", {}).get("name") if session_context["product_context"] else None,
        has_design=session_context["design_urls"] is not None and session_context["design_urls"].get("design_url") is not None,
        has_upscaled=session_context["design_urls"] is not None and session_context["design_urls"].get("upscaled_url") is not None,
        orders_count=len(session_context["orders_history"]),
        conversation_length=len(session_context["conversation_history"])
    )

@agent_router.delete("/reset")
async def reset_agent(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
): 
    await SessionManager.clear_session(db, current_user.id)
    return {"status": "success", "message": "Agent context reset"}
    

@agent_router.post("/order/place", response_model=OrderPlaceResponse)
async def place_order_endpoint(
    request: OrderPlaceRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Place order with complete order data from frontend UI.
    This endpoint handles the actual order creation and database updates.
    """
    try:
        # Create order using SessionManager
        order = await SessionManager.create_order(
            db, 
            current_user.id, 
            request.product_data,
            request.design_data, 
            request.user_selection,
            request.customer_info,
            request.pricing
        )
        
        if not order:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create order"
            )

        # Add order confirmation message to conversation
        order_message = f"Order {order.order_number} has been placed successfully! Your design will be printed and shipped soon."
        await SessionManager.add_conversation_message(db, current_user.id, "system", order_message)

        # Reset session state and clear conversation history after order
        await SessionManager.clear_session(db, current_user.id)
        
        return OrderPlaceResponse(
            status="success",
            message="Order placed successfully",
            order_id=order.order_number
        )
        
    except Exception as e:
        logger.error(f"Order placement error for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to place order"
        )


@agent_router.post("/ui-event")
async def handle_ui_event(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Handle UI event via REST API (alternative to websocket)
    """
    
    event_result = await SessionManager.add_ui_event(
        db=db,
        user_id=current_user.id,
        event_type=request.get("event_type"),
        event_data=request.get("data", {})
    )
    
    response = {
        "acknowledged": True,
        "should_trigger": event_result.get("should_trigger", False),
        "priority": event_result.get("priority", "low")
    }
    
    # If event should trigger agent response
    if event_result.get("should_trigger"):
        prompt = event_result.get("prompt")
        if prompt:
            # Process through agent
            inputs = {
                "messages": [HumanMessage(content=prompt)],
                "user_id": str(current_user.id),
                "db": db
            }
            
            result = await app_graph.ainvoke(inputs)
            response_message = result["messages"][-1].content
            
            # Store agent response
            await SessionManager.add_conversation_message(
                db, current_user.id, "assistant", response_message
            )
            
            response["agent_response"] = response_message
    
    return response

@agent_router.get("/ui-events/recent")
async def get_recent_ui_events(
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get recent UI events for the current user"""
    events = await SessionManager.get_recent_ui_events(db, current_user.id, limit)
    return {"events": events}