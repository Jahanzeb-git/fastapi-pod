import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.database import init_db, close_db
from auth.router import auth_router
from agent.router import agent_router  # Assuming agent logic is in agent.py
from app.config import settings

from datetime import datetime, timezone
import json
import logging
from typing import Dict, Any, List, Optional
from uuid import UUID
from fastapi import WebSocket, WebSocketDisconnect

from core.websocket_manager import websocket_manager
from auth.dependencies import get_current_user_optional
from core.ui_event_manager import UIEventType

# Configure logging
logging.basicConfig(level=logging.INFO if settings.DEBUG else logging.WARNING)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Print-on-Demand AI Agent API",
    version="1.0.0",
    description="API for an AI-powered print-on-demand design agent.",
    debug=settings.DEBUG,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Event handlers for DB connection
@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup"""
    logger.info("Starting up API and initializing database...")
    await init_db()
    logger.info("Database initialized.")

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown"""
    logger.info("Shutting down API and closing database connection...")
    await close_db()
    logger.info("Database connection closed.")

# Exception handler for uniform error responses
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )

# Include routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(agent_router, prefix="/api/v1/agent")

@app.get("/", tags=["Root"])
async def read_root():
    """Root endpoint providing basic API information"""
    return {
        "message": "Welcome to the Print-on-Demand AI Agent API",
        "version": app.version,
        "docs_url": "/docs",
    }


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time communication"""
    try:
        await websocket_manager.connect(websocket, user_id)
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Receive messages from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

                elif message.get("type") == "ui_event":
                    # Get database session for this request
                    async for db in get_db():
                        try:
                            # Process UI event
                            from core.session_manager import SessionManager
                            from agent.graph import app_graph
                            from langchain_core.messages import HumanMessage
                    
                            event_result = await SessionManager.add_ui_event(
                                db=db,
                                user_id=UUID(user_id),
                                event_type=message.get("event_type"),
                                event_data=message.get("data", {})
                            )
                    
                            # If event should trigger agent response
                            if event_result.get("should_trigger"):
                                prompt = event_result.get("prompt")
                                if prompt:
                                    inputs = {
                                        "messages": [HumanMessage(content=prompt)],
                                        "user_id": user_id,
                                        "db": db
                                    }
                            
                                    result = await app_graph.ainvoke(inputs)
                                    response_message = result["messages"][-1].content
                            
                                    await SessionManager.add_conversation_message(
                                        db, UUID(user_id), "assistant", response_message
                                    )
                            
                                    await websocket.send_text(json.dumps({
                                        "type": "agent_response",
                                        "content": response_message,
                                        "triggered_by": event_result.get("event_type")
                                    }))
                    
                            await websocket.send_text(json.dumps({
                                "type": "event_acknowledged",
                                "event_type": message.get("event_type"),
                                "triggered_agent": event_result.get("should_trigger", False)
                            }))
                        finally:
                            break  # Exit the async for loop after one iteration
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for user: {user_id}")
                break
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from websocket user: {user_id}")
            except Exception as e:
                logger.error(f"WebSocket error for user {user_id}: {e}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket connection error for user {user_id}: {e}")
    finally:
        websocket_manager.disconnect(user_id)


if __name__ == "__main__":
    logger.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning",
    )
