from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import json
import logging
from uuid import UUID

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for real-time communication"""
    
    def __init__(self):
        # Store active connections: {user_id: websocket}
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept websocket connection and store it"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"WebSocket connected for user: {user_id}")
    
    def disconnect(self, user_id: str):
        """Remove websocket connection"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"WebSocket disconnected for user: {user_id}")
    
    async def send_signal(self, user_id: str, signal_type: str, data: Optional[Dict[str, Any]] = None):
        """Send signal to specific user"""
        if user_id in self.active_connections:
            websocket = self.active_connections[user_id]
            message = {
                "type": signal_type,
                "data": data or {},
                "timestamp": str(datetime.now(timezone.utc))
            }
            try:
                await websocket.send_text(json.dumps(message))
                logger.info(f"Signal '{signal_type}' sent to user: {user_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to send signal to user {user_id}: {e}")
                self.disconnect(user_id)
                return False
        else:
            logger.warning(f"No active connection for user: {user_id}")
            return False
    
    async def send_order_ui_trigger(self, user_id: str, order_context: Dict[str, Any]):
        """Specifically trigger order UI for user"""
        return await self.send_signal(
            user_id, 
            "trigger_order_ui", 
            {
                "product": order_context.get("product"),
                "design": order_context.get("design"), 
                "selection": order_context.get("selection"),
                "message": "Please complete your order details"
            }
        )
    
    def get_connected_users(self) -> List[str]:
        """Get list of connected user IDs"""
        return list(self.active_connections.keys())

# Global websocket manager instance
websocket_manager = WebSocketManager()
