import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from agent.graph import app_graph
from core.session_manager import SessionManager

logger = logging.getLogger(__name__)

class StreamingHandler:
    """Handles streaming responses from the agent for both SSE and WebSocket"""
    
    @staticmethod
    async def stream_agent_response(
        message: str,
        user_id: str,
        db: AsyncSession,
        is_ui_event: bool = False,
        event_metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream agent response token by token
        
        Args:
            message: User message or UI event prompt
            user_id: User ID
            db: Database session
            is_ui_event: Whether this is triggered by UI event
            event_metadata: Additional metadata for UI events
            
        Yields:
            Dict containing token and metadata
        """
        try:
            # Store the user message first (unless it's a UI event, which is already stored)
            if not is_ui_event:
                await SessionManager.add_conversation_message(
                    db, UUID(user_id), "user", message
                )
            
            # Prepare input for agent
            inputs = {
                "messages": [HumanMessage(content=message)],
                "user_id": user_id,
                "db": db
            }
            
            # Stream tokens from agent
            full_response = ""
            async for chunk in app_graph.astream(inputs):
                # Extract the AI message from the chunk
                if "agent" in chunk and chunk["agent"].get("messages"):
                    ai_message = chunk["agent"]["messages"][-1]
                    
                    if isinstance(ai_message, AIMessage):
                        # Get the content delta
                        content = ai_message.content
                        if content and len(content) > len(full_response):
                            # Extract new tokens
                            new_content = content[len(full_response):]
                            full_response = content
                            
                            yield {
                                "type": "token",
                                "content": new_content,
                                "is_ui_event": is_ui_event,
                                "metadata": event_metadata
                            }
                
                # Handle tool calls if present
                elif "tools" in chunk and chunk["tools"].get("messages"):
                    tool_message = chunk["tools"]["messages"][-1]
                    yield {
                        "type": "tool_call",
                        "content": str(tool_message.content),
                        "tool_name": getattr(tool_message, 'name', 'unknown'),
                        "is_ui_event": is_ui_event,
                        "metadata": event_metadata
                    }
            
            # Store the complete response in database
            if full_response:
                await SessionManager.add_conversation_message(
                    db, UUID(user_id), "assistant", full_response
                )
                
                # Get updated session context
                session_context = await SessionManager.get_session_context_for_agent(
                    db, UUID(user_id)
                )
                
                yield {
                    "type": "complete",
                    "content": full_response,
                    "session_context": {
                        "agent_state": session_context["agent_state"],
                        "has_product": session_context["product_context"] is not None,
                        "has_design": bool(session_context["design_urls"] and 
                                         session_context["design_urls"].get("design_url")),
                        "has_upscaled": bool(session_context["design_urls"] and 
                                           session_context["design_urls"].get("upscaled_url"))
                    },
                    "is_ui_event": is_ui_event,
                    "metadata": event_metadata
                }
                
        except Exception as e:
            logger.error(f"Streaming error for user {user_id}: {e}")
            yield {
                "type": "error",
                "content": "An error occurred while processing your request",
                "error": str(e),
                "is_ui_event": is_ui_event,
                "metadata": event_metadata
            }
