from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.dialects.postgresql import insert
import logging
import json

from database import UserSession, User, Order
from auth.utils import generate_order_number
from ui_event_manager import UIEventManager, UIEventType

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages user sessions and conversation state"""
    
    @staticmethod
    async def get_user_session(db: AsyncSession, user_id: UUID) -> Optional[UserSession]:
        """Get user's current session"""
        try:
            query = select(UserSession).where(UserSession.user_id == user_id)
            result = await db.execute(query)
            session = result.scalar_one_or_none()
            
            if session:
                logger.info(f"Retrieved session for user {user_id}: state={session.agent_state}")
            else:
                logger.info(f"No existing session found for user {user_id}")
            
            return session
        except Exception as e:
            logger.error(f"Error retrieving session for user {user_id}: {e}")
            return None
    
    @staticmethod
    async def create_or_update_session(
        db: AsyncSession, 
        user_id: UUID,
        agent_state: str = "no_product_selected",
        conversation_history: Optional[List[Dict]] = None,
        product_context: Optional[Dict] = None,
        design_urls: Optional[Dict] = None,
        session_metadata: Optional[Dict] = None
    ) -> UserSession:
        """Create new session or update existing one"""
        try:
            if conversation_history is None:
                conversation_history = []
            
            # Use upsert to handle concurrent requests
            stmt = insert(UserSession).values(
                user_id=user_id,
                agent_state=agent_state,
                conversation_history=conversation_history,
                product_context=product_context,
                design_urls=design_urls,
                session_metadata=session_metadata,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            # On conflict, update the existing record
            stmt = stmt.on_conflict_do_update(
                index_elements=['user_id'],
                set_=dict(
                    agent_state=stmt.excluded.agent_state,
                    conversation_history=stmt.excluded.conversation_history,
                    product_context=stmt.excluded.product_context,
                    design_urls=stmt.excluded.design_urls,
                    session_metadata=stmt.excluded.session_metadata,
                    updated_at=datetime.now(timezone.utc)
                )
            ).returning(UserSession)
            
            result = await db.execute(stmt)
            session = result.scalar_one()
            await db.commit()
            
            logger.info(f"Session created/updated for user {user_id}: state={agent_state}")
            return session
            
        except Exception as e:
            logger.error(f"Error creating/updating session for user {user_id}: {e}")
            await db.rollback()
            raise
    
    @staticmethod
    async def update_session_state(
        db: AsyncSession, 
        user_id: UUID, 
        agent_state: str
    ) -> bool:
        """Update only the agent state"""
        try:
            stmt = update(UserSession).where(
                UserSession.user_id == user_id
            ).values(
                agent_state=agent_state,
                updated_at=datetime.now(timezone.utc)
            )
            
            result = await db.execute(stmt)
            await db.commit()
            
            if result.rowcount > 0:
                logger.info(f"Updated session state for user {user_id}: {agent_state}")
                return True
            else:
                logger.warning(f"No session found to update for user {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating session state for user {user_id}: {e}")
            await db.rollback()
            return False
    
    @staticmethod
    async def add_conversation_message(
        db: AsyncSession, 
        user_id: UUID, 
        role: str, 
        content: str
    ) -> bool:
        """Add a message to conversation history"""
        try:
            session = await SessionManager.get_user_session(db, user_id)
            if not session:
                # Create new session if none exists
                session = await SessionManager.create_or_update_session(db, user_id)
            
            # Add new message to conversation history
            new_message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            conversation_history = session.conversation_history.copy() if session.conversation_history else []
            conversation_history.append(new_message)
            
            # Keep only last 50 messages to prevent database bloat
            if len(conversation_history) > 50:
                conversation_history = conversation_history[-50:]
            
            # Update session
            stmt = update(UserSession).where(
                UserSession.user_id == user_id
            ).values(
                conversation_history=conversation_history,
                updated_at=datetime.now(timezone.utc)
            )
            
            await db.execute(stmt)
            await db.commit()
            
            logger.info(f"Added {role} message to conversation for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding conversation message for user {user_id}: {e}")
            await db.rollback()
            return False
    
    @staticmethod
    async def update_product_context(
        db: AsyncSession, 
        user_id: UUID, 
        product_context: Dict[str, Any]
    ) -> bool:
        """Update product context in session"""
        try:
            stmt = update(UserSession).where(
                UserSession.user_id == user_id
            ).values(
                product_context=product_context,
                agent_state="product_selected",
                updated_at=datetime.now(timezone.utc)
            )
            
            result = await db.execute(stmt)
            await db.commit()
            
            if result.rowcount > 0:
                logger.info(f"Updated product context for user {user_id}")
                return True
            else:
                # Create session if none exists
                await SessionManager.create_or_update_session(
                    db, user_id, 
                    agent_state="product_selected",
                    product_context=product_context
                )
                return True
                
        except Exception as e:
            logger.error(f"Error updating product context for user {user_id}: {e}")
            await db.rollback()
            return False
    
    @staticmethod
    async def update_design_urls(
        db: AsyncSession, 
        user_id: UUID, 
        design_url: Optional[str] = None,
        upscaled_url: Optional[str] = None
    ) -> bool:
        """Update design URLs in session"""
        try:
            session = await SessionManager.get_user_session(db, user_id)
            if not session:
                logger.error(f"No session found for user {user_id}")
                return False
            
            # Get current design URLs or create new dict
            design_urls = session.design_urls.copy() if session.design_urls else {}
            
            if design_url is not None:
                design_urls["design_url"] = design_url
            if upscaled_url is not None:
                design_urls["upscaled_url"] = upscaled_url
            
            # Update agent state based on what URLs we have
            new_state = "design_created" if design_url else session.agent_state
            
            stmt = update(UserSession).where(
                UserSession.user_id == user_id
            ).values(
                design_urls=design_urls,
                agent_state=new_state,
                updated_at=datetime.now(timezone.utc)
            )
            
            await db.execute(stmt)
            await db.commit()
            
            logger.info(f"Updated design URLs for user {user_id}: design={design_url}, upscaled={upscaled_url}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating design URLs for user {user_id}: {e}")
            await db.rollback()
            return False
    
    @staticmethod
    async def clear_session(db: AsyncSession, user_id: UUID) -> bool:
        """Clear user session (reset to initial state)"""
        try:
            stmt = update(UserSession).where(
                UserSession.user_id == user_id
            ).values(
                agent_state="no_product_selected",
                conversation_history=[],
                product_context=None,
                design_urls=None,
                session_metadata=None,
                updated_at=datetime.now(timezone.utc)
            )
            
            result = await db.execute(stmt)
            await db.commit()
            
            if result.rowcount > 0:
                logger.info(f"Cleared session for user {user_id}")
                return True
            else:
                # Create fresh session if none exists
                await SessionManager.create_or_update_session(db, user_id)
                logger.info(f"Created fresh session for user {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing session for user {user_id}: {e}")
            await db.rollback()
            return False
    
    @staticmethod
    async def create_order(
        db: AsyncSession,
        user_id: UUID,
        product_data: Dict[str, Any],
        design_data: Dict[str, Any],
        user_selection: Dict[str, Any],
        customer_info: Optional[Dict[str, Any]] = None,
        pricing: Optional[Dict[str, Any]] = None
    ) -> Optional[Order]:
        """Create an order record with complete order information"""
        try:
            session = await SessionManager.get_user_session(db, user_id)
        
            # Generate unique order number
            order_number = generate_order_number()
        
            # Create comprehensive order data
            complete_order_data = {
                "product_data": product_data,
                "design_data": design_data,
                "user_selection": user_selection,
            }
        
            if customer_info:
                complete_order_data["customer_info"] = customer_info
            if pricing:
                complete_order_data["pricing"] = pricing

            # Create order record
            new_order = Order(
                user_id=user_id,
                session_id=session.id if session else None,
                order_number=order_number,
                product_data=product_data,
                design_data=design_data,
                user_selection=user_selection,
                status="pending",
                payment_status="pending",
                total_amount=str(pricing.get("total", 0)) if pricing else None,
                tracking_info=complete_order_data  # Store all order details here
            )
        
            db.add(new_order)
            await db.flush()
            await db.refresh(new_order)
            await db.commit()
        
            logger.info(f"Created order {order_number} for user {user_id}")
            return new_order
        
        except Exception as e:
            logger.error(f"Error creating order for user {user_id}: {e}")
            await db.rollback()
            return None

    
    @staticmethod
    async def get_user_orders(db: AsyncSession, user_id: UUID, limit: int = 10) -> List[Order]:
        """Get user's order history"""
        try:
            query = select(Order).where(
                Order.user_id == user_id
            ).order_by(Order.created_at.desc()).limit(limit)
            
            result = await db.execute(query)
            orders = result.scalars().all()
            
            logger.info(f"Retrieved {len(orders)} orders for user {user_id}")
            return list(orders)
            
        except Exception as e:
            logger.error(f"Error retrieving orders for user {user_id}: {e}")
            return []
    
    @staticmethod
    async def get_session_context_for_agent(db: AsyncSession, user_id: UUID) -> Dict[str, Any]:
        """Get complete session context for agent processing"""
        try:
            session = await SessionManager.get_user_session(db, user_id)
            orders = await SessionManager.get_user_orders(db, user_id, limit=5)
            
            if not session:
                # Create default session
                session = await SessionManager.create_or_update_session(db, user_id)
            
            # Format orders for agent context
            orders_history = []
            for i, order in enumerate(orders, 1):
                orders_history.append({
                    "order_number": i,
                    "order_id": order.order_number,
                    "product_name": order.product_data.get("name", "Unknown Product"),
                    "timestamp": order.created_at.isoformat(),
                    "status": order.status
                })
            
            return {
                "agent_state": session.agent_state,
                "conversation_history": session.conversation_history or [],
                "product_context": session.product_context,
                "design_urls": session.design_urls,
                "orders_history": orders_history,
                "session_metadata": session.session_metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting session context for user {user_id}: {e}")
            # Return minimal context in case of error
            return {
                "agent_state": "no_product_selected",
                "conversation_history": [],
                "product_context": None,
                "design_urls": None,
                "orders_history": [],
                "session_metadata": None
            }
    
    @staticmethod
    async def cleanup_old_sessions(db: AsyncSession, days_old: int = 30) -> int:
        """Cleanup old inactive sessions (maintenance function)"""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
            
            stmt = delete(UserSession).where(
                UserSession.updated_at < cutoff_date
            )
            
            result = await db.execute(stmt)
            deleted_count = result.rowcount
            await db.commit()
            
            logger.info(f"Cleaned up {deleted_count} old sessions")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")
            await db.rollback()
            return 0

    
    @staticmethod
    async def add_ui_event(
        db: AsyncSession,
        user_id: UUID,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add UI event to conversation history and determine if agent should respond
    
        Returns:
            Dict with 'should_trigger' bool and optional 'prompt' for agent
        """
        try:
            # Convert string to UIEventType
            try:
                event_type_enum = UIEventType(event_type)
            except ValueError:
                logger.warning(f"Unknown UI event type: {event_type}")
                return {"should_trigger": False}
        
            # Get current session
            session = await SessionManager.get_user_session(db, user_id)
            if not session:
                session = await SessionManager.create_or_update_session(db, user_id)
        
            # Get recent UI events for pattern detection
            recent_events = []
            if session.conversation_history:
                recent_events = [
                    msg for msg in session.conversation_history[-10:]
                    if msg.get("role") == "ui_event"
                ]
        
            # Format event for history
            formatted_event = UIEventManager.format_event_for_history(
                event_type_enum,
                event_data
            )
        
            # Add to conversation history
            conversation_history = session.conversation_history.copy() if session.conversation_history else []
            conversation_history.append(formatted_event)
        
            # Keep only last 50 messages
            if len(conversation_history) > 50:
                conversation_history = conversation_history[-50:]
        
            # Update session
            stmt = update(UserSession).where(
                UserSession.user_id == user_id
            ).values(
                conversation_history=conversation_history,
                updated_at=datetime.now(timezone.utc)
            )
        
            await db.execute(stmt)
            await db.commit()
        
            # Determine if agent should be triggered
            should_trigger = UIEventManager.should_trigger_agent(
                event_type_enum,
                event_data,
                recent_events
            )
        
            # Get session context for prompt generation
            session_context = await SessionManager.get_session_context_for_agent(db, user_id)
        
            # Generate agent prompt if needed
            agent_prompt = None
            if should_trigger:
                agent_prompt = UIEventManager.generate_agent_prompt_for_event(
                    event_type_enum,
                    event_data,
                    session_context
                )
        
            logger.info(f"UI event {event_type} added for user {user_id}, trigger={should_trigger}")
        
            return {
                "should_trigger": should_trigger,
                "prompt": agent_prompt,
                "event_type": event_type,
                "priority": UIEventManager.get_event_priority(event_type_enum).value
            }
        
        except Exception as e:
            logger.error(f"Error adding UI event for user {user_id}: {e}")
            await db.rollback()
            return {"should_trigger": False}

    @staticmethod
    async def get_recent_ui_events(
        db: AsyncSession,
        user_id: UUID,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent UI events from user's conversation history"""
        try:
            session = await SessionManager.get_user_session(db, user_id)
            if not session or not session.conversation_history:
                return []
        
            return await UIEventManager.get_recent_ui_events(
                session.conversation_history,
                limit
            )
        except Exception as e:
            logger.error(f"Error getting recent UI events for user {user_id}: {e}")
            return []