from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import json

logger = logging.getLogger(__name__)

class UIEventType(Enum):
    """UI Event types that can be tracked"""
    # Product browsing events
    PRODUCT_VIEWED = "product_viewed"
    PRODUCT_SELECTED = "product_selected"
    PRODUCT_DESELECTED = "product_deselected"
    
    # Design events
    DESIGN_TOOL_OPENED = "design_tool_opened"
    DESIGN_TOOL_CLOSED = "design_tool_closed"
    DESIGN_MODIFIED = "design_modified"
    DESIGN_SAVED = "design_saved"
    
    # Selection events
    COLOR_CHANGED = "color_changed"
    SIZE_SELECTED = "size_selected"
    PLACEMENT_SELECTED = "placement_selected"
    
    # Order flow events
    CART_VIEWED = "cart_viewed"
    CHECKOUT_STARTED = "checkout_started"
    ORDER_UI_OPENED = "order_ui_opened"
    ORDER_UI_CLOSED = "order_ui_closed"
    
    # Navigation events
    PAGE_NAVIGATED = "page_navigated"
    MODAL_OPENED = "modal_opened"
    MODAL_CLOSED = "modal_closed"
    
    # Interaction events
    HELP_REQUESTED = "help_requested"
    FILTER_APPLIED = "filter_applied"
    SEARCH_PERFORMED = "search_performed"

class UIEventPriority(Enum):
    """Priority levels for UI events"""
    LOW = "low"  # Just track in history
    MEDIUM = "medium"  # Track and maybe notify
    HIGH = "high"  # Track and trigger agent response
    CRITICAL = "critical"  # Immediate agent intervention

class UIEventManager:
    """Manages UI event processing and agent triggering"""
    
    # Define which events should trigger agent responses
    TRIGGER_EVENTS = {
        UIEventType.PRODUCT_SELECTED: UIEventPriority.HIGH,
        UIEventType.DESIGN_SAVED: UIEventPriority.HIGH,
        UIEventType.HELP_REQUESTED: UIEventPriority.CRITICAL,
        UIEventType.CHECKOUT_STARTED: UIEventPriority.HIGH,
        UIEventType.PRODUCT_DESELECTED: UIEventPriority.MEDIUM,
    }
    
    # Define events that should only be tracked in history
    TRACK_ONLY_EVENTS = {
        UIEventType.PRODUCT_VIEWED: UIEventPriority.LOW,
        UIEventType.COLOR_CHANGED: UIEventPriority.LOW,
        UIEventType.SIZE_SELECTED: UIEventPriority.LOW,
        UIEventType.MODAL_OPENED: UIEventPriority.LOW,
        UIEventType.MODAL_CLOSED: UIEventPriority.LOW,
        UIEventType.PAGE_NAVIGATED: UIEventPriority.LOW,
        UIEventType.DESIGN_TOOL_OPENED: UIEventPriority.MEDIUM,
        UIEventType.DESIGN_TOOL_CLOSED: UIEventPriority.LOW,
    }
    
    @staticmethod
    def get_event_priority(event_type: UIEventType) -> UIEventPriority:
        """Get priority level for an event type"""
        if event_type in UIEventManager.TRIGGER_EVENTS:
            return UIEventManager.TRIGGER_EVENTS[event_type]
        return UIEventManager.TRACK_ONLY_EVENTS.get(event_type, UIEventPriority.LOW)
    
    @staticmethod
    def should_trigger_agent(
        event_type: UIEventType, 
        context: Optional[Dict[str, Any]] = None,
        recent_events: Optional[List[Dict]] = None
    ) -> bool:
        """
        Determine if an event should trigger agent response
        
        Args:
            event_type: Type of UI event
            context: Additional context about the event
            recent_events: Recent UI events for pattern detection
            
        Returns:
            Boolean indicating if agent should be triggered
        """
        priority = UIEventManager.get_event_priority(event_type)
        
        # High and Critical priority events always trigger
        if priority in [UIEventPriority.HIGH, UIEventPriority.CRITICAL]:
            return True
        
        # Medium priority events may trigger based on context
        if priority == UIEventPriority.MEDIUM:
            # Add contextual rules here
            if event_type == UIEventType.PRODUCT_DESELECTED:
                # Trigger if user deselected after spending time
                if context and context.get("time_spent_seconds", 0) > 30:
                    return True
        
        # Check for patterns that might indicate user confusion
        if recent_events and len(recent_events) >= 3:
            # If user rapidly changed colors/sizes, they might need help
            recent_types = [e.get("event_type") for e in recent_events[-5:]]
            if recent_types.count(UIEventType.COLOR_CHANGED.value) >= 3:
                return True
            if recent_types.count(UIEventType.SIZE_SELECTED.value) >= 3:
                return True
        
        return False
    
    @staticmethod
    def format_event_for_history(
        event_type: UIEventType,
        event_data: Dict[str, Any],
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format UI event for conversation history
        
        Args:
            event_type: Type of UI event
            event_data: Event payload data
            timestamp: Event timestamp
            
        Returns:
            Formatted event for conversation history
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        
        # Create human-readable event description
        description = UIEventManager._generate_event_description(event_type, event_data)
        
        return {
            "role": "ui_event",
            "content": f"[UI Event] {description}",
            "timestamp": timestamp,
            "event_type": event_type.value,
            "event_data": event_data,
            "priority": UIEventManager.get_event_priority(event_type).value
        }
    
    @staticmethod
    def _generate_event_description(
        event_type: UIEventType,
        event_data: Dict[str, Any]
    ) -> str:
        """Generate human-readable description of UI event"""
        
        descriptions = {
            UIEventType.PRODUCT_SELECTED: lambda d: f"User selected product: {d.get('product_name', 'Unknown')}",
            UIEventType.PRODUCT_VIEWED: lambda d: f"User viewed product: {d.get('product_name', 'Unknown')}",
            UIEventType.PRODUCT_DESELECTED: lambda d: f"User deselected product after {d.get('time_spent_seconds', 0)} seconds",
            UIEventType.DESIGN_TOOL_OPENED: lambda d: "User opened design tool",
            UIEventType.DESIGN_TOOL_CLOSED: lambda d: "User closed design tool",
            UIEventType.DESIGN_SAVED: lambda d: f"User saved design: {d.get('design_name', 'Untitled')}",
            UIEventType.COLOR_CHANGED: lambda d: f"User changed color to: {d.get('color_name', d.get('color_hex', 'Unknown'))}",
            UIEventType.SIZE_SELECTED: lambda d: f"User selected size: {d.get('size', 'Unknown')}",
            UIEventType.PLACEMENT_SELECTED: lambda d: f"User selected placement: {d.get('placement', 'Unknown')}",
            UIEventType.CART_VIEWED: lambda d: "User viewed cart",
            UIEventType.CHECKOUT_STARTED: lambda d: "User started checkout process",
            UIEventType.ORDER_UI_OPENED: lambda d: "User opened order placement UI",
            UIEventType.ORDER_UI_CLOSED: lambda d: "User closed order placement UI",
            UIEventType.HELP_REQUESTED: lambda d: f"User requested help: {d.get('help_topic', 'General')}",
            UIEventType.SEARCH_PERFORMED: lambda d: f"User searched for: {d.get('query', '')}",
            UIEventType.FILTER_APPLIED: lambda d: f"User applied filter: {d.get('filter_type', 'Unknown')}",
            UIEventType.PAGE_NAVIGATED: lambda d: f"User navigated to: {d.get('page', 'Unknown')}",
            UIEventType.MODAL_OPENED: lambda d: f"User opened modal: {d.get('modal_type', 'Unknown')}",
            UIEventType.MODAL_CLOSED: lambda d: f"User closed modal: {d.get('modal_type', 'Unknown')}",
        }
        
        generator = descriptions.get(event_type)
        if generator:
            try:
                return generator(event_data)
            except Exception as e:
                logger.error(f"Error generating event description: {e}")
                return f"{event_type.value} occurred"
        
        return f"{event_type.value} occurred"
    
    @staticmethod
    def generate_agent_prompt_for_event(
        event_type: UIEventType,
        event_data: Dict[str, Any],
        session_context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate specific prompt for agent based on UI event
        
        Args:
            event_type: Type of UI event
            event_data: Event payload data
            session_context: Current session context
            
        Returns:
            Prompt for agent or None if no specific prompt needed
        """
        
        prompts = {
            UIEventType.PRODUCT_SELECTED: lambda d, c: (
                f"The user just selected {d.get('product_name', 'a product')}. "
                f"Acknowledge their choice positively and guide them to the next step "
                f"(which is typically creating a design). Be enthusiastic but concise."
            ),
            UIEventType.DESIGN_SAVED: lambda d, c: (
                f"The user has saved their design. Compliment their work and ask if they'd "
                f"like to proceed with upscaling for production quality or make any adjustments."
            ),
            UIEventType.HELP_REQUESTED: lambda d, c: (
                f"The user has requested help with: {d.get('help_topic', 'using the platform')}. "
                f"Provide clear, helpful guidance based on their current state in the process."
            ),
            UIEventType.CHECKOUT_STARTED: lambda d, c: (
                f"The user has started the checkout process. Guide them through completing "
                f"their order and let them know what information they'll need."
            ),
            UIEventType.PRODUCT_DESELECTED: lambda d, c: (
                f"The user deselected the product after {d.get('time_spent_seconds', 0)} seconds. "
                f"Ask if they'd like help finding something else or if they had concerns about the product."
            ) if d.get('time_spent_seconds', 0) > 30 else None,
        }
        
        generator = prompts.get(event_type)
        if generator:
            try:
                return generator(event_data, session_context)
            except Exception as e:
                logger.error(f"Error generating agent prompt: {e}")
                return None
        
        return None
    
    @staticmethod
    async def get_recent_ui_events(
        conversation_history: List[Dict[str, Any]],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent UI events from conversation history
        
        Args:
            conversation_history: Full conversation history
            limit: Maximum number of recent events to return
            
        Returns:
            List of recent UI events
        """
        ui_events = [
            msg for msg in conversation_history 
            if msg.get("role") == "ui_event"
        ]
        return ui_events[-limit:] if ui_events else []
