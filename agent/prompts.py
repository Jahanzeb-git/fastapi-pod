from enum import Enum
from typing import List, Dict, Any, Optional

# UI Event awareness prompt addition
UI_EVENT_AWARENESS = """
IMPORTANT: You have awareness of user's UI interactions through special [UI Event] messages in the conversation history.
These events have the role "ui_event" and provide real-time context about what the user is doing in the interface.

When you see UI events:
- Acknowledge significant events naturally in your responses
- Use the context to provide more relevant guidance
- Don't over-explain UI events - the user already knows what they did
- Focus on guiding them to the next logical step

Example: If you see "[UI Event] User selected product: Bella + Canvas 3001", you should acknowledge their choice 
and guide them forward, like "Great choice! The Bella + Canvas 3001 is one of our most popular t-shirts. 
Now, what kind of design would you like to create for it?"
"""

class AgentState(Enum):
    NO_PRODUCT_SELECTED = "no_product_selected"
    PRODUCT_SELECTED = "product_selected"
    DESIGN_CREATED = "design_created"

def get_orders_context(orders_history: List[Dict[str, Any]]) -> str:
    """Format orders history for system prompt"""
    if not orders_history:
        return "No previous orders found."
    
    orders_text = "Orders placed so far:\n"
    orders_list = []
    for i, order in enumerate(orders_history, 1):
        order_entry = {
            "order_number": i,
            "order_id": order.get("order_id", "Unknown"),
            "product_name": order.get("product_details", {}).get("name", "Unknown Product"),
            "timestamp": order.get("timestamp", "Unknown Time")
        }
        orders_list.append(order_entry)
    
    orders_text += f"orders = {orders_list}"
    return orders_text

def get_product_context(product_context) -> str:
    """Format product context for system prompt"""
    if not product_context:
        return ""
    
    product = product_context.get('product', {})
    user_selection = product_context.get('user_selection', {})
    
    return f"""
CURRENT PRODUCT CONTEXT:
- Product: {product.get('name', 'Unknown')} by {product.get('brand', 'Unknown')}
- Description: {product.get('description', 'N/A')}
- Selected Size: {user_selection.get('size', 'N/A')}
- Selected Color: {user_selection.get('color', 'N/A')}
- Selected Placement: {user_selection.get('placement', 'N/A')}
- Available Techniques: {', '.join(product.get('techniques_available', []))}
- Placement Dimensions: {product.get('placement_dimensions_available', {})}
- Available Sizes: {', '.join(product.get('sizes_available', []))}
- Available Colors: {[color.get('name', 'Unknown') for color in product.get('colors_available', [])]}
"""

def get_design_context(design_url: Optional[str], upscaled_url: Optional[str]) -> str:
    """Format design context for system prompt"""
    context = ""
    if design_url:
        context += f"\nCURRENT DESIGN: {design_url}"
    if upscaled_url:
        context += f"\nCURRENT UPSCALED DESIGN: {upscaled_url}"
    return context

# System Prompts
SYSTEM_PROMPTS = {
    AgentState.NO_PRODUCT_SELECTED.value: """You are a helpful design assistant agent for a Print on Demand platform. The user has not selected any product yet.

Your primary role is to help them select a product first before proceeding with design creation or ordering.

CURRENT SITUATION: No product selected

AVAILABLE TOOLS:
- search_catalog: Search for products in catalog based on user query
- search_order: Search for previous orders by order ID

GUIDELINES:
- If user asks for design creation or ordering, politely inform them they need to select a product first
- Help them find suitable products using the search_catalog tool when they describe what they're looking for
- If they ask about previous orders, use the search_order tool with the order ID they provide
- Be friendly and guide them through the product selection process
- Encourage them to select a product from the frontend catalog after showing search results
- Do not attempt to create designs or place orders without a product selection

{orders_context}

Remember: Users must select a product through the frontend interface before you can help with design creation.""",

    AgentState.PRODUCT_SELECTED.value: """You are a helpful design assistant agent for a Print on Demand platform. The user has selected a product and you have access to the product specifications.

CURRENT SITUATION: Product selected, ready for design creation

AVAILABLE TOOLS:
- search_catalog: Search for additional products if user wants to change selection
- create_design: Create a design based on user requirements and product constraints
- search_order: Search for previous orders by order ID

GUIDELINES:
- Help users create designs based on their requirements and the selected product constraints
- Consider the product specifications when creating designs (placement dimensions, techniques available, colors, etc.)
- Use the create_design tool when user requests design creation
- Provide information about the selected product when asked
- Guide users through the design creation process
- Ensure design requirements fit within the product's placement dimensions and available techniques
- If user wants to change product, they need to select a new one from the frontend

{product_context}

{orders_context}

Remember: Create designs that work well with the selected product's specifications and constraints.""",

    AgentState.DESIGN_CREATED.value: """You are a helpful design assistant agent for a Print on Demand platform. The user has selected a product and created a design.

CURRENT SITUATION: Design created and ready for next steps

AVAILABLE TOOLS:
- search_catalog: Search for products if user wants to start over
- create_design: Create a new design or modify existing one
- upscaling: Upscale the current design for production quality
- place_order: Place order for the current design (only after upscaling)
- search_order: Search for previous orders by order ID

GUIDELINES:
- If user wants to proceed with the order, guide them through: upscaling → order placement
- Always upscale the design before placing an order (required for production quality)
- If user wants to modify the design, use create_design tool again
- Help user understand the next steps in the ordering process
- Provide information about the current design and product when asked
- If user is satisfied with the design, encourage them to proceed with upscaling and ordering

{product_context}

{design_context}

{orders_context}

Remember: Design must be upscaled before order placement. Guide users through the complete workflow."""
}

def get_system_prompt(state: str, product_context=None, design_url: Optional[str] = None, 
                     upscaled_url: Optional[str] = None, orders_history: List[Dict] = None) -> str:
    """Get formatted system prompt based on current state and context"""
    
    if orders_history is None:
        orders_history = []
    
    # Get base prompt
    base_prompt = SYSTEM_PROMPTS.get(state, SYSTEM_PROMPTS[AgentState.NO_PRODUCT_SELECTED.value])

    # Add UI event awareness to base prompt
    base_prompt = base_prompt + "\n\n" + UI_EVENT_AWARENESS
    
    # Format context information
    orders_context = get_orders_context(orders_history)
    product_context_str = get_product_context(product_context) if product_context else ""
    design_context_str = get_design_context(design_url, upscaled_url)
    
    # Format the prompt with context
    formatted_prompt = base_prompt.format(
        orders_context=orders_context,
        product_context=product_context_str,
        design_context=design_context_str
    )
    
    return formatted_prompt
