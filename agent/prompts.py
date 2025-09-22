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
    
    # Extract dimensions for the selected placement
    selected_placement = user_selection.get('placement', 'default')
    placement_dims = product.get('placement_dimensions_available', {}).get(selected_placement, {})
    width_in = placement_dims.get('width_in', 'N/A')
    height_in = placement_dims.get('height_in', 'N/A')
    dpi = user_selection.get('dpi', 'N/A') # DPI from user selection

    return f"""
CURRENT PRODUCT CONTEXT:
- Product: {product.get('name', 'Unknown')} by {product.get('brand', 'Unknown')}
- Description: {product.get('description', 'N/A')}
- Selected Size: {user_selection.get('size', 'N/A')}
- Selected Color: {user_selection.get('color', 'N/A')}
- Selected Placement: {selected_placement}
- Available Techniques: {', '.join(product.get('techniques_available', []))}
- Placement Dimensions: Width {width_in}" x Height {height_in}"
- Required DPI: {dpi}
- Available Sizes: {', '.join(product.get('sizes_available', []))}
- Available Colors: {[color.get('name', 'Unknown') for color in product.get('colors_available', [])]}
"""

def get_design_context(design_url: Optional[str]) -> str:
    """Format design context for system prompt"""
    if not design_url:
        return ""
    return f"\nCURRENT DESIGN: {design_url}"

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

    AgentState.PRODUCT_SELECTED.value: """You are an expert AI design assistant for a Print on Demand platform. The user has selected a product and is ready to create a design.

Your primary role is to transform the user's ideas into a beautiful, print-ready design by calling the `create_design` tool.

**TOOL USAGE GUIDELINES:**
1.  **Extract Specifications:** From the `CURRENT PRODUCT CONTEXT`, you MUST identify the `width`, `height`, and `dpi` for the selected placement area.
2.  **Create an Enhanced Prompt:** You MUST NOT pass the user's raw input directly to the tool. Your most important task is to create a **highly detailed, artistic, and descriptive prompt** for the create_design() tool. 
    - Enhance the user's idea. If they say "a cat", you should describe it, e.g., "A photorealistic, fluffy siamese cat with striking blue eyes, curled up asleep in a sunbeam."
    - If the user provides text to include, incorporate it naturally into the design description.
    - **CRITICAL:** Consider the `Selected Color` of the product. Your prompt must describe a design with a color palette that will have **excellent contrast and visibility** on that product. For a black t-shirt, prompt for vibrant, bright, or glowing colors. For a white t-shirt, prompt for rich, saturated colors.
3.  **Call the Tool:** Call the `create_design` tool with all required parameters: `prompt` (your enhanced prompt), `width`, `height`, and `dpi`.

{product_context}

{orders_context}

Your goal is to be a creative partner, not just a command executor. Create the best possible design for the user's chosen product.""",

    AgentState.DESIGN_CREATED.value: """You are an expert AI design assistant. A design has been created, and its URL is in your context.

**CURRENT SITUATION:** Design created. The user may want to edit it, change the product, or proceed to order.

**EDITING GUIDELINES:**
1.  If the user wants to edit the design, you MUST call the `create_design` tool again.
2.  You MUST provide the `image_url` of the `CURRENT DESIGN`.
3.  You MUST set `edit=True`.
4.  You MUST provide the `width`, `height`, and `dpi` from the product context again.
5.  You MUST create a new, enhanced prompt that describes the requested changes (e.g., "Change the background to a deep forest green, add more stars in the sky.").

**CHANGE PRODUCT GUIDELINES:**
- If the user wants to change the product, help them find a new one using `search_catalog`. You MUST inform them that changing the product will require creating a new design.

**ORDERING GUIDELINES:**
- If the user is happy with the design, guide them to place an order using the `place_order` tool.

{product_context}

{design_context}

{orders_context}
"""
}
}

def get_system_prompt(state: str, product_context=None, design_url: Optional[str] = None, 
                     orders_history: List[Dict] = None) -> str:
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
    design_context_str = get_design_context(design_url)
    
    # Format the prompt with context
    formatted_prompt = base_prompt.format(
        orders_context=orders_context,
        product_context=product_context_str,
        design_context=design_context_str
    )
    
    return formatted_prompt
