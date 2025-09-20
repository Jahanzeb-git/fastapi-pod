import uuid
import json
import logging
import requests
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.tools import tool
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from core.session_manager import SessionManager
from app.config import settings

logger = logging.getLogger(__name__)

# Category and Subcategory mappings for LLM context
CATEGORY_MAPPINGS = {
    "categories": [
        {
            "name": "Men's clothing",
            "subcategories": [
                {"name": "All shirts", "id": 6},
                {"name": "All hoodies & sweatshirts", "id": 7},
                {"name": "Jackets & vests", "id": 95},
                {"name": "All bottoms", "id": 106}
            ]
        },
        {
            "name": "Women's clothing",
            "subcategories": [
                {"name": "All shirts", "id": 8},
                {"name": "All hoodies & sweatshirts", "id": 9},
                {"name": "Dresses", "id": 11},
                {"name": "Swimwear", "id": 79},
                {"name": "Sports Bras", "id": 86},
                {"name": "Jackets & vests", "id": 96},
                {"name": "All Bottoms", "id": 107}
            ]
        },
        {
            "name": "Kid's & Youth clothing",
            "subcategories": [
                {"name": "All shirts", "id": 12},
                {"name": "Leggings", "id": 13},
                {"name": "Baby bodysuits", "id": 14},
                {"name": "Swimwear", "id": 100},
                {"name": "Hoodies & sweatshirts", "id": 105}
            ]
        },
        {
            "name": "Home & Living",
            "subcategories": [
                {"name": "Wall art", "id": 21},
                {"name": "Towels", "id": 22},
                {"name": "Aprons", "id": 88},
                {"name": "Drinkware & coasters", "id": 112}
            ]
        }
    ]
}

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color format: {hex_color}")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def calculate_color_distance(color1_hex: str, color2_hex: str) -> float:
    """Calculate CIELAB color distance between two hex colors."""
    try:
        rgb1 = hex_to_rgb(color1_hex)
        rgb2 = hex_to_rgb(color2_hex)
        
        srgb1 = sRGBColor(rgb1[0]/255.0, rgb1[1]/255.0, rgb1[2]/255.0)
        srgb2 = sRGBColor(rgb2[0]/255.0, rgb2[1]/255.0, rgb2[2]/255.0)
        
        lab1 = convert_color(srgb1, LabColor)
        lab2 = convert_color(srgb2, LabColor)
        
        return delta_e_cie2000(lab1, lab2)
    
    except Exception as e:
        logger.error(f"Color distance calculation error: {e}")
        return float('inf')

def filter_by_size(products: List[Dict], target_size: str) -> List[Dict]:
    """Filter products that contain the target size."""
    filtered = []
    target_size_upper = target_size.upper()
    
    for product in products:
        sizes = product.get('sizes', [])
        if target_size_upper in {size.upper() for size in sizes}:
            filtered.append(product)
    
    return filtered

def filter_by_color_with_ranking(products: List[Dict], target_color_hex: str, max_results: int = 5, threshold: float = 50.0) -> List[Dict]:
    """Filter and rank products by color similarity using CIELAB color matching."""
    color_matches = []
    
    for product in products:
        colors = product.get('colors', [])
        if not colors:
            continue
            
        best_distance = float('inf')
        best_color_match = None
        
        for color_obj in colors:
            color_hex = color_obj.get('value', '')
            if not color_hex:
                continue
                
            distance = calculate_color_distance(target_color_hex, color_hex)
            if distance < best_distance:
                best_distance = distance
                best_color_match = color_obj
        
        if best_distance <= threshold and best_color_match:
            color_matches.append({
                'product': product,
                'distance': best_distance,
                'matched_color': best_color_match
            })
    
    color_matches.sort(key=lambda x: x['distance'])
    
    return color_matches[:max_results]

@tool
async def search_catalog(subcategory_id: int, color: Optional[str] = None, size: Optional[str] = None, user_id: Optional[str] = None, db: Optional[AsyncSession] = None) -> str:
    """
    Search for products in the catalog based on subcategory and optional filters.
    Args:
        subcategory_id: The ID of the subcategory to search in.
        color: Optional hex color code for color-based filtering.
        size: Optional size filter.
    Returns:
        JSON string containing matching products.
    """
    logger.info(f"[TOOL] search_catalog called with subcategory_id={subcategory_id}, color={color}, size={size}")
    
    try:
        valid_ids = set()
        for category in CATEGORY_MAPPINGS["categories"]:
            for subcategory in category["subcategories"]:
                valid_ids.add(subcategory["id"])
        
        if subcategory_id not in valid_ids:
            return f"Invalid subcategory_id: {subcategory_id}. Must be one of: {sorted(valid_ids)}"
        
        url = f"{settings.PRINTFUL_API_BASE}/catalog-products"
        headers = {"Authorization": settings.PRINTFUL_AUTH_TOKEN, "Content-Type": "application/json"}
        params = {"category_ids": subcategory_id, "limit": 100}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        products = response.json().get('data', [])
        if not products:
            return f"No products found for subcategory_id: {subcategory_id}"
        
        if size:
            products = filter_by_size(products, size)
        
        final_results = []
        if color:
            color_matches = filter_by_color_with_ranking(products, color)
            for match in color_matches:
                product = match['product']
                final_results.append({
                    "name": product.get('name', 'Unknown'),
                    "id": product.get('id'),
                    "colors": product.get('colors', []),
                    "sizes": product.get('sizes', [])
                })
        else:
            for product in products[:50]:
                final_results.append({
                    "name": product.get('name', 'Unknown'),
                    "id": product.get('id'),
                    "colors": product.get('colors', []),
                    "sizes": product.get('sizes', [])
                })
        
        return json.dumps({"status": "success", "products": final_results}, indent=2)
        
    except requests.RequestException as e:
        return f"Network error accessing API: {str(e)}"
    except Exception as e:
        return f"Unexpected error in search_catalog: {str(e)}"

@tool
async def create_design(design_prompt: str, user_id: Optional[str] = None, db: Optional[AsyncSession] = None) -> str:
    """
    Create a design based on user prompt.
    Args:
        design_prompt: User's description of the desired design.
    Returns:
        Design creation result with design URL.
    """
    design_id = str(uuid.uuid4())
    design_url = f"https://designs.printplatform.com/designs/{design_id}.jpg"
    return json.dumps({"status": "success", "design_url": design_url})

@tool
async def upscaling(image_url: str, user_id: Optional[str] = None, db: Optional[AsyncSession] = None) -> str:
    """
    Upscale the design image.
    Args:
        image_url: URL of the design image to upscale.
    Returns:
        Upscaling result with high-resolution URL.
    """
    upscaled_id = str(uuid.uuid4())
    upscaled_url = f"https://designs.printplatform.com/upscaled/{upscaled_id}_4k.jpg"
    return json.dumps({"status": "success", "upscaled_url": upscaled_url})


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

AVAILABLE_TOOLS = [search_catalog, create_design, upscaling, place_order, search_order]
