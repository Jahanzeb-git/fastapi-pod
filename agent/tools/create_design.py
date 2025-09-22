import asyncio
import httpx
import logging
import json
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.config import settings
from core.session_manager import SessionManager
from core.websocket_manager import websocket_manager
from core.cloudinary_utils import upload_image_from_buffer

logger = logging.getLogger(__name__)

BFL_API_ENDPOINT = "https://api.bfl.ai/v1/flux-kontext-pro"
POLLING_INTERVAL_SECONDS = 2
POLLING_TIMEOUT_SECONDS = 40

@tool
async def create_design(
    prompt: str, 
    aspect_ratio: str, 
    user_id: str, # Injected by CustomToolNode
    db: AsyncSession, # Injected by CustomToolNode
    seed: Optional[int] = None, 
    image_url: Optional[str] = None, 
    edit: bool = False
) -> str:
    """Generates or edits an image based on a prompt. For generation, provide a prompt and aspect_ratio (e.g., '1:1', '16:9'). For editing, also provide the image_url of the source image and set edit=True."""
    
    logger.info(f"[TOOL] create_design called for user {user_id}. Prompt: '{prompt}'")

    headers = {
        "accept": "application/json",
        "x-key": settings.BFL_API_KEY,
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "output_format": "jpeg",
    }

    if edit and image_url:
        payload["input_image"] = image_url
        logger.info(f"Editing image: {image_url}")
    if seed:
        payload["seed"] = seed

    try:
        # 1. Start the generation job
        async with httpx.AsyncClient() as client:
            response = await client.post(BFL_API_ENDPOINT, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            request_data = response.json()
            polling_url = request_data.get("polling_url")
            if not polling_url:
                raise Exception("BFL API did not return a polling_url")

            logger.info(f"BFL job submitted. Polling URL: {polling_url}")

            # 2. Asynchronous polling loop
            bfl_result = None
            for _ in range(POLLING_TIMEOUT_SECONDS // POLLING_INTERVAL_SECONDS):
                await asyncio.sleep(POLLING_INTERVAL_SECONDS)
                poll_response = await client.get(polling_url, headers=headers, timeout=10)
                poll_response.raise_for_status()
                result_data = poll_response.json()

                if result_data['status'] == 'Ready':
                    bfl_result = result_data['result']['sample']
                    logger.info(f"BFL image ready at: {bfl_result}")
                    break
                elif result_data['status'] in ['Error', 'Failed']:
                    raise Exception(f"BFL job failed: {result_data}")
            
            if not bfl_result:
                raise Exception("BFL job timed out after {POLLING_TIMEOUT_SECONDS} seconds.")

            # 3. Download the image from BFL
            image_response = await client.get(bfl_result, timeout=30)
            image_response.raise_for_status()
            image_buffer = image_response.content
            logger.info("Successfully downloaded image from BFL.")

        # 4. Upload to Cloudinary
        cloudinary_result = await upload_image_from_buffer(image_buffer)
        cloudinary_url = cloudinary_result["url"]
        logger.info(f"Uploaded to Cloudinary: {cloudinary_url}")

        # 5. Update state and notify frontend
        await SessionManager.update_design_urls(db, UUID(user_id), design_url=cloudinary_url)
        await websocket_manager.send_signal(user_id, "image_ready", {"design_url": cloudinary_url})
        logger.info(f"Session updated and frontend notified for user {user_id}.")

        # 6. Return the final URL to the agent
        return json.dumps({"status": "success", "design_url": cloudinary_url})

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during create_design: {e.response.text}")
        return json.dumps({"status": "error", "message": f"An API error occurred: {e.response.text}"})
    except Exception as e:
        logger.error(f"An unexpected error occurred in create_design: {e}")
        return json.dumps({"status": "error", "message": f"An unexpected error occurred: {e}"})
