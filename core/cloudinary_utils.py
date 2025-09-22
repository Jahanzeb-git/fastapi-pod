import asyncio
import cloudinary.uploader
from typing import Dict, Any

async def upload_image_from_buffer(image_buffer: bytes, folder: str = "printplatform") -> Dict[str, Any]:
    """
    Asynchronously uploads an image from an in-memory buffer to Cloudinary.

    Args:
        image_buffer: The raw bytes of the image to upload.
        folder: The folder in Cloudinary to upload the image to.

    Returns:
        A dictionary containing the secure URL and public ID of the uploaded image.
    """
    
    def upload_sync():
        return cloudinary.uploader.upload(
            image_buffer,
            folder=folder,
            resource_type="image"
        )

    # Run the synchronous, blocking upload call in a separate thread
    # to avoid blocking the main asyncio event loop.
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, upload_sync)

    return {
        "url": result["secure_url"],
        "public_id": result["public_id"]
    }
