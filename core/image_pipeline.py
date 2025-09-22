"""printful_image_pipeline_commercial.py

Production-ready, commercial-grade utility for AI-based Print-on-Demand platform.
Handles professional print preparation with bleed, safe zones, color profiles, and quality validation.

Author: Jahanzeb Ahmed
License: MIT-style permissive header for internal use
"""
from __future__ import annotations

import logging
import math
import os
import io
import json
import time
import requests
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Tuple, Optional, List, Union

from PIL import Image, UnidentifiedImageError, ImageCms
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# Logging configuration (library-friendly)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Enhanced Exceptions
# ---------------------------------------------------------------------------

class PipelineError(Exception):
    """Base exception for pipeline errors."""


class InvalidDimensionsError(PipelineError, ValueError):
    """Raised when width/height/dpi values are invalid."""


class ImageReadError(PipelineError, IOError):
    """Raised when the Flux image cannot be read or is invalid."""


class AspectRatioError(PipelineError):
    """Raised when aspect ratio is outside acceptable range."""


class ESRGANError(PipelineError):
    """Raised when Real-ESRGAN upscaling fails."""


class ColorProfileError(PipelineError):
    """Raised when color profile conversion fails."""


class QualityValidationError(PipelineError):
    """Raised when image quality doesn't meet print standards."""


# ---------------------------------------------------------------------------
# Enhanced Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FluxRequirements:
    """Phase 1 Result: What to request from Flux (for generate_image tool)"""
    aspect_ratio_str: str
    reduced_ratio: Tuple[int, int]
    target_canvas_px: Tuple[int, int]
    target_canvas_with_bleed_px: Tuple[int, int]
    predicted_flux_px: Tuple[int, int]
    flux_compatible: bool
    bleed_size_px: Tuple[int, int]
    safe_zone_px: Tuple[int, int]
    notes: Tuple[str, ...]


@dataclass(frozen=True)
class ProcessedImageResult:
    """Phase 2 Result: Final processed image ready for Printful (for place_order tool)"""
    processed_image_path: str
    original_flux_size: Tuple[int, int]
    final_size: Tuple[int, int]
    esrgan_scale_used: int
    aspect_ratio_valid: bool
    dpi_validated: bool
    color_profile_converted: bool
    processing_notes: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Enhanced Helper functions
# ---------------------------------------------------------------------------

def _validate_positive_number(name: str, value: float) -> None:
    """Validate that a value is a positive number."""
    try:
        v = float(value)
    except Exception:
        raise InvalidDimensionsError(f"{name} must be a number, got {value!r}")
    if not (v > 0):
        raise InvalidDimensionsError(f"{name} must be > 0, got {value}")


def _reduce_fraction(width_in: float, height_in: float, max_denominator: int = 1000) -> Tuple[int, int, str]:
    """Return reduced integer ratio (w:h) and string representation."""
    if height_in == 0:
        raise InvalidDimensionsError("height_in cannot be zero")

    frac = Fraction(width_in / height_in).limit_denominator(max_denominator)
    return frac.numerator, frac.denominator, f"{frac.numerator}:{frac.denominator}"


def _compute_target_canvas_px(width_in: float, height_in: float, dpi: float) -> Tuple[int, int]:
    """Compute target canvas size in pixels (integers)."""
    _validate_positive_number("width_in", width_in)
    _validate_positive_number("height_in", height_in)
    _validate_positive_number("dpi", dpi)

    target_w = max(1, int(round(width_in * dpi)))
    target_h = max(1, int(round(height_in * dpi)))
    return target_w, target_h


def _compute_bleed_and_safe_zone(width_in: float, height_in: float, dpi: float) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Compute canvas with bleed, bleed size, and safe zone dimensions.
    
    Returns:
        - canvas_with_bleed_px: (width, height) including 0.125" bleed
        - bleed_size_px: (width_bleed, height_bleed) in pixels
        - safe_zone_px: (width_safe, height_safe) safe area margins
    """
    BLEED_INCHES = 0.125  # Industry standard
    SAFE_ZONE_INCHES = 0.25  # Safe area margin
    
    # Calculate bleed in pixels
    bleed_px_w = int(round(BLEED_INCHES * dpi))
    bleed_px_h = int(round(BLEED_INCHES * dpi))
    
    # Calculate safe zone in pixels  
    safe_px_w = int(round(SAFE_ZONE_INCHES * dpi))
    safe_px_h = int(round(SAFE_ZONE_INCHES * dpi))
    
    # Original canvas
    canvas_w, canvas_h = _compute_target_canvas_px(width_in, height_in, dpi)
    
    # Canvas with bleed (extended)
    canvas_bleed_w = canvas_w + (2 * bleed_px_w)  # Bleed on both sides
    canvas_bleed_h = canvas_h + (2 * bleed_px_h)  # Bleed on top/bottom
    
    return (canvas_bleed_w, canvas_bleed_h), (bleed_px_w, bleed_px_h), (safe_px_w, safe_px_h)


def _predict_flux_size(ratio_w: int, ratio_h: int, target_area: int = 1_000_000) -> Tuple[int, int]:
    """Predict Flux output size given aspect ratio and target area."""
    predicted_h = int(round(math.sqrt(target_area * ratio_h / ratio_w)))
    predicted_w = max(1, int(round(predicted_h * ratio_w / ratio_h)))
    return predicted_w, predicted_h


def _is_flux_compatible(ratio_w: int, ratio_h: int) -> bool:
    """Check if aspect ratio is within Flux.1 Kontext Pro supported range (3:7 to 7:3)."""
    ratio = ratio_w / ratio_h
    min_ratio = 3 / 7  # ≈ 0.429
    max_ratio = 7 / 3  # ≈ 2.333
    return min_ratio <= ratio <= max_ratio


def _read_image_size(image_path: Union[str, Path]) -> Tuple[int, int]:
    """Read actual pixel dimensions of an image using Pillow."""
    p = Path(image_path)
    if not p.exists():
        raise ImageReadError(f"Image file not found: {image_path}")

    try:
        with Image.open(p) as im:
            w, h = im.size
            return int(w), int(h)
    except UnidentifiedImageError as e:
        raise ImageReadError(f"Unable to identify image file: {image_path}") from e
    except Exception as e:
        raise ImageReadError(f"Error reading image {image_path}: {e}") from e


def _validate_aspect_ratio(target_w: int, target_h: int, flux_w: int, flux_h: int, 
                          tolerance: float = 0.001) -> bool:
    """Validate aspect ratio within acceptable tolerance (0.1% default)."""
    if flux_h == 0 or target_h == 0:
        return False
    
    target_ratio = target_w / target_h
    flux_ratio = flux_w / flux_h
    
    # Calculate relative difference
    relative_diff = abs(target_ratio - flux_ratio) / target_ratio
    return relative_diff <= tolerance


def _validate_print_dpi(width_px: int, height_px: int, width_in: float, height_in: float, 
                       min_dpi: float = 150.0) -> Tuple[bool, float, float]:
    """Validate that image resolution meets minimum print DPI requirements.
    
    Returns:
        - meets_requirement: True if both width and height DPI >= min_dpi
        - actual_dpi_w: Actual width DPI
        - actual_dpi_h: Actual height DPI
    """
    actual_dpi_w = width_px / width_in if width_in > 0 else 0
    actual_dpi_h = height_px / height_in if height_in > 0 else 0
    
    meets_requirement = (actual_dpi_w >= min_dpi) and (actual_dpi_h >= min_dpi)
    return meets_requirement, actual_dpi_w, actual_dpi_h


def _upscale_with_esrgan(image_path: Union[str, Path], scale_factor: int, 
                        output_path: Optional[Union[str, Path]] = None) -> str:
    """Upscale image using Real-ESRGAN via WaveSpeed API.
    
    Args:
        image_path: Path to input image
        scale_factor: Scale factor (1-10, must be integer)
        output_path: Optional output path, auto-generated if None
        
    Returns:
        Path to upscaled image
        
    Raises:
        ESRGANError: If upscaling fails
    """
    API_KEY = os.getenv("WAVESPEED_API_KEY")
    if not API_KEY:
        raise ESRGANError("WAVESPEED_API_KEY environment variable not set")
    
    if not (1 <= scale_factor <= 10):
        raise ESRGANError(f"ESRGAN scale factor must be 1-10, got {scale_factor}")
    
    # Convert local path to URL (this would need your file hosting solution)
    # For now, assuming you have a way to upload to a temporary URL
    image_url = _upload_temp_image(image_path)  # You'll need to implement this
    
    # Submit ESRGAN request
    url = "https://api.wavespeed.ai/api/v3/wavespeed-ai/real-esrgan"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "face_enhance": False,
        "image": image_url,
        "scale": scale_factor
    }
    
    logger.info(f"Submitting ESRGAN upscaling request: {scale_factor}x scale")
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        if response.status_code != 200:
            raise ESRGANError(f"ESRGAN API request failed: {response.status_code} - {response.text}")
        
        result = response.json()["data"]
        request_id = result["id"]
        logger.info(f"ESRGAN task submitted. Request ID: {request_id}")
        
    except requests.exceptions.RequestException as e:
        raise ESRGANError(f"ESRGAN API request failed: {str(e)}") from e
    except (KeyError, json.JSONDecodeError) as e:
        raise ESRGANError(f"Invalid ESRGAN API response format: {str(e)}") from e
    
    # Poll for results
    result_url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    max_wait_time = 300  # 5 minutes timeout
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(result_url, headers=headers, timeout=10)
            if response.status_code != 200:
                raise ESRGANError(f"ESRGAN result polling failed: {response.status_code}")
            
            result = response.json()["data"]
            status = result["status"]
            
            if status == "completed":
                output_image_url = result["outputs"][0]
                logger.info(f"ESRGAN upscaling completed: {output_image_url}")
                
                # Download result
                if output_path is None:
                    output_path = Path(image_path).parent / f"{Path(image_path).stem}_esrgan_{scale_factor}x.png"
                
                return _download_image(output_image_url, output_path)
                
            elif status == "failed":
                error_msg = result.get('error', 'Unknown error')
                raise ESRGANError(f"ESRGAN upscaling failed: {error_msg}")
            
            # Still processing
            time.sleep(2)
            
        except requests.exceptions.RequestException as e:
            raise ESRGANError(f"ESRGAN polling failed: {str(e)}") from e
        except (KeyError, json.JSONDecodeError) as e:
            raise ESRGANError(f"Invalid ESRGAN result format: {str(e)}") from e
    
    raise ESRGANError(f"ESRGAN upscaling timed out after {max_wait_time} seconds")


def _upload_temp_image(image_path: Union[str, Path]) -> str:
    """Upload image to Cloudinary and return public URL"""
    try:
        # Configure Cloudinary (set these env vars)
        cloudinary.config(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"), 
            api_secret=os.getenv("CLOUDINARY_API_SECRET")
        )
        
        # Upload with auto-deletion after 1 hour
        result = cloudinary.uploader.upload(
            str(image_path),
            resource_type="image",
            public_id=f"temp_{int(time.time())}",
            expiration=3600  # Delete after 1 hour
        )
        
        return result['secure_url']
        
    except Exception as e:
        raise ESRGANError(f"Failed to upload image to Cloudinary: {str(e)}") from e


def _download_image(url: str, output_path: Union[str, Path]) -> str:
    """Download image from URL to local path."""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return str(output_path)
        
    except requests.exceptions.RequestException as e:
        raise ESRGANError(f"Failed to download ESRGAN result: {str(e)}") from e


def _resize_image_high_quality(image_path: Union[str, Path], target_size: Tuple[int, int], 
                              output_path: Union[str, Path]) -> str:
    """Resize image using high-quality Lanczos resampling."""
    try:
        with Image.open(image_path) as img:
            # Use Lanczos for high-quality downscaling
            resized = img.resize(target_size, Image.Resampling.LANCZOS)
            resized.save(output_path, "PNG", optimize=True)
            logger.info(f"High-quality resize: {img.size} → {target_size}")
            return str(output_path)
            
    except Exception as e:
        raise PipelineError(f"High-quality resize failed: {str(e)}") from e


def _convert_to_srgb(image_path: Union[str, Path], output_path: Union[str, Path]) -> str:
    """Convert image to sRGB IEC61966-2.1 color profile."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create or get sRGB profile
            srgb_profile = ImageCms.createProfile('sRGB')
            
            # Get current profile or assume sRGB if none
            try:
                current_profile = img.info.get('icc_profile')
                if current_profile:
                    current_profile = ImageCms.ImageCmsProfile(io.BytesIO(current_profile))
                else:
                    current_profile = srgb_profile
            except:
                current_profile = srgb_profile
            
            # Convert to sRGB
            if current_profile.profile.profile_description != srgb_profile.profile.profile_description:
                transform = ImageCms.buildTransformFromOpenProfiles(
                    current_profile, srgb_profile, 'RGB', 'RGB'
                )
                converted_img = ImageCms.applyTransform(img, transform)
            else:
                converted_img = img.copy()
            
            # Embed sRGB profile
            icc_profile = srgb_profile.tobytes()
            converted_img.save(output_path, "PNG", icc_profile=icc_profile, optimize=True)
            
            logger.info("Image converted to sRGB IEC61966-2.1 color profile")
            return str(output_path)
            
    except Exception as e:
        raise ColorProfileError(f"sRGB color profile conversion failed: {str(e)}") from e


# ---------------------------------------------------------------------------
# Phase 1: Pre-Generation Planning (Enhanced)
# ---------------------------------------------------------------------------

def compute_flux_requirements(
    width_in: float,
    height_in: float,
    dpi: float,
    flux_area_target: int = 1_000_000,
) -> FluxRequirements:
    """Phase 1: Compute what to request from Flux.1 Kontext Pro (for generate_image tool).
    
    Enhanced with professional bleed and safe zone calculations.
    
    Args:
        width_in: Printful placement width in inches
        height_in: Printful placement height in inches
        dpi: Product DPI from Printful mockup-styles
        flux_area_target: Expected Flux output area in pixels (default ~1MP)
        
    Returns:
        FluxRequirements with aspect ratio, target dimensions, and bleed info
        
    Raises:
        InvalidDimensionsError: Invalid input dimensions
        AspectRatioError: Aspect ratio outside Flux.1 supported range
    """
    # Validate inputs
    _validate_positive_number("width_in", width_in)
    _validate_positive_number("height_in", height_in)
    _validate_positive_number("dpi", dpi)
    
    notes: List[str] = []
    
    # Step 1: Aspect ratio reduction
    ratio_w, ratio_h, ratio_str = _reduce_fraction(width_in, height_in)
    logger.debug("Reduced aspect ratio: %s (%d:%d)", ratio_str, ratio_w, ratio_h)
    notes.append(f"Reduced aspect ratio: {ratio_str}")
    
    # Step 2: Target canvas and bleed calculations
    target_w, target_h = _compute_target_canvas_px(width_in, height_in, dpi)
    canvas_bleed, bleed_size, safe_zone = _compute_bleed_and_safe_zone(width_in, height_in, dpi)
    
    logger.debug("Target canvas: %dx%d px", target_w, target_h)
    logger.debug("Canvas with bleed: %dx%d px", canvas_bleed[0], canvas_bleed[1])
    
    notes.append(f"Target canvas: {target_w}×{target_h} px")
    notes.append(f"Canvas with bleed: {canvas_bleed[0]}×{canvas_bleed[1]} px")
    notes.append(f"Bleed margins: {bleed_size[0]}×{bleed_size[1]} px (0.125\" industry standard)")
    notes.append(f"Safe zone margins: {safe_zone[0]}×{safe_zone[1]} px (0.25\" recommended)")
    
    # Step 3: Flux compatibility check
    flux_compatible = _is_flux_compatible(ratio_w, ratio_h)
    if not flux_compatible:
        ratio_decimal = ratio_w / ratio_h
        raise AspectRatioError(
            f"Aspect ratio {ratio_str} ({ratio_decimal:.3f}) is outside Flux.1 Kontext Pro "
            f"supported range (3:7 to 7:3). Consider adjusting print dimensions."
        )
    
    notes.append(f"✓ Aspect ratio {ratio_str} is compatible with Flux.1 Kontext Pro")
    
    # Step 4: Predict Flux output size
    predicted_w, predicted_h = _predict_flux_size(ratio_w, ratio_h, flux_area_target)
    notes.append(f"Predicted Flux output: ~{predicted_w}×{predicted_h} px")
    
    result = FluxRequirements(
        aspect_ratio_str=ratio_str,
        reduced_ratio=(ratio_w, ratio_h),
        target_canvas_px=(target_w, target_h),
        target_canvas_with_bleed_px=canvas_bleed,
        predicted_flux_px=(predicted_w, predicted_h),
        flux_compatible=flux_compatible,
        bleed_size_px=bleed_size,
        safe_zone_px=safe_zone,
        notes=tuple(notes),
    )
    
    logger.info("Phase 1 complete: %s", result)
    return result


# ---------------------------------------------------------------------------
# Phase 2: Post-Generation Processing (Commercial Grade)
# ---------------------------------------------------------------------------

def process_final_image(
    flux_image_path: Union[str, Path],
    target_canvas_px: Tuple[int, int],
    width_in: float,
    height_in: float,
    output_dir: Union[str, Path],
    aspect_tolerance: float = 0.001,
    min_print_dpi: float = 150.0,
) -> ProcessedImageResult:
    """Phase 2: Process Flux image for commercial print quality (for place_order tool).
    
    Complete post-processing pipeline:
    1. Aspect ratio validation
    2. DPI quality validation  
    3. Intelligent ESRGAN upscaling
    4. High-quality downscaling if needed
    5. sRGB color profile conversion
    6. Final validation
    
    Args:
        flux_image_path: Path to generated Flux image
        target_canvas_px: Target dimensions from Phase 1 (width, height)
        width_in: Product width in inches
        height_in: Product height in inches
        output_dir: Directory for processed output
        aspect_tolerance: Relative tolerance for aspect ratio (0.001 = 0.1%)
        min_print_dpi: Minimum acceptable DPI for print quality
        
    Returns:
        ProcessedImageResult with final processed image path
        
    Raises:
        AspectRatioError: Aspect ratio validation failed
        QualityValidationError: Image quality insufficient for print
        ESRGANError: Upscaling failed
        ColorProfileError: Color profile conversion failed
    """
    target_w, target_h = target_canvas_px
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    notes: List[str] = []
    
    # Step 1: Read actual Flux dimensions
    flux_w, flux_h = _read_image_size(flux_image_path)
    logger.info(f"Processing Flux image: {flux_w}×{flux_h} px")
    notes.append(f"Flux output dimensions: {flux_w}×{flux_h} px")
    
    # Step 2: Aspect ratio validation (CRITICAL)
    aspect_valid = _validate_aspect_ratio(target_w, target_h, flux_w, flux_h, aspect_tolerance)
    if not aspect_valid:
        target_ratio = target_w / target_h
        flux_ratio = flux_w / flux_h
        relative_diff = abs(target_ratio - flux_ratio) / target_ratio * 100
        
        raise AspectRatioError(
            f"Aspect ratio validation FAILED. Target: {target_ratio:.6f}, "
            f"Flux: {flux_ratio:.6f}, Difference: {relative_diff:.3f}% "
            f"(tolerance: {aspect_tolerance*100:.3f}%). Image rejected for print quality."
        )
    
    notes.append("✓ Aspect ratio validation passed")
    
    # Step 3: Initial DPI validation
    dpi_valid, actual_dpi_w, actual_dpi_h = _validate_print_dpi(
        flux_w, flux_h, width_in, height_in, min_print_dpi
    )
    notes.append(f"Flux DPI: {actual_dpi_w:.1f} × {actual_dpi_h:.1f}")
    
    # Step 4: Calculate required scaling
    scale_w = target_w / flux_w
    scale_h = target_h / flux_h
    max_scale_required = max(scale_w, scale_h)
    
    notes.append(f"Required scaling: {scale_w:.4f}x width, {scale_h:.4f}x height")
    
    # Step 5: Intelligent ESRGAN scaling
    processed_image_path = flux_image_path
    esrgan_scale_used = 1
    
    if max_scale_required > 1.0:
        # Calculate optimal ESRGAN scale (ceiling of required scale)
        optimal_esrgan_scale = min(10, int(math.ceil(max_scale_required)))
        
        logger.info(f"Upscaling required: {max_scale_required:.3f}x → using {optimal_esrgan_scale}x ESRGAN")
        notes.append(f"ESRGAN upscaling: {optimal_esrgan_scale}x (quality-first approach)")
        
        try:
            esrgan_output_path = output_dir / f"{Path(flux_image_path).stem}_esrgan_{optimal_esrgan_scale}x.png"
            processed_image_path = _upscale_with_esrgan(
                flux_image_path, optimal_esrgan_scale, esrgan_output_path
            )
            esrgan_scale_used = optimal_esrgan_scale
            notes.append(f"✓ ESRGAN upscaling completed: {optimal_esrgan_scale}x")
            
        except ESRGANError as e:
            logger.error(f"ESRGAN upscaling failed: {e}")
            raise ESRGANError(f"Real-ESRGAN upscaling failed: {str(e)}. Cannot proceed with print processing.") from e
    
    # Step 6: Final precise resizing (if needed)
    current_w, current_h = _read_image_size(processed_image_path)
    
    if current_w != target_w or current_h != target_h:
        logger.info(f"Final resize: {current_w}×{current_h} → {target_w}×{target_h}")
        notes.append(f"High-quality resize: {current_w}×{current_h} → {target_w}×{target_h} (Lanczos)")
        
        final_resize_path = output_dir / f"{Path(flux_image_path).stem}_final_resized.png"
        processed_image_path = _resize_image_high_quality(
            processed_image_path, (target_w, target_h), final_resize_path
        )
    else:
        notes.append("✓ No final resize needed - dimensions match exactly")
    
    # Step 7: sRGB color profile conversion
    srgb_output_path = output_dir / f"{Path(flux_image_path).stem}_final_srgb.png"
    
    try:
        processed_image_path = _convert_to_srgb(processed_image_path, srgb_output_path)
        notes.append("✓ sRGB IEC61966-2.1 color profile applied")
        color_profile_converted = True
        
    except ColorProfileError as e:
        logger.error(f"Color profile conversion failed: {e}")
        raise ColorProfileError(f"sRGB color profile conversion failed: {str(e)}. Cannot proceed with print processing.") from e
    
    # Step 8: Final quality validation
    final_w, final_h = _read_image_size(processed_image_path)
    final_dpi_valid, final_dpi_w, final_dpi_h = _validate_print_dpi(
        final_w, final_h, width_in, height_in, min_print_dpi
    )
    
    if not final_dpi_valid:
        raise QualityValidationError(
            f"Final image quality insufficient for print. DPI: {final_dpi_w:.1f} × {final_dpi_h:.1f}, "
            f"Required minimum: {min_print_dpi} DPI. Consider larger source image or different scaling approach."
        )
    
    notes.append(f"✓ Final quality validation passed: {final_dpi_w:.1f} × {final_dpi_h:.1f} DPI")
    
    # Final result
    result = ProcessedImageResult(
        processed_image_path=str(processed_image_path),
        original_flux_size=(flux_w, flux_h),
        final_size=(final_w, final_h),
        esrgan_scale_used=esrgan_scale_used,
        aspect_ratio_valid=aspect_valid,
        dpi_validated=final_dpi_valid,
        color_profile_converted=color_profile_converted,
        processing_notes=tuple(notes),
    )
    
    logger.info("Phase 2 processing complete: %s", result)
    return result


# ---------------------------------------------------------------------------
# Utility Functions for Integration
# ---------------------------------------------------------------------------

def validate_flux_image_for_processing(
    flux_image_path: Union[str, Path],
    expected_aspect_ratio: str,
    tolerance: float = 0.001
) -> bool:
    """Quick validation function to check if Flux image is suitable for processing.
    
    Args:
        flux_image_path: Path to Flux generated image
        expected_aspect_ratio: Expected aspect ratio string (e.g., "3:4")
        tolerance: Aspect ratio tolerance
        
    Returns:
        True if image is suitable for processing
        
    Raises:
        ImageReadError: Cannot read image file
        ValueError: Invalid aspect ratio format
    """
    try:
        # Parse expected aspect ratio
        if ":" not in expected_aspect_ratio:
            raise ValueError(f"Invalid aspect ratio format: {expected_aspect_ratio}. Expected format: 'width:height'")
        
        ratio_parts = expected_aspect_ratio.split(":")
        if len(ratio_parts) != 2:
            raise ValueError(f"Invalid aspect ratio format: {expected_aspect_ratio}")
        
        expected_w, expected_h = int(ratio_parts[0]), int(ratio_parts[1])
        expected_ratio = expected_w / expected_h
        
        # Read actual image dimensions
        actual_w, actual_h = _read_image_size(flux_image_path)
        actual_ratio = actual_w / actual_h
        
        # Check tolerance
        relative_diff = abs(expected_ratio - actual_ratio) / expected_ratio
        return relative_diff <= tolerance
        
    except Exception as e:
        logger.error(f"Flux image validation failed: {e}")
        return False


def get_recommended_flux_dimensions(aspect_ratio_str: str, target_area: int = 1_000_000) -> Tuple[int, int]:
    """Get recommended dimensions for Flux generation based on aspect ratio.
    
    Args:
        aspect_ratio_str: Aspect ratio string (e.g., "3:4")
        target_area: Target pixel area for generation
        
    Returns:
        Tuple of (width, height) recommended for Flux
        
    Raises:
        ValueError: Invalid aspect ratio format
    """
    try:
        ratio_parts = aspect_ratio_str.split(":")
        if len(ratio_parts) != 2:
            raise ValueError(f"Invalid aspect ratio format: {aspect_ratio_str}")
        
        ratio_w, ratio_h = int(ratio_parts[0]), int(ratio_parts[1])
        return _predict_flux_size(ratio_w, ratio_h, target_area)
        
    except Exception as e:
        raise ValueError(f"Cannot compute Flux dimensions: {e}") from e


def estimate_processing_time(
    flux_width: int, 
    flux_height: int, 
    target_width: int, 
    target_height: int
) -> dict:
    """Estimate processing time for different pipeline stages.
    
    Args:
        flux_width: Source image width
        flux_height: Source image height  
        target_width: Target image width
        target_height: Target image height
        
    Returns:
        Dictionary with estimated times in seconds
    """
    # Calculate scale factor
    scale_w = target_width / flux_width
    scale_h = target_height / flux_height
    max_scale = max(scale_w, scale_h)
    
    estimates = {
        "validation": 1,  # Quick validation steps
        "esrgan_upscaling": 0,
        "resize_downscale": 0,
        "color_conversion": 2,
        "total_estimated": 3
    }
    
    # ESRGAN time estimation (based on scale factor and image size)
    if max_scale > 1.0:
        esrgan_scale = min(10, int(math.ceil(max_scale)))
        # Rough estimate: ~10-30 seconds for ESRGAN depending on scale and size
        base_time = 15  # Base processing time
        size_factor = (flux_width * flux_height) / 1_000_000  # Size complexity
        scale_factor = esrgan_scale / 4  # Scale complexity
        
        estimates["esrgan_upscaling"] = int(base_time * size_factor * scale_factor)
    
    # Resize time estimation
    if max_scale > 1.0:
        upscaled_pixels = flux_width * flux_height * (int(math.ceil(max_scale)) ** 2)
        resize_complexity = upscaled_pixels / 10_000_000  # Rough complexity measure
        estimates["resize_downscale"] = max(1, int(resize_complexity * 3))
    
    # Update total
    estimates["total_estimated"] = sum(estimates.values()) - estimates["total_estimated"]
    
    return estimates


# ---------------------------------------------------------------------------
# Error Recovery and Debugging Utilities
# ---------------------------------------------------------------------------

def diagnose_image_issues(image_path: Union[str, Path]) -> dict:
    """Diagnose common issues with generated images.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with diagnostic information
    """
    diagnostics = {
        "file_exists": False,
        "file_readable": False,
        "dimensions": None,
        "file_size_mb": 0,
        "format": None,
        "color_mode": None,
        "has_transparency": False,
        "issues": [],
        "recommendations": []
    }
    
    try:
        p = Path(image_path)
        diagnostics["file_exists"] = p.exists()
        
        if not p.exists():
            diagnostics["issues"].append("Image file does not exist")
            diagnostics["recommendations"].append("Check file path and ensure image was generated successfully")
            return diagnostics
        
        # File size check
        file_size_bytes = p.stat().st_size
        diagnostics["file_size_mb"] = file_size_bytes / (1024 * 1024)
        
        if file_size_bytes == 0:
            diagnostics["issues"].append("Image file is empty (0 bytes)")
            diagnostics["recommendations"].append("Regenerate image - file appears corrupted")
            return diagnostics
        
        # Try to read with PIL
        try:
            with Image.open(p) as img:
                diagnostics["file_readable"] = True
                diagnostics["dimensions"] = img.size
                diagnostics["format"] = img.format
                diagnostics["color_mode"] = img.mode
                diagnostics["has_transparency"] = img.mode in ("RGBA", "LA", "P") and "transparency" in img.info
                
                # Check for common issues
                width, height = img.size
                
                if width < 512 or height < 512:
                    diagnostics["issues"].append(f"Image resolution very low: {width}×{height}")
                    diagnostics["recommendations"].append("Consider regenerating with higher resolution")
                
                if width * height > 50_000_000:  # 50MP
                    diagnostics["issues"].append(f"Image resolution extremely high: {width}×{height}")
                    diagnostics["recommendations"].append("Consider reducing resolution to avoid processing issues")
                
                # Check aspect ratio extremes
                ratio = max(width, height) / min(width, height)
                if ratio > 10:
                    diagnostics["issues"].append(f"Extreme aspect ratio: {ratio:.2f}:1")
                    diagnostics["recommendations"].append("Very wide/tall images may have processing limitations")
                
        except UnidentifiedImageError:
            diagnostics["issues"].append("Image file format not recognized or corrupted")
            diagnostics["recommendations"].append("Regenerate image - file may be corrupted or in unsupported format")
        except Exception as e:
            diagnostics["issues"].append(f"Error reading image: {str(e)}")
            diagnostics["recommendations"].append("Check file integrity and format")
    
    except Exception as e:
        diagnostics["issues"].append(f"Diagnostic error: {str(e)}")
    
    return diagnostics


def create_debug_report(
    flux_requirements: FluxRequirements,
    processed_result: Optional[ProcessedImageResult] = None,
    error: Optional[Exception] = None
) -> str:
    """Create comprehensive debug report for troubleshooting.
    
    Args:
        flux_requirements: Result from Phase 1
        processed_result: Result from Phase 2 (if completed)
        error: Any exception that occurred
        
    Returns:
        Formatted debug report string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PRINTFUL IMAGE PIPELINE DEBUG REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    report_lines.append("")
    
    # Phase 1 Information
    report_lines.append("PHASE 1 - FLUX REQUIREMENTS:")
    report_lines.append("-" * 40)
    report_lines.append(f"Aspect Ratio: {flux_requirements.aspect_ratio_str}")
    report_lines.append(f"Target Canvas: {flux_requirements.target_canvas_px[0]}×{flux_requirements.target_canvas_px[1]} px")
    report_lines.append(f"Canvas with Bleed: {flux_requirements.target_canvas_with_bleed_px[0]}×{flux_requirements.target_canvas_with_bleed_px[1]} px")
    report_lines.append(f"Predicted Flux Size: {flux_requirements.predicted_flux_px[0]}×{flux_requirements.predicted_flux_px[1]} px")
    report_lines.append(f"Flux Compatible: {flux_requirements.flux_compatible}")
    report_lines.append("")
    
    for note in flux_requirements.notes:
        report_lines.append(f"  • {note}")
    report_lines.append("")
    
    # Phase 2 Information (if available)
    if processed_result:
        report_lines.append("PHASE 2 - PROCESSING RESULT:")
        report_lines.append("-" * 40)
        report_lines.append(f"Original Flux Size: {processed_result.original_flux_size[0]}×{processed_result.original_flux_size[1]} px")
        report_lines.append(f"Final Size: {processed_result.final_size[0]}×{processed_result.final_size[1]} px")
        report_lines.append(f"ESRGAN Scale Used: {processed_result.esrgan_scale_used}x")
        report_lines.append(f"Aspect Ratio Valid: {processed_result.aspect_ratio_valid}")
        report_lines.append(f"DPI Validated: {processed_result.dpi_validated}")
        report_lines.append(f"Color Profile Converted: {processed_result.color_profile_converted}")
        report_lines.append(f"Final Image Path: {processed_result.processed_image_path}")
        report_lines.append("")
        
        for note in processed_result.processing_notes:
            report_lines.append(f"  • {note}")
        report_lines.append("")
    
    # Error Information (if any)
    if error:
        report_lines.append("ERROR DETAILS:")
        report_lines.append("-" * 40)
        report_lines.append(f"Error Type: {type(error).__name__}")
        report_lines.append(f"Error Message: {str(error)}")
        report_lines.append("")
        
        # Add specific guidance based on error type
        if isinstance(error, AspectRatioError):
            report_lines.append("TROUBLESHOOTING GUIDANCE:")
            report_lines.append("• Check if Flux generated image with correct aspect ratio")
            report_lines.append("• Verify input dimensions are correct")
            report_lines.append("• Consider regenerating image with exact aspect ratio specification")
        elif isinstance(error, ESRGANError):
            report_lines.append("TROUBLESHOOTING GUIDANCE:")
            report_lines.append("• Check WAVESPEED_API_KEY environment variable")
            report_lines.append("• Verify image upload/hosting is working")
            report_lines.append("• Check WaveSpeed API status and quotas")
            report_lines.append("• Consider fallback upscaling methods")
        elif isinstance(error, QualityValidationError):
            report_lines.append("TROUBLESHOOTING GUIDANCE:")
            report_lines.append("• Source image resolution may be too low")
            report_lines.append("• Consider regenerating at higher resolution")
            report_lines.append("• Check if upscaling parameters are optimal")
    
    report_lines.append("=" * 80)
    return "\n".join(report_lines)


# ---------------------------------------------------------------------------
# Integration Examples and Documentation
# ---------------------------------------------------------------------------

def example_usage():
    """Example usage of the enhanced pipeline functions."""
    
    # Example 1: Phase 1 - Pre-generation planning (for generate_image tool)
    try:
        # Get requirements for a design: 10" × 12" at 300 DPI
        flux_req = compute_flux_requirements(
            width_in=10.0,
            height_in=12.0,
            dpi=300.0
        )
        
        print("Phase 1 Complete - Flux Requirements:")
        print(f"Aspect Ratio: {flux_req.aspect_ratio_str}")
        print(f"Target Canvas: {flux_req.target_canvas_px}")
        print(f"Canvas with Bleed: {flux_req.target_canvas_with_bleed_px}")
        
        # Use flux_req.aspect_ratio_str for Flux generation
        
    except (InvalidDimensionsError, AspectRatioError) as e:
        print(f"Phase 1 failed: {e}")
        return
    
    # Example 2: Phase 2 - Post-generation processing (for place_order tool)
    flux_image_path = "path/to/flux/generated/image.png"  # From Flux API
    output_directory = "processed_images"
    
    try:
        processed = process_final_image(
            flux_image_path=flux_image_path,
            target_canvas_px=flux_req.target_canvas_px,
            width_in=10.0,
            height_in=12.0,
            output_dir=output_directory
        )
        
        print("Phase 2 Complete - Image Processed:")
        print(f"Final Image: {processed.processed_image_path}")
        print(f"Ready for Printful upload: {processed.dpi_validated and processed.color_profile_converted}")
        
    except (AspectRatioError, QualityValidationError, ESRGANError, ColorProfileError) as e:
        print(f"Phase 2 failed: {e}")
        
        # Create debug report for troubleshooting
        debug_report = create_debug_report(flux_req, error=e)
        print("\nDEBUG REPORT:")
        print(debug_report)


if __name__ == "__main__":
    # Run example usage
    example_usage()