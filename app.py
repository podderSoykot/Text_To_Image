"""
AI Artistic QR Code Generator API
==================================
Production-ready REST API for generating AI-artistic images with embedded scannable QR codes.
Uses Stable Diffusion for image generation and advanced image processing for QR embedding.

Features:
- Generate artistic QR codes from text prompts
- QR code validation and scannability testing
- Batch generation support
- Image preview and thumbnails
- Comprehensive API documentation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, status, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import os
import uuid
import asyncio
from pathlib import Path
from datetime import datetime
import time
from PIL import Image
import io
from artistic_qr_pipeline import ArtisticQRPipeline
from qr_validator import QRCodeValidator
import traceback
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Artistic QR Code Generator API",
    description="""
    🎨 **AI Artistic QR Code Generator API**
    
    Generate beautiful, scannable QR codes embedded in AI-generated artistic images.
    
    ## Features
    
    - 🖼️ **AI Image Generation**: Uses Stable Diffusion to create artistic images
    - 📱 **Scannable QR Codes**: Embedded QR codes that remain fully scannable
    - ✅ **Validation**: Automatic QR code scannability testing
    - 🎯 **Customizable**: Control image style, QR subtlety, and generation parameters
    - 📦 **Batch Generation**: Generate multiple QR codes in one request
    - 🖼️ **Preview**: Get image previews and thumbnails
    
    ## Quick Start
    
    1. **Generate a QR Code**:
       ```bash
       POST /api/v1/generate
       {
         "prompt": "A dog in the sky with clouds",
         "qr_data": "https://example.com"
       }
       ```
    
    2. **Download the Image**:
       ```bash
       GET /api/v1/download/{filename}
       ```
    
    3. **Validate QR Code**:
       ```bash
       GET /api/v1/validate/{filename}
       ```
    
    Visit `/docs` for interactive API documentation.
    """,
    version="1.0.0",
    contact={
        "name": "Soykot Podder",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "body": exc.body
        }
    )

# Initialize pipeline (lazy loading)
pipeline: Optional[ArtisticQRPipeline] = None
validator = QRCodeValidator()

# Output directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


class QRGenerationRequest(BaseModel):
    """Request model for QR code generation"""
    prompt: str = Field(..., description="Text prompt for image generation (e.g., 'A dog in the sky with clouds')", min_length=1, max_length=500)
    qr_data: str = Field(..., description="Data to encode in QR code (URL, text, contact info, etc.)", min_length=1, max_length=2000)
    image_size: int = Field(512, ge=256, le=1024, description="Image size in pixels (256-1024)")
    subtlety: float = Field(0.92, ge=0.85, le=0.95, description="QR code subtlety - higher values make QR more subtle (0.85-0.95). Only used if use_controlnet=False")
    num_inference_steps: int = Field(50, ge=20, le=100, description="Number of inference steps - more steps = better quality but slower (20-100)")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale - how closely to follow the prompt (1.0-20.0)")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility (leave None for random)")
    validate_qr: bool = Field(True, description="Automatically validate QR code scannability after generation")
    use_controlnet: bool = Field(False, description="Use ControlNet for seamless QR integration (like reference image). Generates image WITH QR structure woven in.")
    controlnet_conditioning_scale: float = Field(1.3, ge=0.5, le=2.0, description="ControlNet conditioning scale - higher = stronger QR structure (0.5-2.0). Only used if use_controlnet=True")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A dog in the sky with clouds",
                "qr_data": "https://example.com",
                "image_size": 512,
                "subtlety": 0.92,
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "seed": 42,
                "validate_qr": True,
                "use_controlnet": False,
                "controlnet_conditioning_scale": 1.3
            },
            "example_controlnet": {
                "prompt": "anime style character with white hair, dark patterned outfit, blue sky with mountains, artistic illustration",
                "qr_data": "https://example.com",
                "image_size": 512,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "use_controlnet": True,
                "controlnet_conditioning_scale": 1.3,
                "validate_qr": True
            }
        }


class QRGenerationResponse(BaseModel):
    """Response model for QR code generation"""
    success: bool
    job_id: str
    image_url: str = Field(..., description="URL to download the generated image")
    qr_reference_url: Optional[str] = Field(None, description="URL to download the original QR code for comparison")
    preview_url: Optional[str] = Field(None, description="URL to preview thumbnail")
    scannability_level: Optional[str] = Field(None, description="QR code scannability level (High/Medium/Low/No Scannable)")
    scannable: Optional[bool] = Field(None, description="Whether the QR code is scannable")
    decoded_data: Optional[str] = Field(None, description="Decoded QR code data")
    confidence: Optional[float] = Field(None, description="Scannability confidence score (0.0-1.0)")
    message: str
    generated_at: str = Field(..., description="ISO timestamp of generation")
    parameters: Dict = Field(..., description="Generation parameters used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "job_id": "123e4567-e89b-12d3-a456-426614174000",
                "image_url": "/api/v1/download/qr_123e4567.png",
                "qr_reference_url": "/api/v1/download/qr_123e4567_original_qr.png",
                "preview_url": "/api/v1/preview/qr_123e4567.png?size=256",
                "scannability_level": "High Scannability",
                "scannable": True,
                "decoded_data": "https://example.com",
                "confidence": 0.95,
                "message": "QR code generated successfully",
                "generated_at": "2024-01-01T12:00:00Z",
                "parameters": {
                    "prompt": "A dog in the sky",
                    "image_size": 512,
                    "subtlety": 0.92
                }
            }
        }


class BatchQRGenerationRequest(BaseModel):
    """Request model for batch QR code generation"""
    requests: List[QRGenerationRequest] = Field(..., description="List of QR generation requests", min_items=1, max_items=10)
    
    class Config:
        json_schema_extra = {
            "example": {
                "requests": [
                    {
                        "prompt": "Futuristic city",
                        "qr_data": "https://example.com/1"
                    },
                    {
                        "prompt": "Nature landscape",
                        "qr_data": "https://example.com/2"
                    }
                ]
            }
        }


class BatchQRGenerationResponse(BaseModel):
    """Response model for batch QR code generation"""
    success: bool
    total: int
    completed: int
    failed: int
    results: List[QRGenerationResponse]
    message: str


def get_pipeline():
    """Get or initialize the pipeline"""
    global pipeline
    if pipeline is None:
        pipeline = ArtisticQRPipeline()
    return pipeline


@app.get("/", tags=["Info"])
async def root():
    """
    Root endpoint - API information and available endpoints.
    """
    return {
        "name": "AI Artistic QR Code Generator API",
        "version": "1.0.0",
        "description": "Generate AI-artistic images with embedded scannable QR codes",
        "endpoints": {
            "generate": "/api/v1/generate",
            "batch_generate": "/api/v1/batch-generate",
            "download": "/api/v1/download/{filename}",
            "preview": "/api/v1/preview/{filename}",
            "validate": "/api/v1/validate/{filename}",
            "health": "/api/v1/health",
            "stats": "/api/v1/stats",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "author": "Soykot Podder"
    }


@app.get("/api/v1/health", tags=["Health"])
async def health():
    """
    Health check endpoint - Check API and pipeline status.
    """
    try:
        pipe = get_pipeline()
        model_loaded = pipe.pipe is not None
        
        return {
            "status": "healthy" if model_loaded else "degraded",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "pipeline": {
                "loaded": model_loaded,
                "device": pipe.device,
                "model_id": pipe.model_id
            },
            "validator": {
                "available": True
            },
            "output_directory": {
                "path": str(OUTPUT_DIR),
                "exists": OUTPUT_DIR.exists(),
                "writable": os.access(OUTPUT_DIR, os.W_OK) if OUTPUT_DIR.exists() else False
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": str(e)
        }


@app.post("/api/v1/generate", response_model=QRGenerationResponse, tags=["Generation"])
async def generate_qr_code(
    request: QRGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate an artistic QR code image.
    
    This endpoint generates an AI-artistic image with an embedded scannable QR code using Stable Diffusion.
    
    **Two Generation Modes:**
    
    1. **Post-Processing Embedding** (default, `use_controlnet=false`):
       - Generate image first, then embed QR code
       - Good control over subtlety
       - Image quality independent of QR code
    
    2. **ControlNet Integration** (`use_controlnet=true`):
       - Generate image WITH QR code structure woven in
       - More seamless integration (like reference image)
       - QR pattern is part of the generation process
       - Better for artistic integration where QR is woven into artwork
    
    **Process:**
    1. Create QR code from provided data
    2. Generate AI image (with or without QR guidance)
    3. Embed/Integrate QR code artistically
    4. Validate QR code scannability (if enabled)
    
    **Response includes:**
    - Download URLs for the generated image and original QR code
    - QR code validation results
    - Generation metadata
    """
    start_time = time.time()
    job_id = str(uuid.uuid4())
    
    try:
        logger.info(f"[{job_id}] Starting QR code generation")
        logger.info(f"[{job_id}] Prompt: {request.prompt[:50]}...")
        logger.info(f"[{job_id}] QR Data: {request.qr_data[:50]}...")
        
        # Generate unique job ID
        output_filename = f"qr_{job_id}.png"
        output_path = OUTPUT_DIR / output_filename
        qr_reference_path = OUTPUT_DIR / f"qr_{job_id}_original_qr.png"
        
        # Get pipeline
        pipe = get_pipeline()
        
        # Generate QR code image
        final_image = pipe.process(
            prompt=request.prompt,
            qr_data=request.qr_data,
            output_path=str(output_path),
            image_size=request.image_size,
            subtlety=request.subtlety,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            use_controlnet=request.use_controlnet,
            controlnet_conditioning_scale=request.controlnet_conditioning_scale
        )
        
        generation_time = time.time() - start_time
        logger.info(f"[{job_id}] Generation completed in {generation_time:.2f}s")
        
        # Validate QR code if requested
        scannability_level = None
        scannable = None
        decoded_data = None
        confidence = None
        
        if request.validate_qr:
            try:
                logger.info(f"[{job_id}] Validating QR code...")
                result = validator.validate_qr_code(str(output_path), request.qr_data)
                scannability_level = validator.assess_scannability_level(str(output_path))
                scannable = result['scannable']
                decoded_data = result.get('data_decoded')
                confidence = result.get('confidence', 0.0)
                logger.info(f"[{job_id}] Validation: {scannability_level} (confidence: {confidence:.2f})")
            except Exception as e:
                logger.warning(f"[{job_id}] Validation error: {e}")
        
        # Clean up old files in background
        background_tasks.add_task(cleanup_old_files)
        
        # Prepare response
        response = QRGenerationResponse(
            success=True,
            job_id=job_id,
            image_url=f"/api/v1/download/{output_filename}",
            qr_reference_url=f"/api/v1/download/{qr_reference_path.name}" if qr_reference_path.exists() else None,
            preview_url=f"/api/v1/preview/{output_filename}?size=256",
            scannability_level=scannability_level,
            scannable=scannable,
            decoded_data=decoded_data,
            confidence=confidence,
            message="QR code generated successfully",
            generated_at=datetime.utcnow().isoformat() + "Z",
            parameters={
                "prompt": request.prompt,
                "qr_data": request.qr_data,
                "image_size": request.image_size,
                "subtlety": request.subtlety,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed,
                "generation_time_seconds": round(generation_time, 2)
            }
        )
        
        logger.info(f"[{job_id}] Request completed successfully")
        return response
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{job_id}] Error: {error_msg}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error generating QR code: {error_msg}"
        )


@app.post("/api/v1/batch-generate", response_model=BatchQRGenerationResponse, tags=["Generation"])
async def batch_generate_qr_codes(
    request: BatchQRGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate multiple artistic QR codes in batch.
    
    Generate up to 10 QR codes in a single request. Each QR code is generated independently.
    """
    logger.info(f"Starting batch generation of {len(request.requests)} QR codes")
    start_time = time.time()
    
    results = []
    completed = 0
    failed = 0
    
    for i, req in enumerate(request.requests):
        try:
            logger.info(f"Processing batch item {i+1}/{len(request.requests)}")
            response = await generate_qr_code(req, background_tasks)
            results.append(response)
            completed += 1
        except Exception as e:
            logger.error(f"Batch item {i+1} failed: {e}")
            failed += 1
            # Create error response
            results.append(QRGenerationResponse(
                success=False,
                job_id=str(uuid.uuid4()),
                image_url="",
                message=f"Generation failed: {str(e)}",
                generated_at=datetime.utcnow().isoformat() + "Z",
                parameters={}
            ))
    
    total_time = time.time() - start_time
    logger.info(f"Batch generation completed: {completed} succeeded, {failed} failed in {total_time:.2f}s")
    
    return BatchQRGenerationResponse(
        success=completed > 0,
        total=len(request.requests),
        completed=completed,
        failed=failed,
        results=results,
        message=f"Batch generation completed: {completed} succeeded, {failed} failed"
    )


@app.get("/api/v1/download/{filename}", tags=["Files"])
async def download_file(filename: str):
    """
    Download a generated QR code image file.
    
    Returns the full-resolution PNG image.
    """
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Invalid file: {filename}")
    
    # Security: Ensure filename doesn't contain path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="image/png"
    )


@app.get("/api/v1/preview/{filename}", tags=["Files"])
async def preview_file(
    filename: str,
    size: int = Query(256, ge=64, le=512, description="Preview size in pixels")
):
    """
    Get a preview/thumbnail of a generated QR code image.
    
    Returns a resized version of the image for quick preview.
    """
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Invalid file: {filename}")
    
    # Security check
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    try:
        # Open and resize image
        img = Image.open(file_path)
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        return StreamingResponse(
            img_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f'inline; filename="preview_{filename}"'}
        )
    except Exception as e:
        logger.error(f"Error creating preview: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating preview: {str(e)}")


@app.get("/api/v1/validate/{filename}", tags=["Validation"])
async def validate_qr_code(
    filename: str,
    expected_data: Optional[str] = Query(None, description="Expected QR code data for verification")
):
    """
    Validate QR code scannability in a generated image.
    
    Tests the QR code using multiple decoding methods and returns scannability assessment.
    """
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Invalid file: {filename}")
    
    # Security check
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    try:
        logger.info(f"Validating QR code: {filename}")
        result = validator.validate_qr_code(str(file_path), expected_data)
        scannability_level = validator.assess_scannability_level(str(file_path))
        
        return {
            "filename": filename,
            "scannable": result['scannable'],
            "scannability_level": scannability_level,
            "decoded_data": result.get('data_decoded'),
            "matches_expected": result.get('matches_expected', False),
            "confidence": result.get('confidence', 0.0),
            "methods_tried": result.get('methods_tried', []),
            "errors": result.get('errors', []),
            "validated_at": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")


def cleanup_old_files(max_age_hours: int = 24):
    """Clean up old generated files"""
    import time
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for file_path in OUTPUT_DIR.glob("qr_*.png"):
        try:
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                file_path.unlink()
                print(f"Cleaned up old file: {file_path.name}")
        except Exception as e:
            print(f"Error cleaning up {file_path}: {e}")


@app.get("/api/v1/stats", tags=["Info"])
async def get_stats():
    """
    Get API statistics and usage information.
    
    Returns information about generated files, storage usage, and system status.
    """
    try:
        files = list(OUTPUT_DIR.glob("qr_*.png"))
        file_count = len(files)
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        
        # Get file age statistics
        import time
        current_time = time.time()
        file_ages = []
        for f in files:
            if f.is_file():
                age_hours = (current_time - f.stat().st_mtime) / 3600
                file_ages.append(age_hours)
        
        avg_age_hours = sum(file_ages) / len(file_ages) if file_ages else 0
        
        return {
            "statistics": {
                "total_files": file_count,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "total_size_gb": round(total_size / (1024 * 1024 * 1024), 3),
                "average_file_size_mb": round((total_size / file_count) / (1024 * 1024), 2) if file_count > 0 else 0,
                "average_file_age_hours": round(avg_age_hours, 2)
            },
            "output_directory": {
                "path": str(OUTPUT_DIR),
                "exists": OUTPUT_DIR.exists(),
                "writable": os.access(OUTPUT_DIR, os.W_OK) if OUTPUT_DIR.exists() else False
            },
            "pipeline": {
                "model_loaded": pipeline.pipe is not None if pipeline else False,
                "device": pipeline.device if pipeline else None
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

