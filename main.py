from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import logging
from ocr_processor import ReceiptOCR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Receipt OCR API",
    description="High-accuracy OCR API for extracting text from receipt images using OpenCV and Tesseract",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR processor
ocr_processor = ReceiptOCR()

# Pydantic models for request/response
class OCRResponse(BaseModel):
    extracted_text: str
    receipt_data: dict

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API information"""
    return HealthResponse(
        status="healthy",
        message="Receipt OCR API is running",
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test if Tesseract is available
        import pytesseract
        pytesseract.get_tesseract_version()
        
        return HealthResponse(
            status="healthy",
            message="All services are operational",
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: {str(e)}"
        )

@app.post("/extract-text", response_model=OCRResponse)
async def extract_text_from_image(file: UploadFile = File(...)):
    """
    Extract text from uploaded receipt image
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
        
    Returns:
        OCRResponse with extracted text and metadata
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file."
        )
    
    # Validate file size (10MB limit)
    max_size = 10 * 1024 * 1024  # 10MB
    
    try:
        # Read file content
        content = await file.read()
        
        if len(content) > max_size:
            raise HTTPException(
                status_code=413,
                detail="File size too large. Maximum size is 10MB."
            )
        
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded."
            )
        
        logger.info(f"Processing image: {file.filename}, size: {len(content)} bytes")
        
        # Process the image
        result = ocr_processor.extract_text(content)
        
        if result.get('error'):
            raise HTTPException(
                status_code=422,
                detail=f"OCR processing failed: {result['error']}"
            )
        
        logger.info(f"OCR completed. Confidence: {result['confidence']}% (Adjusted: {result.get('adjusted_confidence', result['confidence'])}%), Words: {result['word_count']}")
        logger.info(f"Method used: {result.get('preprocessing_method', 'unknown')} + {result.get('tesseract_config', 'default')}")
        
        return OCRResponse(
            extracted_text=result["extracted_text"],
            receipt_data=result["receipt_data"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/extract-text-detailed", response_model=dict)
async def extract_text_detailed(file: UploadFile = File(...)):
    """
    Extract text with detailed word-level information
    
    Args:
        file: Uploaded image file
        
    Returns:
        Detailed OCR results with bounding boxes and confidence scores
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file."
        )
    
    try:
        # Read file content
        content = await file.read()
        
        logger.info(f"Processing detailed OCR for: {file.filename}")
        
        # Process the image
        result = ocr_processor.extract_text(content)
        
        if result.get('error'):
            raise HTTPException(
                status_code=422,
                detail=f"OCR processing failed: {result['error']}"
            )
        
        # Return full detailed response
        return {
            "filename": file.filename,
            "file_size": len(content),
            "ocr_results": result,
            "api_version": "1.0.0"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in detailed processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported image formats"""
    return {
        "supported_formats": [
            "JPEG", "JPG", "PNG", "BMP", "TIFF", "TIF", "WEBP"
        ],
        "max_file_size": "10MB",
        "recommended_format": "PNG or JPEG",
        "tips": [
            "Higher resolution images generally produce better results",
            "Ensure good lighting and minimal shadows",
            "Avoid blurry or skewed images",
            "Remove any obstructions from the receipt"
        ]
    }

if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting Receipt OCR API on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
