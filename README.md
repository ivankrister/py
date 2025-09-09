# Receipt OCR API

A high-accuracy web API for extracting text from receipt images using OpenCV and Tesseract OCR.

## Features

- **High-accuracy OCR**: Advanced image preprocessing for better text extraction
- **Multiple preprocessing techniques**: Gaussian blur, adaptive thresholding, deskewing, and more
- **RESTful API**: Easy-to-use FastAPI endpoints
- **Image format support**: JPEG, PNG, BMP, TIFF, WEBP
- **Structured output**: Word-level confidence scores and bounding boxes
- **Error handling**: Comprehensive error handling and validation
- **CORS enabled**: Ready for web applications

## Installation

### Prerequisites

1. **Install Tesseract OCR**:
   ```bash
   # macOS
   brew install tesseract
   
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # Windows
   # Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Setup

1. Clone or download this project
2. Install dependencies as shown above
3. Run the API:
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Basic Text Extraction
```
POST /extract-text
```
Upload an image file and get extracted text with confidence score.

**Request**: Multipart form data with image file
**Response**:
```json
{
  "extracted_text": "Receipt text here...",
  "confidence": 85.5,
  "word_count": 25,
  "structured_data": {
    "total_words": 25,
    "total_lines": 8,
    "words": [...],
    "lines": [...]
  },
  "preprocessing_applied": true,
  "image_dimensions": {
    "width": 800,
    "height": 1200
  }
}
```

### 2. Detailed Text Extraction
```
POST /extract-text-detailed
```
Get detailed OCR results with word-level bounding boxes and confidence scores.

### 3. Health Check
```
GET /health
```
Check if the API and Tesseract are working properly.

### 4. Supported Formats
```
GET /supported-formats
```
Get information about supported image formats and usage tips.

## Usage Examples

### Python
```python
import requests

# Upload and process image
with open('receipt.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/extract-text', files=files)
    result = response.json()
    print(f"Extracted text: {result['extracted_text']}")
    print(f"Confidence: {result['confidence']}%")
```

### cURL
```bash
curl -X POST "http://localhost:8000/extract-text" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@receipt.jpg"
```

### JavaScript (Browser)
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/extract-text', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Extracted text:', data.extracted_text);
    console.log('Confidence:', data.confidence);
});
```

## Image Preprocessing

The API applies several preprocessing techniques to improve OCR accuracy:

1. **Grayscale conversion**
2. **Gaussian blur** - Noise reduction
3. **Adaptive thresholding** - Better contrast
4. **Morphological operations** - Text cleanup
5. **Deskewing** - Rotation correction
6. **CLAHE** - Contrast enhancement
7. **Sharpening** - Edge enhancement

For low-confidence results, alternative preprocessing is automatically applied.

## Configuration

Environment variables:
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `DEBUG`: Enable debug mode (default: false)

## Tips for Better Results

1. **Image Quality**: Use high-resolution, well-lit images
2. **Orientation**: Ensure receipts are upright and not skewed
3. **Format**: PNG and JPEG work best
4. **Size**: Keep files under 10MB
5. **Clarity**: Avoid blurry or damaged receipts

## API Documentation

Once the server is running, visit:
- **Interactive docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Error Handling

The API provides detailed error messages for:
- Invalid file types
- File size limits
- OCR processing errors
- Service unavailability

## Performance

- Typical processing time: 1-3 seconds per image
- Memory usage: ~50-100MB per request
- Concurrent requests: Supported

## License

This project is open source. Feel free to modify and distribute.
