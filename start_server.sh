#!/bin/bash

# Receipt OCR API Startup Script

echo "🚀 Starting Receipt OCR API..."
echo "================================"

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if Tesseract is installed
echo "🔍 Checking Tesseract installation..."
if ! command -v tesseract &> /dev/null; then
    echo "❌ Tesseract not found. Please install it with:"
    echo "   brew install tesseract"
    exit 1
fi

tesseract_version=$(tesseract --version 2>&1 | head -n 1)
echo "✅ $tesseract_version"

# Start the API server
echo "🌐 Starting API server on http://localhost:8000..."
echo ""
echo "📖 API Documentation will be available at:"
echo "   • Interactive docs: http://localhost:8000/docs"
echo "   • ReDoc: http://localhost:8000/redoc"
echo ""
echo "🧪 Test endpoints:"
echo "   • Health check: http://localhost:8000/health"
echo "   • Supported formats: http://localhost:8000/supported-formats"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================"

python main.py
