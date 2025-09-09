#!/usr/bin/env python3
"""
Simple demo script for the Receipt OCR API
"""

import requests
import json
import sys
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def check_server():
    """Check if the API server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_with_image(image_path):
    """Test the API with an image file"""
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        return
    
    print(f"📷 Testing with image: {image_path}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_BASE_URL}/extract-text", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ OCR extraction successful!")
            print(f"📊 Confidence: {result['confidence']}%")
            print(f"📝 Word count: {result['word_count']}")
            print("📄 Extracted text:")
            print("-" * 40)
            print(result['extracted_text'])
            print("-" * 40)
            
            if result.get('structured_data'):
                print(f"📋 Lines detected: {result['structured_data'].get('total_lines', 0)}")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("Receipt OCR API Demo")
    print("=" * 30)
    
    # Check if server is running
    if not check_server():
        print("❌ API server is not running!")
        print("Please start the server first:")
        print("   ./start_server.sh")
        print("   or")
        print("   python main.py")
        return
    
    print("✅ API server is running")
    
    # Test basic endpoints
    try:
        response = requests.get(f"{API_BASE_URL}/supported-formats")
        if response.status_code == 200:
            formats = response.json()
            print("📋 Supported formats:", ", ".join(formats['supported_formats']))
    except:
        pass
    
    # Check for command line image argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_with_image(image_path)
    else:
        # Look for sample images
        sample_images = [
            "sample_receipt.jpg", "sample_receipt.png",
            "test_receipt.jpg", "test_receipt.png",
            "receipt.jpg", "receipt.png"
        ]
        
        found_image = None
        for img in sample_images:
            if Path(img).exists():
                found_image = img
                break
        
        if found_image:
            test_with_image(found_image)
        else:
            print("\n📋 Usage:")
            print(f"   {sys.argv[0]} <image_path>")
            print("\n💡 Tips:")
            print("   • Place a receipt image in this directory")
            print("   • Supported formats: JPEG, PNG, BMP, TIFF, WEBP")
            print("   • For best results, use clear, well-lit images")

if __name__ == "__main__":
    main()
