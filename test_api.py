#!/usr/bin/env python3
"""
Test script for the Receipt OCR API
"""

import requests
import json
import os
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(response.text)
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running.")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_supported_formats():
    """Test the supported formats endpoint"""
    print("\nTesting supported formats...")
    try:
        response = requests.get(f"{API_BASE_URL}/supported-formats")
        if response.status_code == 200:
            print("✅ Supported formats retrieved")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Failed to get supported formats: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_ocr_with_sample_image():
    """Test OCR with a sample image (if available)"""
    print("\nTesting OCR extraction...")
    
    # Look for sample images in common locations
    sample_paths = [
        "sample_receipt.jpg",
        "sample_receipt.png", 
        "test_receipt.jpg",
        "test_receipt.png"
    ]
    
    sample_image = None
    for path in sample_paths:
        if os.path.exists(path):
            sample_image = path
            break
    
    if not sample_image:
        print("⚠️  No sample image found. Place a receipt image in the current directory.")
        print("   Supported names: sample_receipt.jpg, sample_receipt.png, test_receipt.jpg, test_receipt.png")
        return
    
    try:
        with open(sample_image, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_BASE_URL}/extract-text", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ OCR extraction successful")
            print(f"Confidence: {result['confidence']}%")
            print(f"Word count: {result['word_count']}")
            print(f"Extracted text preview: {result['extracted_text'][:200]}...")
        else:
            print(f"❌ OCR extraction failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error: {e}")

def create_sample_test_image():
    """Create a simple test image with text"""
    print("\nCreating sample test image...")
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io
        
        # Create a simple image with text
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font
        try:
            font = ImageFont.truetype("Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw some sample receipt text
        text = """SAMPLE STORE
123 Main Street
Receipt #12345
Date: 2024-01-01

Item 1         $10.99
Item 2          $5.50
Tax             $1.32
Total          $17.81"""
        
        draw.multiline_text((10, 10), text, fill='black', font=font)
        
        # Save the image
        img.save('test_sample.png')
        print("✅ Created test_sample.png")
        
        # Test with this image
        with open('test_sample.png', 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_BASE_URL}/extract-text", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ OCR test with generated image successful")
            print(f"Confidence: {result['confidence']}%")
            print(f"Extracted text:\n{result['extracted_text']}")
        else:
            print(f"❌ OCR test failed: {response.status_code}")
            print(response.text)
            
    except ImportError:
        print("⚠️  PIL not available for creating test image. Install with: pip install Pillow")
    except Exception as e:
        print(f"❌ Error creating test image: {e}")

if __name__ == "__main__":
    print("Receipt OCR API Test Script")
    print("=" * 40)
    
    test_health_check()
    test_supported_formats() 
    test_ocr_with_sample_image()
    
    # If no sample image found, create one
    if not any(os.path.exists(p) for p in ["sample_receipt.jpg", "sample_receipt.png", "test_receipt.jpg", "test_receipt.png"]):
        create_sample_test_image()
    
    print("\n" + "=" * 40)
    print("Test complete!")
