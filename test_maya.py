#!/usr/bin/env python3
"""
Test script specifically for maya.jpeg to analyze OCR accuracy
"""

import requests
import json
import os
from pathlib import Path

API_BASE_URL = "http://localhost:8001"
IMAGE_PATH = "maya.jpeg"

def test_maya_image():
    """Test OCR on maya.jpeg image"""
    print("Testing OCR on maya.jpeg")
    print("=" * 50)
    
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Image {IMAGE_PATH} not found!")
        return
    
    try:
        # Test health first
        health_response = requests.get(f"{API_BASE_URL}/health")
        if health_response.status_code != 200:
            print(f"❌ API health check failed: {health_response.status_code}")
            return
        
        print("✅ API is healthy")
        
        # Test with basic extraction
        print(f"\n🔍 Testing basic text extraction on {IMAGE_PATH}...")
        with open(IMAGE_PATH, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_BASE_URL}/extract-text", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Basic OCR extraction successful")
            print(f"Confidence: {result['confidence']}%")
            print(f"Word count: {result['word_count']}")
            print(f"Image dimensions: {result.get('image_dimensions', 'N/A')}")
            print(f"\nExtracted text:")
            print("-" * 30)
            print(result['extracted_text'])
            print("-" * 30)
            
            # Check if 10.00 is detected
            extracted_text = result['extracted_text']
            if "10.00" in extracted_text:
                print("✅ Amount 10.00 DETECTED in extracted text!")
            else:
                print("❌ Amount 10.00 NOT DETECTED in extracted text")
                print("Looking for similar patterns...")
                
                # Look for similar patterns
                import re
                money_patterns = re.findall(r'\d+\.?\d*', extracted_text)
                print(f"Found number patterns: {money_patterns}")
                
        else:
            print(f"❌ Basic OCR extraction failed: {response.status_code}")
            print(response.text)
            return
        
        # Test with detailed extraction
        print(f"\n🔍 Testing detailed text extraction on {IMAGE_PATH}...")
        with open(IMAGE_PATH, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_BASE_URL}/extract-text-detailed", files=files)
        
        if response.status_code == 200:
            detailed_result = response.json()
            print("✅ Detailed OCR extraction successful")
            
            ocr_results = detailed_result['ocr_results']
            structured_data = ocr_results.get('structured_data', {})
            
            print(f"\nDetailed Analysis:")
            print(f"Total words detected: {structured_data.get('total_words', 0)}")
            print(f"Total lines detected: {structured_data.get('total_lines', 0)}")
            
            # Show word-level analysis
            words = structured_data.get('words', [])
            if words:
                print(f"\nWord-level confidence analysis:")
                for i, word in enumerate(words[:20]):  # Show first 20 words
                    print(f"  {i+1:2d}. '{word['text']}' (confidence: {word['confidence']}%)")
                
                if len(words) > 20:
                    print(f"  ... and {len(words) - 20} more words")
            
            # Show lines
            lines = structured_data.get('lines', [])
            if lines:
                print(f"\nLine-by-line text:")
                for i, line in enumerate(lines):
                    print(f"  Line {i+1:2d}: {line}")
            
        else:
            print(f"❌ Detailed OCR extraction failed: {response.status_code}")
            print(response.text)
    
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running on port 8001.")
    except Exception as e:
        print(f"❌ Error: {e}")

def analyze_image_preprocessing():
    """Analyze the image and suggest improvements"""
    print(f"\n🔬 Analyzing {IMAGE_PATH} for preprocessing improvements...")
    
    try:
        import cv2
        import numpy as np
        from PIL import Image
        
        # Load and analyze the image
        image = cv2.imread(IMAGE_PATH)
        if image is None:
            print("❌ Could not load image")
            return
        
        height, width = image.shape[:2]
        print(f"Image dimensions: {width}x{height}")
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Analyze image quality
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"Blur score: {blur_score:.2f} (higher is better, >100 is good)")
        
        # Analyze brightness
        brightness = np.mean(gray)
        print(f"Average brightness: {brightness:.2f} (0-255, ~127 is ideal)")
        
        # Analyze contrast
        contrast = gray.std()
        print(f"Contrast score: {contrast:.2f} (higher is better, >50 is good)")
        
        # Suggestions
        print(f"\n💡 Preprocessing suggestions:")
        if blur_score < 100:
            print("  - Image appears blurry, consider sharpening")
        if brightness < 100:
            print("  - Image is dark, consider brightness enhancement")
        elif brightness > 180:
            print("  - Image is too bright, consider reducing brightness")
        if contrast < 50:
            print("  - Low contrast, consider contrast enhancement")
        
    except ImportError:
        print("⚠️ OpenCV not available for image analysis")
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")

if __name__ == "__main__":
    test_maya_image()
    analyze_image_preprocessing()
    
    print("\n" + "=" * 50)
    print("Maya.jpeg OCR test complete!")
    print("If 10.00 was not detected, consider:")
    print("1. Image quality improvements")
    print("2. Enhanced preprocessing")
    print("3. Alternative OCR configurations")
