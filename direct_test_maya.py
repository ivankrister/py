#!/usr/bin/env python3
"""
Direct OCR test on maya.jpeg without API
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import os

def test_maya_direct():
    """Test OCR directly on maya.jpeg"""
    image_path = "maya.jpeg"
    
    if not os.path.exists(image_path):
        print(f"❌ Image {image_path} not found!")
        return
    
    print("Testing maya.jpeg with direct OCR")
    print("=" * 50)
    
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Could not load image")
            return
        
        print(f"✅ Image loaded: {image.shape}")
        
        # Original image OCR
        print("\n1. Testing original image...")
        gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text_original = pytesseract.image_to_string(gray_original)
        print(f"Original text: '{text_original.strip()}'")
        
        if "10.00" in text_original:
            print("✅ Found 10.00 in original!")
        else:
            print("❌ 10.00 not found in original")
        
        # Preprocessing method 1: Simple threshold
        print("\n2. Testing with simple threshold...")
        _, thresh1 = cv2.threshold(gray_original, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_thresh1 = pytesseract.image_to_string(thresh1)
        print(f"Threshold text: '{text_thresh1.strip()}'")
        
        if "10.00" in text_thresh1:
            print("✅ Found 10.00 with threshold!")
        else:
            print("❌ 10.00 not found with threshold")
        
        # Preprocessing method 2: Adaptive threshold
        print("\n3. Testing with adaptive threshold...")
        adaptive_thresh = cv2.adaptiveThreshold(gray_original, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        text_adaptive = pytesseract.image_to_string(adaptive_thresh)
        print(f"Adaptive text: '{text_adaptive.strip()}'")
        
        if "10.00" in text_adaptive:
            print("✅ Found 10.00 with adaptive threshold!")
        else:
            print("❌ 10.00 not found with adaptive threshold")
        
        # Preprocessing method 3: Blur + threshold
        print("\n4. Testing with blur + threshold...")
        blurred = cv2.GaussianBlur(gray_original, (5, 5), 0)
        _, thresh_blur = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_blur = pytesseract.image_to_string(thresh_blur)
        print(f"Blur+threshold text: '{text_blur.strip()}'")
        
        if "10.00" in text_blur:
            print("✅ Found 10.00 with blur+threshold!")
        else:
            print("❌ 10.00 not found with blur+threshold")
        
        # Preprocessing method 4: Scale up
        print("\n5. Testing with image scaling...")
        scaled = cv2.resize(gray_original, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        text_scaled = pytesseract.image_to_string(scaled)
        print(f"Scaled text: '{text_scaled.strip()}'")
        
        if "10.00" in text_scaled:
            print("✅ Found 10.00 with scaling!")
        else:
            print("❌ 10.00 not found with scaling")
        
        # Preprocessing method 5: Different Tesseract config
        print("\n6. Testing with different Tesseract config...")
        config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.$ '
        text_config = pytesseract.image_to_string(gray_original, config=config)
        print(f"Custom config text: '{text_config.strip()}'")
        
        if "10.00" in text_config:
            print("✅ Found 10.00 with custom config!")
        else:
            print("❌ 10.00 not found with custom config")
        
        # Try digit-only recognition
        print("\n7. Testing digit-only recognition...")
        config_digits = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.'
        text_digits = pytesseract.image_to_string(gray_original, config=config_digits)
        print(f"Digits only text: '{text_digits.strip()}'")
        
        if "10.00" in text_digits:
            print("✅ Found 10.00 with digits-only!")
        else:
            print("❌ 10.00 not found with digits-only")
        
        # Save processed images for inspection
        cv2.imwrite("maya_original_gray.png", gray_original)
        cv2.imwrite("maya_threshold.png", thresh1)
        cv2.imwrite("maya_adaptive.png", adaptive_thresh)
        cv2.imwrite("maya_blur_thresh.png", thresh_blur)
        cv2.imwrite("maya_scaled.png", scaled)
        
        print(f"\n📁 Saved processed images for inspection:")
        print("  - maya_original_gray.png")
        print("  - maya_threshold.png") 
        print("  - maya_adaptive.png")
        print("  - maya_blur_thresh.png")
        print("  - maya_scaled.png")
        
        # Analyze all results
        all_texts = [text_original, text_thresh1, text_adaptive, text_blur, text_scaled, text_config, text_digits]
        
        print(f"\n📊 Summary:")
        found_10_00 = any("10.00" in text for text in all_texts)
        if found_10_00:
            print("✅ 10.00 was found in at least one method!")
        else:
            print("❌ 10.00 was not found in any method")
            
            # Look for similar numbers
            import re
            all_numbers = []
            for text in all_texts:
                numbers = re.findall(r'\d+\.?\d*', text)
                all_numbers.extend(numbers)
            
            unique_numbers = list(set(all_numbers))
            print(f"🔍 All numbers detected: {unique_numbers}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_maya_direct()
