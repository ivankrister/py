#!/usr/bin/env python3
"""
Test current OCR on gcash.jpeg to analyze the image structure
"""

import cv2
import numpy as np
import pytesseract
import os

def test_gcash_current():
    """Test current OCR methods on gcash.jpeg"""
    
    if not os.path.exists("gcash.jpeg"):
        print("❌ gcash.jpeg not found!")
        return
    
    print("Testing current OCR methods on gcash.jpeg")
    print("=" * 60)
    
    # Load image
    image = cv2.imread("gcash.jpeg")
    if image is None:
        print("❌ Could not load image")
        return
    
    print(f"✅ Image loaded: {image.shape}")
    
    # Test different preprocessing methods
    methods = {
        "original": lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        "adaptive_threshold": lambda img: cv2.adaptiveThreshold(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        ),
        "otsu_threshold": lambda img: cv2.threshold(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1],
        "scaled_2x": lambda img: cv2.resize(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )
    }
    
    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-+₱$ '
    
    all_texts = []
    
    for method_name, method_func in methods.items():
        print(f"\n{method_name.upper()}:")
        print("-" * 40)
        
        try:
            processed = method_func(image)
            text = pytesseract.image_to_string(processed, config=config).strip()
            all_texts.append(text)
            
            print(f"Text: '{text}'")
            
            # Check for expected values
            checks = {
                "10.00": "10.00" in text,
                "+639296681405": "+639296681405" in text or "639296681405" in text,
                "9032469742237": "9032469742237" in text
            }
            
            for item, found in checks.items():
                status = "✅" if found else "❌"
                print(f"   {status} {item}: {found}")
            
            # Save processed image
            cv2.imwrite(f"gcash_{method_name}.png", processed)
            print(f"   Saved: gcash_{method_name}.png")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    # Look for patterns in all texts
    print(f"\n" + "=" * 60)
    print("PATTERN ANALYSIS:")
    
    all_text_combined = "\n".join(all_texts)
    
    # Look for numbers
    import re
    numbers = re.findall(r'\d+\.?\d*', all_text_combined)
    unique_numbers = list(set(numbers))
    print(f"All numbers found: {unique_numbers}")
    
    # Look for phone patterns
    phone_patterns = re.findall(r'[\+]?[0-9]{10,13}', all_text_combined)
    print(f"Phone-like patterns: {phone_patterns}")
    
    # Look for reference patterns (long numbers)
    ref_patterns = re.findall(r'\d{10,15}', all_text_combined)
    print(f"Reference-like patterns: {ref_patterns}")

if __name__ == "__main__":
    test_gcash_current()
