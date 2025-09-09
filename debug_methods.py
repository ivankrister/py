#!/usr/bin/env python3
"""
Debug test to see exactly what each method is detecting
"""

import cv2
import numpy as np
import pytesseract
import os

def debug_all_methods():
    """Debug all preprocessing methods on maya.jpeg"""
    
    if not os.path.exists("maya.jpeg"):
        print("❌ maya.jpeg not found!")
        return
    
    print("Debug: All preprocessing methods on maya.jpeg")
    print("=" * 70)
    
    # Load image
    image = cv2.imread("maya.jpeg")
    if image is None:
        print("❌ Could not load image")
        return
    
    print(f"Image loaded: {image.shape}")
    
    # Test each method
    methods = {
        "original": lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        "adaptive_only": lambda img: cv2.adaptiveThreshold(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    }
    
    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-$€£¥₹ '
    
    for method_name, method_func in methods.items():
        print(f"\n{method_name.upper()}:")
        print("-" * 40)
        
        try:
            processed = method_func(image)
            text = pytesseract.image_to_string(processed, config=config).strip()
            
            print(f"Text: '{text}'")
            
            if "10.00" in text:
                print("✅ FOUND 10.00!")
            else:
                print("❌ 10.00 not found")
            
            # Save processed image
            cv2.imwrite(f"debug_{method_name}.png", processed)
            print(f"Saved: debug_{method_name}.png")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    debug_all_methods()
