#!/usr/bin/env python3
"""
Debug Maya OCR - Check what text is being extracted
"""

import cv2
import pytesseract
from PIL import Image
import re

def debug_maya_ocr():
    # Load Maya image
    image = cv2.imread('maya.jpeg')
    if image is None:
        print("❌ Could not load maya.jpeg")
        return
    
    print("🔍 Maya OCR Debug Results")
    print("="*50)
    
    # Test different preprocessing methods
    methods = {
        'Original': image,
        'Grayscale': cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        'Adaptive Threshold': cv2.adaptiveThreshold(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 
            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    }
    
    # Test different Tesseract configs
    configs = {
        'default': '--psm 6',
        'enhanced': '--psm 6 --oem 3',
        'single_column': '--psm 4',
        'single_word': '--psm 8'
    }
    
    for method_name, processed_image in methods.items():
        print(f"\n📄 {method_name.upper()}:")
        
        # Convert to PIL for tesseract
        if len(processed_image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(processed_image)
        
        for config_name, config in configs.items():
            try:
                text = pytesseract.image_to_string(pil_image, config=config).strip()
                if text:
                    print(f"\n  {config_name}:")
                    print(f"    Raw text: {repr(text)}")
                    
                    # Try to extract our target fields
                    # Amount
                    amount_patterns = [
                        r'P(\d+\.?\d*)',
                        r'(\d+\.\d{2})',
                        r'P\s*(\d+\.\d{2})',
                        r'₱\s*(\d+\.\d{2})'
                    ]
                    amount = None
                    for pattern in amount_patterns:
                        match = re.search(pattern, text)
                        if match:
                            amount = match.group(1)
                            break
                    
                    # Phone
                    phone_pattern = r'\+?639\d{9}'
                    phone_match = re.search(phone_pattern, text)
                    phone = phone_match.group() if phone_match else None
                    if phone and not phone.startswith('+'):
                        phone = '+' + phone
                    
                    # Reference ID
                    ref_patterns = [
                        r'Reference[:\s]*ID[:\s]*([A-Z0-9]{12})',
                        r'Reference[:\s]*([A-Z0-9]{12})',
                        r'Ref[\.:\s]*([A-Z0-9]{12})',
                        r'([A-Z0-9]{12})',
                        r'([A-Z0-9]{8,15})'  # More flexible
                    ]
                    reference = None
                    for pattern in ref_patterns:
                        match = re.search(pattern, text)
                        if match:
                            reference = match.group(1)
                            break
                    
                    print(f"    Extracted amount: {amount}")
                    print(f"    Extracted phone: {phone}")
                    print(f"    Extracted reference: {reference}")
                    
                    # Check if we got all expected values
                    expected = {
                        'amount': '10.00',
                        'phone': '+639772478589',
                        'reference': 'EB8C64C5C67B'
                    }
                    
                    matches = 0
                    if amount == expected['amount']:
                        matches += 1
                        print("    ✅ Amount matches!")
                    if phone == expected['phone']:
                        matches += 1
                        print("    ✅ Phone matches!")
                    if reference == expected['reference']:
                        matches += 1
                        print("    ✅ Reference matches!")
                    
                    if matches == 3:
                        print(f"    🎯 PERFECT MATCH with {method_name} + {config_name}!")
                    elif matches > 0:
                        print(f"    ⚠️  Partial match ({matches}/3) with {method_name} + {config_name}")
                else:
                    print(f"  {config_name}: No text extracted")
            except Exception as e:
                print(f"  {config_name}: Error - {e}")

if __name__ == "__main__":
    debug_maya_ocr()
