#!/usr/bin/env python3

from ocr_processor import ReceiptOCR
import re

def test_structured_extraction():
    """Test the structured data extraction specifically"""
    
    ocr = ReceiptOCR()
    
    # The extracted text from maya2.jpeg
    text = """Received money from

P'1.00 m
+639 162907953 re]
v Completed Sep &, 2025, 01:08 pm

- from Sarah Jane

Reference ID 172F EDCS5 80BD

maya"""
    
    print("Original text:")
    print(f"'{text}'")
    print("\n" + "="*50)
    
    # Test the structured data extraction
    receipt_data = ocr._extract_receipt_data_optimized(text, 'maya')
    
    print("Structured data extraction results:")
    for key, value in receipt_data.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*50)
    print("Manual pattern testing:")
    
    # Test phone number patterns manually
    phone_patterns = [
        r'(\+639\d{9})',  # +639xxxxxxxxx format
        r'(\+?63\s*9\d{8})',  # +63 9xxxxxxxx with space
        r'(639\d{9})',  # 639xxxxxxxxx format (without +)
        r'(\+639\s*\d{9})',  # +639 xxxxxxxxx with space
    ]
    
    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        if matches:
            print(f"Phone pattern '{pattern}' found: {matches}")
    
    # Test reference ID patterns manually
    ref_patterns = [
        r'Reference\s*:?\s*ID\s*:?\s*([A-Z0-9\s@:\'éç©\.]{8,20})',
        r'Reference\s*:?\s*([A-Z0-9\s@:\'éç©\.]{8,20})',
        r'([A-Z0-9]{2,4}\s*[A-Z0-9]{2,4}\s*[A-Z0-9]{2,4})',
    ]
    
    for pattern in ref_patterns:
        matches = re.findall(pattern, text)
        if matches:
            print(f"Reference pattern '{pattern}' found: {matches}")

if __name__ == "__main__":
    test_structured_extraction()
