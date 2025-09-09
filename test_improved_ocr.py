#!/usr/bin/env python3
"""
Simple test of the improved OCR processor directly
"""

import sys
import os
sys.path.append('.')

from ocr_processor import ReceiptOCR

def test_improved_ocr():
    """Test the improved OCR processor on maya.jpeg"""
    
    if not os.path.exists("maya.jpeg"):
        print("❌ maya.jpeg not found!")
        return
    
    print("Testing improved OCR processor on maya.jpeg")
    print("=" * 60)
    
    # Initialize the OCR processor
    ocr = ReceiptOCR()
    
    # Read the image file
    with open("maya.jpeg", "rb") as f:
        image_data = f.read()
    
    # Process with improved OCR
    result = ocr.extract_text(image_data)
    
    print(f"✅ OCR Processing completed!")
    print(f"Confidence: {result['confidence']}%")
    print(f"Preprocessing method: {result.get('preprocessing_method', 'unknown')}")
    print(f"Word count: {result['word_count']}")
    print(f"Image dimensions: {result.get('image_dimensions', 'N/A')}")
    
    print(f"\nExtracted text:")
    print("-" * 60)
    print(result['extracted_text'])
    print("-" * 60)
    
    # Check for 10.00
    if "10.00" in result['extracted_text']:
        print("🎉 SUCCESS: 10.00 DETECTED!")
    else:
        print("❌ 10.00 not detected")
        
        # Look for any numbers
        import re
        numbers = re.findall(r'\d+\.?\d*', result['extracted_text'])
        print(f"Numbers found: {numbers}")
    
    # Show structured data summary
    structured = result.get('structured_data', {})
    if structured:
        print(f"\nStructured data:")
        print(f"  Total words: {structured.get('total_words', 0)}")
        print(f"  Total lines: {structured.get('total_lines', 0)}")
        
        # Show high-confidence words
        words = structured.get('words', [])
        high_conf_words = [w for w in words if w.get('confidence', 0) > 70]
        if high_conf_words:
            print(f"  High confidence words ({len(high_conf_words)}):")
            for word in high_conf_words[:10]:  # Show first 10
                print(f"    '{word['text']}' ({word['confidence']}%)")

if __name__ == "__main__":
    test_improved_ocr()
