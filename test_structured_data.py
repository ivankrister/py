#!/usr/bin/env python3
"""
Test structured receipt data extraction from maya.jpeg
"""

import sys
import os
sys.path.append('.')

from ocr_processor import ReceiptOCR
import json

def test_structured_extraction():
    """Test structured receipt data extraction on maya.jpeg"""
    
    if not os.path.exists("maya.jpeg"):
        print("❌ maya.jpeg not found!")
        return
    
    print("Testing structured receipt data extraction on maya.jpeg")
    print("=" * 70)
    
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
    
    print(f"\nRaw extracted text:")
    print("-" * 50)
    print(result['extracted_text'])
    print("-" * 50)
    
    # Check for 10.00
    if "10.00" in result['extracted_text']:
        print("✅ Raw text contains 10.00")
    else:
        print("❌ Raw text doesn't contain 10.00")
    
    # Display structured receipt data
    receipt_data = result.get('receipt_data', {})
    if receipt_data:
        print(f"\n📊 STRUCTURED RECEIPT DATA:")
        print("=" * 50)
        
        # Check each required field
        expected_fields = {
            'amount': '10.00',
            'phone_number': '+639772478589', 
            'reference_id': 'EB8CC4C5C67B'
        }
        
        all_correct = True
        
        for field, expected in expected_fields.items():
            actual = receipt_data.get(field)
            print(f"{field.upper()}: {actual}")
            
            if field == 'amount' and actual == expected:
                print("  ✅ Amount correct!")
            elif field == 'phone_number' and actual == expected:
                print("  ✅ Phone number correct!")
            elif field == 'reference_id':
                # Be flexible with reference ID format
                if actual and expected.replace(' ', '') in actual.replace(' ', ''):
                    print("  ✅ Reference ID correct!")
                else:
                    print(f"  ❌ Reference ID incorrect (expected: {expected})")
                    all_correct = False
            else:
                print(f"  ❌ Incorrect (expected: {expected})")
                all_correct = False
        
        # Show additional fields
        additional_fields = ['date', 'sender']
        print(f"\nAdditional extracted data:")
        for field in additional_fields:
            value = receipt_data.get(field)
            if value:
                print(f"{field.upper()}: {value}")
        
        # Summary
        print(f"\n{'='*50}")
        if all_correct:
            print("🎉 SUCCESS: All required fields extracted correctly!")
        else:
            print("⚠️  Some fields need improvement")
            
        # Show JSON format
        print(f"\n📋 JSON Response:")
        print(json.dumps(receipt_data, indent=2))
        
    else:
        print("❌ No structured receipt data extracted")
    
    return result

if __name__ == "__main__":
    test_structured_extraction()
