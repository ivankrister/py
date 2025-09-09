#!/usr/bin/env python3
"""
Test enhanced OCR with GCash receipt support
"""

import sys
import os
sys.path.append('.')

from ocr_processor import ReceiptOCR
import json

def test_gcash_enhanced():
    """Test the enhanced OCR processor on gcash.jpeg"""
    
    if not os.path.exists("gcash.jpeg"):
        print("❌ gcash.jpeg not found!")
        return
    
    print("Testing enhanced OCR processor on gcash.jpeg")
    print("=" * 70)
    
    # Initialize the OCR processor
    ocr = ReceiptOCR()
    
    # Read the image file
    with open("gcash.jpeg", "rb") as f:
        image_data = f.read()
    
    # Process with enhanced OCR
    result = ocr.extract_text(image_data)
    
    print(f"✅ OCR Processing completed!")
    print(f"Confidence: {result['confidence']}%")
    print(f"Preprocessing method: {result.get('preprocessing_method', 'unknown')}")
    print(f"Word count: {result['word_count']}")
    
    print(f"\nRaw extracted text:")
    print("-" * 70)
    print(result['extracted_text'])
    print("-" * 70)
    
    # Display structured receipt data
    receipt_data = result.get('receipt_data', {})
    if receipt_data:
        print(f"\n📊 STRUCTURED GCASH RECEIPT DATA:")
        print("=" * 50)
        
        # Check each required field
        expected_fields = {
            'amount': '10.00',
            'phone_number': '+639296681405', 
            'reference_id': '9032469742237'
        }
        
        all_correct = True
        
        print(f"Receipt Type: {receipt_data.get('receipt_type', 'unknown')}")
        print()
        
        for field, expected in expected_fields.items():
            actual = receipt_data.get(field)
            print(f"{field.replace('_', ' ').title()}: {actual}")
            
            if field == 'amount' and actual == expected:
                print("  ✅ Amount correct!")
            elif field == 'phone_number' and actual == expected:
                print("  ✅ Phone number correct!")
            elif field == 'reference_id' and actual == expected:
                print("  ✅ Reference ID correct!")
            else:
                print(f"  ❌ Incorrect (expected: {expected})")
                all_correct = False
        
        # Show additional fields
        additional_fields = ['date', 'sender']
        print(f"\nAdditional extracted data:")
        for field in additional_fields:
            value = receipt_data.get(field)
            if value:
                print(f"{field.title()}: {value}")
        
        # Summary
        print(f"\n{'='*50}")
        if all_correct:
            print("🎉 SUCCESS: All required GCash fields extracted correctly!")
        else:
            print("⚠️  Some fields need improvement")
            
        # Show JSON format
        print(f"\n📋 JSON Response:")
        print(json.dumps(receipt_data, indent=2))
        
    else:
        print("❌ No structured receipt data extracted")
    
    return result

def test_maya_still_works():
    """Test that Maya receipt still works after GCash enhancements"""
    
    if not os.path.exists("maya.jpeg"):
        print("⚠️ maya.jpeg not found, skipping Maya compatibility test")
        return
    
    print(f"\n" + "=" * 70)
    print("Testing Maya receipt compatibility...")
    
    ocr = ReceiptOCR()
    
    with open("maya.jpeg", "rb") as f:
        image_data = f.read()
    
    result = ocr.extract_text(image_data)
    receipt_data = result.get('receipt_data', {})
    
    if receipt_data:
        print(f"✅ Maya receipt type: {receipt_data.get('receipt_type', 'unknown')}")
        print(f"✅ Maya amount: {receipt_data.get('amount')}")
        print(f"✅ Maya phone: {receipt_data.get('phone_number')}")
        print(f"✅ Maya reference: {receipt_data.get('reference_id')}")
    else:
        print("❌ Maya receipt processing failed")

if __name__ == "__main__":
    test_gcash_enhanced()
    test_maya_still_works()
