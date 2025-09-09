#!/usr/bin/env python3
"""
Test the optimized OCR API with our receipt images
"""

import requests
import json
from pathlib import Path

def test_optimized_api():
    """Test the API with Maya and GCash receipts"""
    
    # Start the server first (you need to run this in another terminal)
    api_url = "http://localhost:8000"
    
    # Test images
    test_images = [
        ('maya.jpeg', 'Maya receipt'),
        ('gcash.jpeg', 'GCash receipt')
    ]
    
    print("🧪 Testing Optimized OCR API")
    print("="*50)
    
    # Check if server is running
    try:
        health_response = requests.get(f"{api_url}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ API server is running")
        else:
            print("❌ API server health check failed")
            return
    except requests.exceptions.RequestException:
        print("❌ API server is not running. Please start it with: python main.py")
        return
    
    for image_file, description in test_images:
        if not Path(image_file).exists():
            print(f"❌ {image_file} not found, skipping...")
            continue
            
        print(f"\n📄 Testing {description} ({image_file})")
        print("-" * 40)
        
        try:
            # Read image file
            with open(image_file, 'rb') as f:
                files = {'file': (image_file, f, 'image/jpeg')}
                
                # Make API request
                response = requests.post(
                    f"{api_url}/extract-text",
                    files=files,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ Status: Success")
                print(f"📊 Confidence: {result['confidence']:.1f}%", end="")
                if result.get('adjusted_confidence'):
                    print(f" (Adjusted: {result['adjusted_confidence']:.1f}%)")
                else:
                    print()
                
                print(f"🔧 Method: {result.get('preprocessing_method', 'unknown')}")
                if result.get('tesseract_config'):
                    print(f"⚙️  Config: {result['tesseract_config']}")
                
                print(f"📝 Words: {result['word_count']}")
                
                # Show receipt data
                if result.get('receipt_data'):
                    receipt = result['receipt_data']
                    print(f"\n💰 Extracted Data:")
                    print(f"   Type: {receipt.get('receipt_type', 'unknown')}")
                    print(f"   Amount: {receipt.get('amount', 'Not found')}")
                    print(f"   Phone: {receipt.get('phone_number', 'Not found')}")
                    print(f"   Reference: {receipt.get('reference_id', 'Not found')}")
                    print(f"   Date: {receipt.get('date', 'Not found')}")
                    print(f"   Sender: {receipt.get('sender', 'Not found')}")
                    
                    # Check accuracy for known expected values
                    expected_values = {
                        'maya.jpeg': {
                            'amount': '10.00',
                            'phone_number': '+639772478589',
                            'reference_id': 'EB8C C4C5 C67B'
                        },
                        'gcash.jpeg': {
                            'amount': '10.00',
                            'phone_number': '+639296681405',
                            'reference_id': '9032469742237'
                        }
                    }
                    
                    if image_file in expected_values:
                        expected = expected_values[image_file]
                        print(f"\n🎯 Accuracy Check:")
                        
                        matches = 0
                        total = 3
                        
                        if receipt.get('amount') == expected['amount']:
                            print(f"   ✅ Amount: {receipt['amount']} (matches)")
                            matches += 1
                        else:
                            print(f"   ❌ Amount: {receipt.get('amount')} (expected: {expected['amount']})")
                        
                        if receipt.get('phone_number') == expected['phone_number']:
                            print(f"   ✅ Phone: {receipt['phone_number']} (matches)")
                            matches += 1
                        else:
                            print(f"   ❌ Phone: {receipt.get('phone_number')} (expected: {expected['phone_number']})")
                        
                        if receipt.get('reference_id') == expected['reference_id']:
                            print(f"   ✅ Reference: {receipt['reference_id']} (matches)")
                            matches += 1
                        else:
                            print(f"   ❌ Reference: {receipt.get('reference_id')} (expected: {expected['reference_id']})")
                        
                        accuracy = (matches / total) * 100
                        print(f"\n📈 Overall Accuracy: {accuracy:.1f}% ({matches}/{total})")
                        
                        if accuracy == 100:
                            print("🎉 PERFECT EXTRACTION! 🎉")
                        elif accuracy >= 80:
                            print("✅ Excellent extraction!")
                        elif accuracy >= 60:
                            print("⚠️  Good extraction with minor issues")
                        else:
                            print("❌ Needs improvement")
                
                # Show sample of extracted text
                sample_text = result['extracted_text'][:200] + "..." if len(result['extracted_text']) > 200 else result['extracted_text']
                print(f"\n📄 Sample Text:")
                print(f'   "{sample_text}"')
                
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"   {response.text}")
                
        except Exception as e:
            print(f"❌ Error testing {image_file}: {str(e)}")
    
    print(f"\n🏁 Testing completed!")

if __name__ == "__main__":
    test_optimized_api()
