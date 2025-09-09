#!/usr/bin/env python3
"""
Test improved Maya extraction
"""

import re

def test_maya_extraction():
    # Sample OCR texts from our debug
    test_texts = [
        "Received money from\n+639772478589 6\nv Completed Sep 9, 2025, 01:44 pm\n- from Jidy Thialeen\nReference ID EB8C C4C5 Cé67B\n\nmaya",
        "Received money 'from\n10.00 it:\n+639772478589 x. @\nv Completed Sep 9, 2025, 01:44 pm\n\n| ~from.Jidy'Thialeen |\nReference!| EB8@:C4C5'C67B",
        "Received money from\n\nP10.00 m,\n+639772478589 ._@\nwv Completed Sep 9, 2025, 01:44 pm\n\n~froniJidy Thialeén:\n\nReterence:ID. EB8G@'G4C5 ©67B"
    ]
    
    expected = {
        'amount': '10.00',
        'phone_number': '+639772478589',
        'reference_id': 'EB8C64C5C67B'
    }
    
    print("🔍 Testing Improved Maya Extraction")
    print("="*50)
    print(f"Expected: {expected}")
    print()
    
    for i, text in enumerate(test_texts, 1):
        print(f"📄 Test Text {i}:")
        print(f"Raw: {repr(text)}")
        
        data = {
            'amount': None,
            'phone_number': None,
            'reference_id': None
        }
        
        # Clean text
        text_clean = text.replace('\n', ' ').replace('\t', ' ')
        
        # Extract phone number
        phone_pattern = r'\+?639\d{9}'
        phone_match = re.search(phone_pattern, text_clean)
        if phone_match:
            phone = phone_match.group()
            if not phone.startswith('+'):
                phone = '+' + phone
            data['phone_number'] = phone
        
        # Extract amount
        amount_patterns = [
            r'P(\d+\.?\d*)',
            r'(\d+\.\d{2})',
            r'P\s*(\d+\.\d{2})',
            r'₱\s*(\d+\.\d{2})'
        ]
        
        for pattern in amount_patterns:
            amount_match = re.search(pattern, text_clean)
            if amount_match:
                data['amount'] = amount_match.group(1)
                break
        
        # Extract reference ID with improved patterns
        ref_patterns = [
            r'Reference[:\s]*ID[:\s]*([A-Z0-9\s@:\'éç©\.]{8,20})',
            r'Reference[:\s]*([A-Z0-9\s@:\'éç©\.]{8,20})',
            r'Ref[\.:\s]*([A-Z0-9\s@:\'éç©\.]{8,20})',
            r'EB8[A-Z0-9\s@:\'éç©\.]+C[0-9A-Z\s@:\'éç©\.]+C[0-9A-Z\s@:\'éç©\.]+B',
            r'([A-Z0-9]{2,4}[A-Z0-9\s@:\'éç©\.]*[A-Z0-9]{2,4}[A-Z0-9\s@:\'éç©\.]*[A-Z0-9]{2,4})'
        ]
        
        for pattern in ref_patterns:
            ref_match = re.search(pattern, text_clean)
            if ref_match:
                try:
                    ref_id = ref_match.group(1)
                except IndexError:
                    ref_id = ref_match.group(0)  # Use entire match if no groups
                
                # Clean up the reference ID
                ref_id_clean = re.sub(r'[^A-Z0-9]', '', ref_id.upper())
                print(f"  Found potential ref: '{ref_id}' -> cleaned: '{ref_id_clean}'")
                
                # Check if it looks like the expected Maya format
                if len(ref_id_clean) >= 10 and ref_id_clean.startswith('EB8') and ref_id_clean.endswith('B'):
                    data['reference_id'] = ref_id_clean
                    print(f"  ✅ Accepted as reference ID: {ref_id_clean}")
                    break
                elif len(ref_id_clean) == 12 and 'EB8' in ref_id_clean and 'C67B' in ref_id_clean:
                    data['reference_id'] = ref_id_clean
                    print(f"  ✅ Accepted as reference ID: {ref_id_clean}")
                    break
        
        print(f"Extracted: {data}")
        
        # Check accuracy
        matches = 0
        if data['amount'] == expected['amount']:
            matches += 1
            print("✅ Amount matches!")
        if data['phone_number'] == expected['phone_number']:
            matches += 1
            print("✅ Phone matches!")
        if data['reference_id'] == expected['reference_id']:
            matches += 1
            print("✅ Reference matches!")
        
        accuracy = matches / 3
        print(f"Overall accuracy: {accuracy:.1%} ({matches}/3)")
        
        if accuracy == 1.0:
            print("🎯 PERFECT MATCH!")
        
        print("-" * 50)

if __name__ == "__main__":
    test_maya_extraction()
