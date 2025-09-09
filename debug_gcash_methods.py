import cv2
import numpy as np
from ocr_processor import ReceiptOCR
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_gcash_methods():
    """Debug specific method outputs for GCash"""
    processor = ReceiptOCR()
    
    # Load and check each method individually
    image_path = "gcash.jpeg"
    
    methods_to_test = ['standard', 'alternative']
    
    for method_name in methods_to_test:
        print(f"\n======== {method_name.upper()} METHOD ========")
        
        # Get the specific method result
        if method_name == 'standard':
            processed = processor._standard_preprocessing(cv2.imread(image_path))
        elif method_name == 'alternative':
            processed = processor._alternative_preprocessing(cv2.imread(image_path))
        
        # Extract text
        import pytesseract
        text = pytesseract.image_to_string(processed, config='--psm 6').strip()
        
        print(f"Raw text:\n{text}\n")
        print(f"Contains '+639296681405': {'+639296681405' in text}")
        print(f"Contains ',+639296681405': {',+639296681405' in text}")
        print(f"Contains '.+639296681405': {'.+639296681405' in text}")
        print(f"Contains '639296681405': {'639296681405' in text}")
        
        # Look for the phone number pattern
        import re
        phone_patterns = re.findall(r'[+]?63\d{9,10}', text)
        print(f"Phone patterns found: {phone_patterns}")
        
        print("-" * 50)

if __name__ == "__main__":
    debug_gcash_methods()
