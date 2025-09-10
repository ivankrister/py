#!/usr/bin/env python3

import os
import sys
from ocr_processor import ReceiptOCR
import logging

# Configure logging to see detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_maya2_ocr():
    """Test OCR processing on maya2.jpeg"""
    
    # Initialize OCR processor
    ocr = ReceiptOCR()
    
    # Test file path
    test_file = "maya2.jpeg"
    
    if not os.path.exists(test_file):
        logger.error(f"File {test_file} not found!")
        return
    
    logger.info(f"Testing OCR on {test_file}")
    
    try:
        # Read the image file
        with open(test_file, 'rb') as f:
            image_data = f.read()
        
        logger.info(f"Image data size: {len(image_data)} bytes")
        
        # Process with OCR
        result = ocr.extract_text(image_data)
        
        print("\n" + "="*50)
        print("OCR RESULTS FOR maya2.jpeg")
        print("="*50)
        
        print(f"Extracted Text:")
        print(f"'{result.get('extracted_text', 'NO TEXT EXTRACTED')}'")
        print(f"\nConfidence: {result.get('confidence', 0)}%")
        print(f"Adjusted Confidence: {result.get('adjusted_confidence', 0)}%")
        print(f"Word Count: {result.get('word_count', 0)}")
        print(f"Preprocessing Method: {result.get('preprocessing_method', 'none')}")
        print(f"Tesseract Config: {result.get('tesseract_config', 'none')}")
        
        if result.get('receipt_data'):
            print(f"\nStructured Receipt Data:")
            for key, value in result['receipt_data'].items():
                print(f"  {key}: {value}")
        
        if result.get('error'):
            print(f"\nError: {result['error']}")
        
        print("\n" + "="*50)
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    test_maya2_ocr()
