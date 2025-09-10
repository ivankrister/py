#!/usr/bin/env python3

import os
import cv2
import numpy as np
import pytesseract
from ocr_processor import ReceiptOCR
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_maya2_detailed():
    """Detailed debugging of maya2.jpeg OCR processing"""
    
    # Initialize OCR processor
    ocr = ReceiptOCR()
    
    # Test file path
    test_file = "maya2.jpeg"
    
    if not os.path.exists(test_file):
        logger.error(f"File {test_file} not found!")
        return
    
    print(f"Debugging OCR processing for {test_file}")
    print("="*60)
    
    try:
        # Read the image file
        with open(test_file, 'rb') as f:
            image_data = f.read()
        
        # Convert to OpenCV image
        nparr = np.frombuffer(image_data, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        print(f"Original image shape: {original_image.shape}")
        
        # Save original for comparison
        cv2.imwrite("debug_maya2_original.png", original_image)
        print("Saved: debug_maya2_original.png")
        
        # Test each preprocessing method individually
        preprocessing_methods = [
            ("original", lambda img: img),
            ("adaptive_threshold_enhanced", ocr._maya_adaptive_threshold_enhanced),
            ("adaptive_threshold", ocr._adaptive_threshold_preprocessing),
            ("enhanced", ocr._enhanced_preprocessing),
            ("standard", ocr._standard_preprocessing),
            ("alternative", ocr._alternative_preprocessing)
        ]
        
        tesseract_configs = [
            '--psm 6 --oem 3',
            '--psm 6',
            '--psm 4',
            '--psm 3',
            '--psm 8',
            '--psm 13'
        ]
        
        best_result = ""
        best_confidence = 0
        best_method = ""
        best_config = ""
        
        for method_name, method_func in preprocessing_methods:
            print(f"\nTesting preprocessing method: {method_name}")
            print("-" * 40)
            
            if method_name == "original":
                # Convert to grayscale for original
                if len(original_image.shape) == 3:
                    processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                else:
                    processed_image = original_image.copy()
            else:
                processed_image = method_func(original_image)
            
            # Save processed image
            output_filename = f"debug_maya2_{method_name}.png"
            cv2.imwrite(output_filename, processed_image)
            print(f"Saved processed image: {output_filename}")
            print(f"Processed image shape: {processed_image.shape}")
            
            # Test each Tesseract config
            for config in tesseract_configs:
                try:
                    # Get text with confidence data
                    text_data = pytesseract.image_to_data(
                        processed_image, 
                        config=config, 
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Get full text
                    full_text = pytesseract.image_to_string(
                        processed_image, 
                        config=config
                    ).strip()
                    
                    # Calculate confidence
                    confidences = [int(conf) for conf in text_data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    print(f"  Config {config}: confidence={avg_confidence:.1f}%, text_length={len(full_text)}")
                    
                    if full_text:
                        print(f"    Text: '{full_text[:100]}{'...' if len(full_text) > 100 else ''}'")
                    
                    # Track best result
                    if avg_confidence > best_confidence and full_text:
                        best_result = full_text
                        best_confidence = avg_confidence
                        best_method = method_name
                        best_config = config
                        
                except Exception as e:
                    print(f"  Config {config}: ERROR - {str(e)}")
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        if best_result:
            print(f"Best result found:")
            print(f"  Method: {best_method}")
            print(f"  Config: {best_config}")
            print(f"  Confidence: {best_confidence:.1f}%")
            print(f"  Text: '{best_result}'")
        else:
            print("No readable text found with any method/config combination!")
            print("\nTroubleshooting suggestions:")
            print("1. Check if the image contains actual text")
            print("2. Try manual image enhancement")
            print("3. Check if the image is corrupted")
            print("4. Verify Tesseract installation")
        
        # Additional debug: try raw Tesseract on original
        print(f"\nRaw Tesseract test on original image:")
        try:
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            raw_text = pytesseract.image_to_string(gray)
            print(f"Raw text: '{raw_text.strip()}'")
        except Exception as e:
            print(f"Raw Tesseract failed: {e}")
            
    except Exception as e:
        logger.error(f"Debug failed: {str(e)}")
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    debug_maya2_detailed()
