#!/usr/bin/env python3
"""
Quick OCR Test Runner
Run this script to quickly test OCR methods on your Maya and GCash receipts
"""

import subprocess
import sys
from pathlib import Path
import json

def check_requirements():
    """Check if required packages are installed"""
    try:
        import cv2
        import pytesseract
        import PIL
        import numpy
        print("✅ All required packages are available")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_images():
    """Check if image files exist"""
    images = ['maya.jpeg', 'gcash.jpeg']
    found = []
    
    for img in images:
        if Path(img).exists():
            print(f"✅ Found: {img}")
            found.append(img)
        else:
            print(f"❌ Missing: {img}")
    
    return found

def run_analysis():
    """Run the comprehensive OCR analysis"""
    print("\n🚀 Starting comprehensive OCR analysis...")
    print("This will test multiple preprocessing methods and Tesseract configurations")
    print("Expected results:")
    print("  Maya: amount='10.00', phone='+639772478589', ref='EB8C C4C5 C67B'")
    print("  GCash: amount='10.00', phone='+639296681405', ref='9032469742237'")
    print("\n" + "="*60)
    
    try:
        from comprehensive_ocr_analysis import main
        main()
        
        # Load and display key results
        if Path('comprehensive_ocr_analysis_results.json').exists():
            print("\n📊 QUICK SUMMARY:")
            with open('comprehensive_ocr_analysis_results.json', 'r') as f:
                results = json.load(f)
            
            for filename, result in results['individual_results'].items():
                if 'error' in result:
                    continue
                    
                summary = result['summary']
                best = summary['best_balance']
                
                print(f"\n{filename}:")
                print(f"  Best method: {best['method']} + {best['config']}")
                print(f"  Accuracy: {best['accuracy']:.1%}")
                print(f"  Speed: {best['time']:.3f}s")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False
    
    return True

def main():
    print("🔍 OCR Analysis Tool")
    print("="*40)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check images
    found_images = check_images()
    if not found_images:
        print("\n❌ No image files found!")
        print("Please ensure maya.jpeg and gcash.jpeg are in the current directory")
        return
    
    print(f"\n📸 Found {len(found_images)} image(s) to analyze")
    
    # Ask user if they want to proceed
    response = input("\nProceed with analysis? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Analysis cancelled.")
        return
    
    # Run analysis
    success = run_analysis()
    
    if success:
        print("\n✅ Analysis completed successfully!")
        print("📁 Detailed results saved to: comprehensive_ocr_analysis_results.json")
    else:
        print("\n❌ Analysis failed. Check the error messages above.")

if __name__ == "__main__":
    main()
