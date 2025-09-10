#!/usr/bin/env python3

import re

def test_date_patterns():
    text = """Received money from

P'1.00 m
+639 162907953 re]
v Completed Sep &, 2025, 01:08 pm

- from Sarah Jane

Reference ID 172F EDCS5 80BD

maya"""
    
    print("Testing date patterns:")
    print(f"Text: '{text}'")
    print()
    
    date_patterns = [
        r'Completed\s+([A-Za-z]+\s*\&?\d{1,2},?\s*\d{4})',  # "Completed Sep &, 2025"
        r'v\s+Completed\s+([A-Za-z]+\s*\&?\d{1,2},?\s*\d{4})',  # "v Completed Sep &, 2025"
        r'(?:Completed|completed)\s+([A-Za-z]{3}\s*\&?\d{1,2},?\s*\d{4})',  # More specific month pattern
        r'([A-Za-z]{3}\s*\&?\d{1,2},?\s*\d{4})(?=\s*[,\s]*\d{2}:\d{2})',  # Month pattern before time
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # 9/9/2025 or 9-9-2025
        r'(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})',  # 2025/9/9
    ]
    
    for i, pattern in enumerate(date_patterns):
        matches = re.findall(pattern, text, re.IGNORECASE)
        print(f"Pattern {i+1}: {pattern}")
        print(f"  Matches: {matches}")
        print()

if __name__ == "__main__":
    test_date_patterns()
