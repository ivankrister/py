#!/usr/bin/env python3

import re

def analyze_text():
    text = """Received money from

P'1.00 m
+639 162907953 re]
v Completed Sep &, 2025, 01:08 pm

- from Sarah Jane

Reference ID 172F EDCS5 80BD

maya"""
    
    print("Character analysis:")
    for i, char in enumerate(text):
        if i > 60 and i < 120:  # Focus on the area with the date
            print(f"Index {i}: '{char}' (ord {ord(char)})")
    
    print("\nLooking for 'Completed' in text:")
    completed_pos = text.find('Completed')
    print(f"Position: {completed_pos}")
    
    if completed_pos >= 0:
        # Show context around 'Completed'
        start = max(0, completed_pos - 10)
        end = min(len(text), completed_pos + 30)
        context = text[start:end]
        print(f"Context: '{context}'")
        print("Character codes in context:")
        for i, char in enumerate(context):
            print(f"  {i}: '{char}' (ord {ord(char)})")
    
    # Try simpler patterns
    simple_patterns = [
        r'Completed',
        r'Sep',
        r'2025',
        r'Sep.*?2025',
        r'Completed.*?2025',
    ]
    
    print("\nSimple pattern tests:")
    for pattern in simple_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        print(f"'{pattern}': {matches}")

if __name__ == "__main__":
    analyze_text()
