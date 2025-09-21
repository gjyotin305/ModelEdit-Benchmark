#!/usr/bin/env python3
"""
Script to add history_indexs field to JSONL file.
Each entry gets a cumulative list of all previous entry indices.
"""

import json
import argparse
from pathlib import Path

def add_history_indexs(input_file, output_file):
    """
    Add history_indexs field to each JSONL entry and output as JSON array.
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSON file
    """
    history_index = 0
    all_data = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
                
            # Parse the JSON line
            data = json.loads(line)
            
            # Add history_indexs field
            # First entry gets empty list, subsequent entries get [0], [0,1], [0,1,2], etc.
            if history_index == 0:
                data['history_indexs'] = []
            else:
                data['history_indexs'] = list(range(history_index))
            
            all_data.append(data)
            history_index += 1
    
    # Write all data as a JSON array
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(all_data, outfile, ensure_ascii=False, indent=2)
    
    print(f"Successfully processed {history_index} entries")
    print(f"Output written to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Add history_indexs to JSONL file and output as JSON array')
    parser.add_argument('input_file', help='Input JSONL file path')
    parser.add_argument('output_file', help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' does not exist")
        return 1
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    add_history_indexs(args.input_file, args.output_file)
    return 0

if __name__ == "__main__":
    exit(main())
