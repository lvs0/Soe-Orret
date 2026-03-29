"""
Converter - Convertit JSONL/CSV/HF datasets vers format .loop
"""
import json
import csv
from pathlib import Path

def jsonl_to_loop(input_path: str, output_path: str, metadata: dict = None):
    """Convertit JSONL vers .loop"""
    from .writer import LoopWriter
    
    writer = LoopWriter(output_path)
    writer.write_header(metadata or {'source': input_path})
    
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            writer.write_entry(data)
            
    count = writer.close()
    return count

def csv_to_loop(input_path: str, output_path: str, metadata: dict = None):
    """Convertit CSV vers .loop"""
    from .writer import LoopWriter
    
    writer = LoopWriter(output_path)
    writer.write_header(metadata or {'source': input_path})
    
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            writer.write_entry(row)
            
    count = writer.close()
    return count
