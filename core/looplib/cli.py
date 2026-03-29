"""
CLI - Outil en ligne de commande `loop`
Usage: loop [read|write|validate|convert] <args>
"""
import sys
import argparse
from pathlib import Path

def cmd_read(args):
    from reader import LoopReader
    reader = LoopReader(args.file)
    print(f"Metadata: {reader.metadata}")
    for entry in reader:
        print(entry)
    reader.close()

def cmd_validate(args):
    from validator import validate_loop
    errors = validate_loop(args.file)
    if errors:
        print(f"Errors: {errors}")
        return 1
    print("OK")
    return 0

def cmd_convert(args):
    from converter import jsonl_to_loop, csv_to_loop
    
    input_ext = Path(args.input).suffix
    if input_ext == '.jsonl':
        count = jsonl_to_loop(args.input, args.output)
    elif input_ext == '.csv':
        count = csv_to_loop(args.input, args.output)
    else:
        print(f"Unsupported format: {input_ext}")
        return 1
        
    print(f"Converted {count} entries")

def main():
    parser = argparse.ArgumentParser(prog='loop')
    subparsers = parser.add_subparsers()
    
    subparsers.add_parser('read', help='Read .loop file').add_argument('file')
    subparsers.add_parser('validate', help='Validate .loop file').add_argument('file')
    sub = subparsers.add_parser('convert', help='Convert JSONL/CSV to .loop')
    sub.add_argument('input')
    sub.add_argument('output')
    
    args = parser.parse_args()
    if not hasattr(args, 'file') and not hasattr(args, 'input'):
        parser.print_help()
        return 1
        
    if hasattr(args, 'file'):
        return cmd_validate(args)
    return cmd_convert(args)

if __name__ == '__main__':
    sys.exit(main())
