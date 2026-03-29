"""
looplib - Format .loop pour SOE
"""
from .writer import LoopWriter
from .reader import LoopReader
from .validator import validate_loop
from .converter import jsonl_to_loop, csv_to_loop
import binascii

def crc64_compute(data: bytes) -> int:
    """Calcule CRC64 de data"""
    return binascii.crc64(data) & 0xFFFFFFFFFFFFFFFF

__all__ = ['LoopWriter', 'LoopReader', 'validate_loop', 'jsonl_to_loop', 'csv_to_loop', 'crc64_compute']