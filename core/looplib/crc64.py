"""
CRC64 - Vérification d'intégrité pour fichiers .loop
"""
import binascii

def compute(data: bytes) -> int:
    """Calcule CRC64 de data"""
    return binascii.crc64(data) & 0xFFFFFFFFFFFFFFFF

def verify(data: bytes, expected: int) -> bool:
    """Vérifie que le CRC correspond"""
    return compute(data) == expected
