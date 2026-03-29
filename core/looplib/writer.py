"""
LoopWriter - Écriture de fichiers .loop
Format: binary sequential format avec header et CRC32
"""
import struct
import json
import zlib

def crc32_compute(data: bytes) -> int:
    """Calcule CRC32 de data"""
    return zlib.crc32(data) & 0xFFFFFFFF

class LoopWriter:
    def __init__(self, path: str):
        self.path = path
        self.file = open(path, 'wb')
        self.entries = 0
        
    def write_header(self, metadata: dict):
        """Écrit le header du fichier .loop"""
        magic = b'LOOP'
        version = 1
        meta_json = json.dumps(metadata).encode('utf-8')
        # Use big-endian to avoid padding
        header = struct.pack('>4sBI', magic, version, len(meta_json))
        self.file.write(header)
        self.file.write(meta_json)
        
    def write_entry(self, data: dict):
        """Écrit une entrée dans le fichier .loop"""
        entry_json = json.dumps(data, ensure_ascii=False).encode('utf-8')
        length = len(entry_json)
        crc = crc32_compute(entry_json)
        
        self.file.write(struct.pack('>I', crc))
        self.file.write(struct.pack('>I', length))
        self.file.write(entry_json)
        self.entries += 1
        
    def close(self):
        self.file.close()
        return self.entries