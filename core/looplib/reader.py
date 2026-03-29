"""
LoopReader - Lecture de fichiers .loop avec streaming
"""
import struct
import json
import zlib

def crc32_compute(data: bytes) -> int:
    """Calcule CRC32 de data"""
    return zlib.crc32(data) & 0xFFFFFFFF

class LoopReader:
    def __init__(self, path: str):
        self.path = path
        self.file = open(path, 'rb')
        self.entries = 0
        self._read_header()
        
    def _read_header(self):
        magic = self.file.read(4)
        if magic != b'LOOP':
            raise ValueError(f"Invalid magic: {magic}")
        version, meta_len = struct.unpack('>BI', self.file.read(5))
        self.metadata = json.loads(self.file.read(meta_len))
        
    def read_entry(self):
        """Lit une entrée, retourne dict ou None si EOF"""
        crc_bytes = self.file.read(4)
        if not crc_bytes:
            return None
        crc = struct.unpack('>I', crc_bytes)[0]
        length = struct.unpack('>I', self.file.read(4))[0]
        data = self.file.read(length)
        
        # Verify CRC
        if crc32_compute(data) != crc:
            raise ValueError(f"CRC mismatch at entry {self.entries}")
            
        self.entries += 1
        return json.loads(data)
        
    def __iter__(self):
        return self
        
    def __next__(self):
        entry = self.read_entry()
        if entry is None:
            raise StopIteration
        return entry
        
    def close(self):
        self.file.close()