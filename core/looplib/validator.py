"""
Validator - Vérification d'intégrité des fichiers .loop
"""
import struct

def validate_loop(path: str) -> list:
    """Valide un fichier .loop et retourne la liste des erreurs"""
    errors = []
    
    try:
        with open(path, 'rb') as f:
            # Header (9 bytes: 4 magic + 1 version + 4 meta_len)
            magic = f.read(4)
            if magic != b'LOOP':
                errors.append(f"Invalid magic: {magic}")
                return errors
                
            version, meta_len = struct.unpack('>BI', f.read(5))
            if version != 1:
                errors.append(f"Unsupported version: {version}")
                
            f.read(meta_len)  # Skip metadata
            
            # Entries (4 bytes CRC + 4 bytes length + data)
            entry_num = 0
            while True:
                crc_bytes = f.read(4)
                if not crc_bytes:
                    break
                entry_num += 1
                length = struct.unpack('>I', f.read(4))[0]
                f.read(length)  # Skip data
                
    except Exception as e:
        errors.append(f"Read error: {e}")
        
    return errors