#!/usr/bin/env python3
"""Download Orret model from Colab after training"""
import os
import sys

def download_latest_model():
    """Download the latest trained Orret model"""
    gguf_dir = "/content/orret-v01-gguf"
    local_dir = os.path.expanduser("~/soe/models/gguf")
    
    # Find GGUF file
    gguf_files = [f for f in os.listdir(gguf_dir) if f.endswith(".gguf")]
    
    if not gguf_files:
        print("ERROR: No GGUF file found!")
        print(f"Expected in: {gguf_dir}")
        return False
    
    gguf_file = gguf_files[0]
    local_path = os.path.join(local_dir, "orret-latest-Q4_K_M.gguf")
    
    # Copy
    print(f"Copying {gguf_file} to {local_path}...")
    os.system(f"cp {os.path.join(gguf_dir, gguf_file)} {local_path}")
    
    # Create Modelfile
    modelfile = f"""FROM ./{gguf_file}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER num_thread 4

SYSTEM \"\"\"Tu es Orret, le premier modèle de SOE.
Développé par Lévy, open source, local-first.
Tu es spécialisé en code Python/Bash, Linux, IA locale.
Tu réponds toujours en français sauf si demandé autrement.
Tu es précis, direct, sans censure inutile.\"\"\"
"""
    
    with open(os.path.join(local_dir, "Modelfile"), "w") as f:
        f.write(modelfile)
    
    # Register with Ollama
    print("Registering with Ollama...")
    os.system(f"cd {local_dir} && ollama create orret -f Modelfile")
    
    print(f"✅ Orret installed! Test with: ollama run orret")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Use provided path
        gguf_dir = sys.argv[1]
    else:
        gguf_dir = "/content/orret-v01-gguf"
    
    download_latest_model()
