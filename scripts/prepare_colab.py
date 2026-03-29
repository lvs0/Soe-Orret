#!/usr/bin/env python3
"""Colab training launcher - run from local machine after .loop files ready"""
import os
import shutil
import zipfile

# 1. Ensure .loop files exist
raw_dir = os.path.expanduser("~/soe/datasets/raw")
loop_files = [f for f in os.listdir(raw_dir) if f.endswith(".loop")]

if not loop_files:
    print("ERROR: No .loop files in ~/soe/datasets/raw/")
    print("Run: cd ~/soe && python3 scripts/quick_ruche_test.py")
    exit(1)

print(f"Found: {loop_files}")

# 2. Create Colab upload zip
colab_dir = os.path.expanduser("~/soe/colab/upload")
os.makedirs(colab_dir, exist_ok=True)

for f in loop_files:
    src = os.path.join(raw_dir, f)
    dst = os.path.join(colab_dir, f)
    shutil.copy2(src, dst)

# 3. Copy notebook
shutil.copy2(
    os.path.expanduser("~/soe/colab/orret_finetune.ipynb"),
    os.path.join(colab_dir, "orret_finetune.ipynb")
)

# 4. Create zip
zip_path = os.path.expanduser("~/soe/colab/orret-colab.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir(colab_dir):
        zf.write(os.path.join(colab_dir, f), f)

print(f"\n✅ Created: {zip_path}")
print("\nNext steps:")
print("1. Upload orret-colab.zip to Google Drive")
print("2. Open Colab, mount Drive")
print("3. Extract: !unzip /content/orret-colab.zip")
print("4. Run: !python3 orret_finetune.ipynb")
print("\nOr just upload individual files to Colab /content/")
