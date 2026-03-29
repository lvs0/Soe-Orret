# ORRET Colab Training - Guide

## Upload to Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create new notebook
3. Upload `orret_finetune.ipynb`
4. Create folder `/content/datasets/raw/`
5. Upload all `.loop` files from `datasets/raw/`

## Alternative: Google Drive

1. Upload `orret-colab.zip` to Google Drive
2. In Colab: `from google.colab import drive`
3. `drive.mount('/content/drive')`
4. `!unzip /content/drive/MyDrive/orret-colab.zip -d /content/`

## Run Training

```python
# Cell 1: Install
!pip install -q unsloth transformers datasets peft accelerate

# Cell 2-6: Run the notebook cells
```

## After Training

1. Download GGUF file from Colab
2. Copy to `~/soe/models/gguf/`
3. Run: `ollama create orret -f Modelfile`
4. Test: `ollama run orret`

## Files Included

- `coding_fr.loop` - 10 coding examples
- `instructions_fr.loop` - 10 instruction examples  
- `linux_sysadmin.loop` - 3 Linux admin examples
- `science_ia_fr.loop` - 3 AI/science examples
- `wikipedia_fr.loop` - 3 Wikipedia examples

**Total: 29 training examples**
