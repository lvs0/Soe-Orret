"""
Sampler Block Diffusion pour SOE-Orret.
Algorithme :
  1. Encoder le prompt
  2. Initialiser la réponse avec [MASK]*N
  3. Pour N étapes : passer au modèle bidirectionnel, fixer les tokens confiants
  4. Retourner la séquence finale
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import torch en mode différé / fallback gracefull (évite crash si torch absent)
_TORCH_AVAILABLE = True
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    F = None

from typing import List, Optional
import time

class BlockDiffusionSampler:
    def __init__(self, model_path: str, device: str = "auto",
                 n_steps: int = 16, block_size: int = 32,
                 confidence_threshold: float = 0.9):
        # Fallback si torch est absent
        if not _TORCH_AVAILABLE or torch is None:
            raise ImportError(
                "torch est requis pour BlockDiffusionSampler. "
                "Installez-le avec : pip install torch. "
                "Alternative : utilisez le mode Ollama/GGUF dans server.py."
            )
        self.n_steps    = n_steps
        self.block_size = block_size
        self.conf_thr   = confidence_threshold
        self.tokenizer  = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.mask_id    = self.tokenizer.mask_token_id or self.tokenizer.pad_token_id
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device, trust_remote_code=True)
        self.model.eval()
        print(f"[Orret dLLM] Chargé sur {device} | steps={n_steps} | block={block_size}")

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens=512,
                 temperature=0.7, top_p=0.9, verbose=False) -> str:
        t0 = time.time()
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        n_blocks   = (max_new_tokens + self.block_size - 1) // self.block_size
        generated  = []

        for _ in range(n_blocks):
            existing = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(self.device) \
                       if generated else prompt_ids[:, :0]
            prefix = torch.cat([prompt_ids, existing], dim=1)
            block  = self._generate_block(prefix, temperature, top_p, verbose)
            generated.extend(block)
            if self.tokenizer.eos_token_id in block:
                eos = block.index(self.tokenizer.eos_token_id)
                generated = generated[:len(generated) - len(block) + eos]
                break

        if verbose:
            t = time.time() - t0
            print(f"\n[{len(generated)} tokens en {t:.1f}s = {len(generated)/t:.1f} tok/s]")
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    @torch.no_grad()
    def _generate_block(self, prompt_ids, temperature, top_p, verbose) -> List[int]:
        mask_block = torch.full((1, self.block_size), self.mask_id,
                                dtype=torch.long, device=self.device)
        full_seq       = torch.cat([prompt_ids, mask_block], dim=1)
        mask_positions = list(range(prompt_ids.shape[1], full_seq.shape[1]))
        still_masked   = set(mask_positions)
        fixed_tokens   = {}

        for step in range(self.n_steps):
            if not still_masked:
                break
            outputs = self.model(full_seq)
            logits  = outputs.logits
            for pos in list(still_masked):
                probs     = F.softmax(logits[0, pos, :] / max(temperature, 1e-8), dim=-1)
                max_prob, best = probs.max(dim=-1)
                if max_prob.item() >= self.conf_thr or step == self.n_steps - 1:
                    if temperature > 0:
                        sp, si = probs.sort(descending=True)
                        cum = sp.cumsum(dim=-1)
                        sp[cum > top_p] = 0
                        sp /= sp.sum() + 1e-8
                        token_id = si[torch.multinomial(sp, 1)].item()
                    else:
                        token_id = best.item()
                    fixed_tokens[pos] = token_id
                    full_seq[0, pos]  = token_id
                    still_masked.discard(pos)
            if verbose:
                partial = self.tokenizer.decode(
                    [fixed_tokens.get(p, self.mask_id) for p in mask_positions],
                    skip_special_tokens=True)
                print(f"\rStep {step+1}/{self.n_steps} | {partial[:60]}", end="", flush=True)

        return [fixed_tokens.get(p, self.mask_id) for p in mask_positions]

    def generate_simple(self, prompt: str, **kwargs) -> str:
        """Interface simple pour utilisation dans les sous-modules."""
        return self.generate(prompt, **kwargs)
