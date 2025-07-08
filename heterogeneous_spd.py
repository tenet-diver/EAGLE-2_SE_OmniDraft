# heterogeneous_spd.py
# ------------------------------------------------------------
# Speculative decoding with   Tiny  →  Medium  →  Large
#   • cross-tokenizer translation      (OmniDraft-lite)
#   • dynamic tree drafting            (EAGLE-2 core)
#   • mixture verification             (Speculative Ensemble)
# ------------------------------------------------------------
import math, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from typing      import List, Tuple

# ------------------------------------------------------------
# 0.  LOAD MODELS (edit to taste)
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Memory management utilities
def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_model_efficiently(model_id: str, device: str = "cuda", offload_to_cpu: bool = False):
    """Load model with memory management"""
    print(f"Loading model: {model_id}")
    clear_gpu_memory()
    
    # Load with lower precision to save memory
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use half precision
        device_map="auto" if not offload_to_cpu else "cpu",
        low_cpu_mem_usage=True
    )
    
    if not offload_to_cpu and device == "cuda":
        model = model.to(device)
    
    return model.eval()

# Load tokenizers first (lightweight)
print("Loading tokenizers...")
TINY_ID   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LARGE_ID  = "Qwen/Qwen3-0.6B"

tiny_tok  = AutoTokenizer.from_pretrained(TINY_ID)
large_tok = AutoTokenizer.from_pretrained(LARGE_ID)

# Add padding tokens if missing
if tiny_tok.pad_token is None:
    tiny_tok.pad_token = tiny_tok.eos_token
if large_tok.pad_token is None:
    large_tok.pad_token = large_tok.eos_token

# Load models with memory management
print("Loading tiny model...")
tiny_lm = load_model_efficiently(TINY_ID, DEVICE)

print("Loading large model...")
# Try to load large model, fallback to CPU if OOM
try:
    large_lm = load_model_efficiently(LARGE_ID, DEVICE)
    print("Large model loaded on GPU")
except torch.cuda.OutOfMemoryError:
    print("GPU OOM detected, loading large model on CPU")
    clear_gpu_memory()
    large_lm = load_model_efficiently(LARGE_ID, "cpu", offload_to_cpu=True)
    print("Large model loaded on CPU")

# Optional: medium model (4-bit quantised large) would fit here
# ------------------------------------------------------------
# 1.  OMNIDRAFT-LITE TRANSLATOR
# ------------------------------------------------------------
class NGramTranslator:
    """
    Maps drafter subtokens (tiny_tok) → one target token (large_tok)
    by string concatenation.  Caches discovered merges.
    """
    def __init__(self,
                 draft_tokenizer: AutoTokenizer,
                 tgt_tokenizer:   AutoTokenizer):
        self.dtok   = draft_tokenizer
        self.ttok   = tgt_tokenizer
        self.cache  = {}                 # str(text) -> List[int] tgt_ids
        self._seed_common_tokens()

    def _seed_common_tokens(self):
        # if token strings coincide, store 1-to-1 mapping
        dtok_strings = {self.dtok.convert_ids_to_tokens(i): i
                        for i in range(self.dtok.vocab_size)}
        for tid in range(self.ttok.vocab_size):
            s = self.ttok.convert_ids_to_tokens(tid)
            if s in dtok_strings:
                self.cache[s] = [tid]

    def merge_subtokens(self,
                        draft_ids: List[int]) -> Tuple[List[int], List[slice]]:
        """
        Greedy left-to-right merge of draft_ids → target_ids.
        Returns:
            tgt_ids    – merged ids (target vocab)
            spans      – list of slices: which draft indices compose each tgt_id
        """
        text = "".join(self.dtok.convert_ids_to_tokens(i) for i in draft_ids)
        # fast path: all at once
        if text in self.cache:
            return self.cache[text], [slice(0, len(draft_ids))]
        tgt_ids, spans = [], []
        start = 0
        while start < len(draft_ids):
            acc, end = "", start
            found = None
            while end < len(draft_ids):
                acc += self.dtok.convert_ids_to_tokens(draft_ids[end])
                if acc in self.cache:
                    found = self.cache[acc]
                    found_slice = slice(start, end+1)
                end += 1
                if len(acc) > 50: break  # safety stop
            if found is None:
                # fall back to byte-level decomposition
                bytestr = acc.encode("utf-8")
                found = self.ttok(bytestr, add_special_tokens=False)["input_ids"]
                found_slice = slice(start, end)
            tgt_ids.extend(found)
            spans.extend([found_slice]*len(found))
            start = end
        return tgt_ids, spans

    # log-prob merge helper --------------------------------------------------
    def merge_logits(self,
                     draft_logits: torch.Tensor,
                     spans: List[slice]) -> torch.Tensor:
        """
        draft_logits: [len(draft_ids), vocab_draft]
        Returns merged logits   [len(tgt_ids), vocab_tgt]
        (simple sum of log-probs across composing draft subtokens)
        """
        logp = draft_logits.log_softmax(-1)
        merged_lp = []
        for sp in spans:
            lp_sum = logp[sp, :].logsumexp(0)  # product in prob-space
            merged_lp.append(lp_sum)
        return torch.stack(merged_lp, dim=0)

translator = NGramTranslator(tiny_tok, large_tok)

# ------------------------------------------------------------
# 2.  EAGLE-2 DYNAMIC TREE  (minimal version)
# ------------------------------------------------------------
def build_tree(draft_lp: torch.Tensor,
               max_depth: int = 8,
               conf_thresh: float = 0.2) -> List[int]:
    """
    draft_lp:   [L, vocab] log-probs for each speculative position.
    Returns the *draft_prefix* we will propose this round (token ids).
    Heuristic: extend while top-prob ≥ conf_thresh, else stop.
    """
    prefix = []
    for lp in draft_lp:
        top_prob, top_id = lp.softmax(-1).max(dim=-1)
        if top_prob < conf_thresh or len(prefix) >= max_depth:
            break
        prefix.append(int(top_id))
    return prefix

# ------------------------------------------------------------
# 3.  SPECULATIVE-ENSEMBLE VERIFIER
# ------------------------------------------------------------
def spec_ensemble_verify(prefix_ids:  List[int],
                         draft_lp:    torch.Tensor,
                         large_lp:    torch.Tensor,
                         alpha: float = 0.15) -> int:
    """
    Returns length of accepted prefix (0..len(prefix))
    """
    # mixture logits then probs
    mix_lp = torch.logaddexp(math.log1p(-alpha) + large_lp,
                             math.log(alpha)    + draft_lp)
    draft_probs = draft_lp.softmax(-1)
    mix_probs   = mix_lp.softmax(-1)

    accepted = 0
    for i, tok in enumerate(prefix_ids):
        ratio = (mix_probs[i, tok] / draft_probs[i, tok]).item()
        if ratio >= np.random.rand():
            accepted += 1
        else:
            break
    return accepted, mix_lp

# ------------------------------------------------------------
# 4.  MAIN GENERATION LOOP
# ------------------------------------------------------------
def heterogeneous_spec_decode(prompt: str,
                              max_new_tokens: int = 128,
                              K: int = 64,
                              alpha: float = 0.15):
    """
    Heterogeneous speculative decoding with memory-efficient model handling.
    Handles cases where models might be on different devices (CPU/GPU).
    """
    # encode prompt separately for each model
    # Handle device placement for mixed CPU/GPU scenarios
    tiny_device = next(tiny_lm.parameters()).device
    large_device = next(large_lm.parameters()).device
    
    tiny_ids   = tiny_tok(prompt, return_tensors='pt').input_ids.to(tiny_device)
    large_ids  = large_tok(prompt, return_tensors='pt').input_ids.to(large_device)
    out_large  = large_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens // K + 1):
            # 1) tiny model generates K drafter tokens
            tiny_out = tiny_lm.generate(tiny_ids,
                                        max_new_tokens=K,
                                        do_sample=True,
                                        top_p=0.9,
                                        pad_token_id=tiny_tok.eos_token_id)
            draft_sub = tiny_out[0, tiny_ids.shape[1]:]  # only new part
            if len(draft_sub) == 0:
                break

            # 2) translate to target tokens & merge log-probs
            tgt_ids, spans = translator.merge_subtokens(draft_sub.tolist())
            # compute drafter logits for *each* drafter subtoken
            # (batch trick: feed once, slice hidden)
            tiny_logits = tiny_lm(tiny_out).logits[0, tiny_ids.shape[1]:]
            draft_lp    = translator.merge_logits(tiny_logits, spans)

            # 3) draft tree chooses prefix to propose
            draft_prefix = build_tree(draft_lp)
            if not draft_prefix:
                # drafter not confident – fall back to large model for 1 token
                next_tok = large_lm.generate(out_large, max_new_tokens=1)[0, -1:]
                out_large = torch.cat([out_large, next_tok], dim=-1)
                # re-sync tiny context
                tiny_ids  = tiny_tok(large_tok.decode(out_large[0]),
                                     return_tensors='pt').input_ids.to(DEVICE)
                continue

            # 4) Large model: single forward for verification logits
            ctx_plus = torch.cat([out_large,
                                  torch.tensor(draft_prefix, device=DEVICE).unsqueeze(0)],
                                 dim=-1)
            large_logits = large_lm(ctx_plus).logits[:, -len(draft_prefix):]
            large_lp     = torch.log_softmax(large_logits, dim=-1)[0]

            # 5) SE verification
            accept_len, mix_lp = spec_ensemble_verify(draft_prefix,
                                                      draft_lp[:len(draft_prefix)],
                                                      large_lp,
                                                      alpha)
            # 6) commit accepted tokens
            if accept_len > 0:
                accepted = draft_prefix[:accept_len]
                out_large = torch.cat([out_large,
                                       torch.tensor(accepted,
                                                    device=DEVICE).unsqueeze(0)],
                                      dim=-1)

            # 7) if rejection occurred, resample 1 token from mixture
            if accept_len < len(draft_prefix):
                mix_probs_last = mix_lp[accept_len].softmax(-1)
                resample = torch.multinomial(mix_probs_last, 1)
                out_large = torch.cat([out_large, resample.unsqueeze(0)], dim=-1)

            # 8) update tiny context by re-encoding **only new chunk**
            new_txt = large_tok.decode(out_large[0])
            tiny_ids = tiny_tok(new_txt, return_tensors='pt').input_ids.to(DEVICE)

            # 9) termination
            if out_large[0, -1] == large_tok.eos_token_id:
                break

    return large_tok.decode(out_large[0])

# ------------------------------------------------------------
# 5.  SIMPLE TEST
# ------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    prompt = "In speculative decoding, researchers are exploring"
    print(heterogeneous_spec_decode(prompt, max_new_tokens=120))
