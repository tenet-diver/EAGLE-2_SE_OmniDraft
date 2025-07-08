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
import logging

# Configure debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('heterogeneous_spd_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# 0.  LOAD MODELS (edit to taste)
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Memory management utilities
def clear_gpu_memory():
    """Clear GPU memory cache"""
    logger.debug("Clearing GPU memory cache")
    if torch.cuda.is_available():
        memory_before = torch.cuda.memory_allocated()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        memory_after = torch.cuda.memory_allocated()
        logger.debug(f"GPU memory cleared: {memory_before} -> {memory_after} bytes")
    else:
        logger.debug("CUDA not available, skipping GPU memory clear")

def load_model_efficiently(model_id: str, device: str = "cuda", offload_to_cpu: bool = False):
    """Load model with memory management"""
    logger.info(f"Starting model load: {model_id}")
    logger.debug(f"Device: {device}, Offload to CPU: {offload_to_cpu}")
    print(f"Loading model: {model_id}")
    clear_gpu_memory()
    
    # Load with lower precision to save memory
    logger.debug("Loading model with torch.float16 precision")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use half precision
        device_map="auto" if not offload_to_cpu else "cpu",
        low_cpu_mem_usage=True
    )
    
    if not offload_to_cpu and device == "cuda":
        logger.debug(f"Moving model to device: {device}")
        model = model.to(device)
    
    model = model.eval()
    logger.info(f"Model {model_id} loaded successfully")
    logger.debug(f"Model device: {next(model.parameters()).device}")
    logger.debug(f"Model dtype: {next(model.parameters()).dtype}")
    return model

# Load tokenizers first (lightweight)
logger.info("Starting tokenizer loading...")
print("Loading tokenizers...")
TINY_ID   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LARGE_ID  = "Qwen/Qwen3-0.6B"

logger.debug(f"Loading tiny tokenizer: {TINY_ID}")
tiny_tok  = AutoTokenizer.from_pretrained(TINY_ID)
logger.debug(f"Tiny tokenizer vocab size: {tiny_tok.vocab_size}")
logger.debug(f"Tiny tokenizer EOS token: {tiny_tok.eos_token_id}")

logger.debug(f"Loading large tokenizer: {LARGE_ID}")
large_tok = AutoTokenizer.from_pretrained(LARGE_ID)
logger.debug(f"Large tokenizer vocab size: {large_tok.vocab_size}")
logger.debug(f"Large tokenizer EOS token: {large_tok.eos_token_id}")

logger.info("Tokenizers loaded successfully")

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
        logger.info("Initializing NGramTranslator")
        self.dtok   = draft_tokenizer
        self.ttok   = tgt_tokenizer
        self.cache  = {}                 # str(text) -> List[int] tgt_ids
        logger.debug(f"Draft tokenizer vocab size: {draft_tokenizer.vocab_size}")
        logger.debug(f"Target tokenizer vocab size: {tgt_tokenizer.vocab_size}")
        self._seed_common_tokens()
        logger.info(f"NGramTranslator initialized with {len(self.cache)} cached mappings")

    def _seed_common_tokens(self):
        # if token strings coincide, store 1-to-1 mapping
        logger.debug("Seeding common tokens between tokenizers")
        dtok_strings = {self.dtok.convert_ids_to_tokens(i): i
                        for i in range(self.dtok.vocab_size)}
        logger.debug(f"Built draft tokenizer string mapping with {len(dtok_strings)} entries")
        
        common_count = 0
        for tid in range(self.ttok.vocab_size):
            s = self.ttok.convert_ids_to_tokens(tid)
            if s in dtok_strings:
                self.cache[s] = [tid]
                common_count += 1
        
        logger.debug(f"Found {common_count} common tokens between tokenizers")
        logger.debug(f"Cache now contains {len(self.cache)} entries")

    def merge_subtokens(self,
                        draft_ids: List[int]) -> Tuple[List[int], List[slice]]:
        """
        Greedy left-to-right merge of draft_ids → target_ids.
        Returns:
            tgt_ids    – merged ids (target vocab)
            spans      – list of slices: which draft indices compose each tgt_id
        """
        logger.debug(f"merge_subtokens called with {len(draft_ids)} draft_ids: {draft_ids[:10]}{'...' if len(draft_ids) > 10 else ''}")
        text = "".join(self.dtok.convert_ids_to_tokens(i) for i in draft_ids)
        logger.debug(f"Draft text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        # fast path: all at once
        if text in self.cache:
            logger.debug("Fast path: entire text found in cache")
            result = self.cache[text], [slice(0, len(draft_ids))]
            logger.debug(f"Returning cached result: {len(result[0])} target tokens")
            return result
            
        logger.debug("Slow path: greedy left-to-right merge")
        tgt_ids, spans = [], []
        start = 0
        merge_steps = 0
        
        while start < len(draft_ids):
            acc, end = "", start
            found = None
            logger.debug(f"Merge step {merge_steps}: starting at position {start}")
            
            while end < len(draft_ids):
                acc += self.dtok.convert_ids_to_tokens(draft_ids[end])
                if acc in self.cache:
                    found = self.cache[acc]
                    found_slice = slice(start, end+1)
                    logger.debug(f"Found cached mapping: '{acc}' -> {found} (span {start}:{end+1})")
                end += 1
                if len(acc) > 50: 
                    logger.debug("Safety stop: accumulated text > 50 chars")
                    break  # safety stop
                    
            if found is None:
                # fall back to byte-level decomposition
                logger.debug(f"No cache hit for '{acc}', using byte-level decomposition")
                # Use the accumulated string directly for tokenizer
                found = self.ttok(acc, add_special_tokens=False)["input_ids"]
                found_slice = slice(start, end)
                logger.debug(f"Byte-level result: {len(found)} tokens")
                
            tgt_ids.extend(found)
            spans.extend([found_slice]*len(found))
            start = end
            merge_steps += 1
            
        logger.debug(f"Merge completed in {merge_steps} steps: {len(draft_ids)} draft -> {len(tgt_ids)} target tokens")
        logger.debug(f"Target token IDs: {tgt_ids[:10]}{'...' if len(tgt_ids) > 10 else ''}")
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
        logger.debug(f"merge_logits called with draft_logits shape: {draft_logits.shape}")
        logger.debug(f"Number of spans: {len(spans)}")
        logger.debug(f"Spans: {spans[:5]}{'...' if len(spans) > 5 else ''}")
        
        logp = draft_logits.log_softmax(-1)
        logger.debug(f"Log probabilities computed, shape: {logp.shape}")
        
        merged_lp = []
        for i, sp in enumerate(spans):
            lp_sum = logp[sp, :].logsumexp(0)  # product in prob-space
            merged_lp.append(lp_sum)
            if i < 3:  # Log first few for debugging
                logger.debug(f"Span {i} ({sp}): merged logit shape {lp_sum.shape}")
                
        result = torch.stack(merged_lp, dim=0)
        logger.debug(f"Final merged logits shape: {result.shape}")
        return result

translator = NGramTranslator(tiny_tok, large_tok)

# ------------------------------------------------------------
# 2.  EAGLE-2 DYNAMIC TREE  (minimal version)
# ------------------------------------------------------------
def decode_token_safely(tokenizer, token_id):
    """Safely decode a single token ID to text, handling potential errors."""
    try:
        return repr(tokenizer.decode([token_id]))
    except:
        return f"<id:{token_id}>"

def build_tree(draft_lp: torch.Tensor,
               max_depth: int = 8,
               conf_thresh: float = 0.2,
               tokenizer=None):
    """
    draft_lp:   [L, vocab] log-probs for each speculative position.
    Returns the *draft_prefix* we will propose this round (token ids).
    Heuristic: extend while top-prob ≥ conf_thresh, else stop.
    """
    logger.debug(f"build_tree called with draft_lp shape: {draft_lp.shape}")
    logger.debug(f"Parameters: max_depth={max_depth}, conf_thresh={conf_thresh}")
    
    prefix = []
    for pos in range(min(max_depth, draft_lp.shape[0])):
        # Get top-5 predictions for debugging
        top5_logprobs, top5_ids = torch.topk(draft_lp[pos], k=5)
        top5_probs = torch.exp(top5_logprobs)
        
        top_prob = top5_probs[0].item()
        top_id = top5_ids[0].item()
        
        logger.debug(f"Position {pos}: top_prob={top_prob:.4f}, top_id={top_id}, current_prefix_len={len(prefix)}")
        if tokenizer:
            logger.debug(f"  Top-5 draft predictions:")
            for i in range(5):
                token_id = top5_ids[i].item()
                prob = top5_probs[i].item()
                token_text = decode_token_safely(tokenizer, token_id)
                logger.debug(f"    {i+1}. {token_text} (id:{token_id}, prob:{prob:.4f})")
        else:
            logger.debug(f"  Top-5 draft predictions: {[(top5_ids[i].item(), f'{top5_probs[i].item():.4f}') for i in range(5)]}")
        
        if top_prob < conf_thresh:
            logger.debug(f"Stopping: confidence {top_prob:.4f} < threshold {conf_thresh}")
            break
        if len(prefix) >= max_depth:
            logger.debug(f"Stopping: reached max_depth {max_depth}")
            break
            
        prefix.append(int(top_id))
        
    logger.debug(f"build_tree returning prefix of length {len(prefix)}: {prefix}")
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
    logger.debug(f"spec_ensemble_verify called with {len(prefix_ids)} prefix tokens: {prefix_ids}")
    logger.debug(f"draft_lp shape: {draft_lp.shape}, large_lp shape: {large_lp.shape}")
    logger.debug(f"alpha parameter: {alpha}")
    
    # Check if vocabulary sizes match
    if draft_lp.shape[-1] != large_lp.shape[-1]:
        logger.debug(f"Vocabulary size mismatch: draft={draft_lp.shape[-1]}, large={large_lp.shape[-1]}")
        logger.debug("Projecting draft logits to target vocabulary space")
        
        # Create a projection tensor to map draft vocab to target vocab
        # For tokens that exist in both vocabs, use direct mapping
        # For tokens that don't exist in target vocab, distribute probability uniformly
        device = draft_lp.device
        target_vocab_size = large_lp.shape[-1]
        draft_vocab_size = draft_lp.shape[-1]
        
        # Create expanded draft logits with target vocab size
        expanded_draft_lp = torch.full(
            (draft_lp.shape[0], target_vocab_size), 
            float('-inf'), 
            device=device, 
            dtype=draft_lp.dtype
        )
        
        # Copy the draft logits to the first draft_vocab_size positions
        # This assumes the first part of target vocab overlaps with draft vocab
        expanded_draft_lp[:, :draft_vocab_size] = draft_lp
        
        # For remaining positions, use a small uniform probability
        remaining_positions = target_vocab_size - draft_vocab_size
        if remaining_positions > 0:
            # Set a very low but finite log probability for unknown tokens
            uniform_logp = math.log(1e-10)  # Very small probability
            expanded_draft_lp[:, draft_vocab_size:] = uniform_logp
            
        draft_lp = expanded_draft_lp
        logger.debug(f"Draft logits expanded to shape: {draft_lp.shape}")
    
    # mixture logits then probs
    mix_lp = torch.logaddexp(math.log1p(-alpha) + large_lp,
                             math.log(alpha)    + draft_lp)
    logger.debug(f"Mixture logits computed, shape: {mix_lp.shape}")
    
    draft_probs = draft_lp.softmax(-1)
    mix_probs   = mix_lp.softmax(-1)
    logger.debug(f"Probabilities computed - draft_probs: {draft_probs.shape}, mix_probs: {mix_probs.shape}")

    accepted = 0
    for i, tok in enumerate(prefix_ids):
        # Get top-5 predictions from both models for debugging
        draft_top5_logprobs, draft_top5_ids = torch.topk(draft_lp[i], k=5)
        draft_top5_probs = torch.exp(draft_top5_logprobs)
        
        large_top5_logprobs, large_top5_ids = torch.topk(large_lp[i], k=5)
        large_top5_probs = torch.exp(large_top5_logprobs)
        
        mix_top5_logprobs, mix_top5_ids = torch.topk(mix_lp[i], k=5)
        mix_top5_probs = torch.exp(mix_top5_logprobs)
        
        proposed_token_text = decode_token_safely(large_tok, tok)
        logger.debug(f"=== Verification step {i} for token {proposed_token_text} (id:{tok}) ===")
        
        logger.debug(f"  Draft model top-5 predictions:")
        for j in range(5):
            token_id = draft_top5_ids[j].item()
            prob = draft_top5_probs[j].item()
            token_text = decode_token_safely(large_tok, token_id)
            logger.debug(f"    {j+1}. {token_text} (id:{token_id}, prob:{prob:.4f})")
            
        logger.debug(f"  Large model top-5 predictions:")
        for j in range(5):
            token_id = large_top5_ids[j].item()
            prob = large_top5_probs[j].item()
            token_text = decode_token_safely(large_tok, token_id)
            logger.debug(f"    {j+1}. {token_text} (id:{token_id}, prob:{prob:.4f})")
            
        logger.debug(f"  Mixture model top-5 predictions:")
        for j in range(5):
            token_id = mix_top5_ids[j].item()
            prob = mix_top5_probs[j].item()
            token_text = decode_token_safely(large_tok, token_id)
            logger.debug(f"    {j+1}. {token_text} (id:{token_id}, prob:{prob:.4f})")
        
        ratio = (mix_probs[i, tok] / draft_probs[i, tok]).item()
        random_val = np.random.rand()
        logger.debug(f"Token {i} (id={tok}): ratio={ratio:.4f}, random={random_val:.4f}")
        
        if ratio >= random_val:
            accepted += 1
            logger.debug(f"Token {i} accepted (ratio >= random)")
        else:
            logger.debug(f"Token {i} rejected (ratio < random), stopping verification")
            break
            
    logger.debug(f"Verification complete: {accepted}/{len(prefix_ids)} tokens accepted")
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
    logger.info(f"Starting heterogeneous_spec_decode with prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
    logger.debug(f"Parameters: max_new_tokens={max_new_tokens}, K={K}, alpha={alpha}")
    
    # encode prompt separately for each model
    # Handle device placement for mixed CPU/GPU scenarios
    tiny_device = next(tiny_lm.parameters()).device
    large_device = next(large_lm.parameters()).device
    logger.debug(f"Model devices - tiny: {tiny_device}, large: {large_device}")
    
    tiny_ids   = tiny_tok(prompt, return_tensors='pt').input_ids.to(tiny_device)
    large_ids  = large_tok(prompt, return_tensors='pt').input_ids.to(large_device)
    logger.debug(f"Encoded prompt - tiny_ids shape: {tiny_ids.shape}, large_ids shape: {large_ids.shape}")
    logger.debug(f"Tiny tokens: {tiny_ids[0].tolist()[:10]}{'...' if tiny_ids.shape[1] > 10 else ''}")
    logger.debug(f"Large tokens: {large_ids[0].tolist()[:10]}{'...' if large_ids.shape[1] > 10 else ''}")
    
    out_large  = large_ids.clone()
    logger.debug(f"Initial output length: {out_large.shape[1]} tokens")

    with torch.no_grad():
        iteration = 0
        total_accepted = 0
        total_proposed = 0
        
        for _ in range(max_new_tokens // K + 1):
            iteration += 1
            logger.info(f"=== Generation iteration {iteration} ===")
            logger.debug(f"Current output length: {out_large.shape[1]} tokens")
            
            # 1) tiny model generates K drafter tokens
            logger.debug(f"Step 1: Generating {K} draft tokens with tiny model")
            logger.debug(f"Tiny input shape: {tiny_ids.shape}")
            
            tiny_out = tiny_lm.generate(tiny_ids,
                                        max_new_tokens=K,
                                        do_sample=True,
                                        top_p=0.9,
                                        pad_token_id=tiny_tok.eos_token_id)
            
            draft_sub = tiny_out[0, tiny_ids.shape[1]:]  # only new part
            logger.debug(f"Generated draft tokens: {len(draft_sub)} tokens")
            logger.debug(f"Draft token IDs: {draft_sub.tolist()}")
            
            if len(draft_sub) == 0:
                logger.info("No draft tokens generated, breaking")
                break

            # 2) translate to target tokens & merge log-probs
            logger.debug("Step 2: Translating draft tokens to target vocabulary")
            tgt_ids, spans = translator.merge_subtokens(draft_sub.tolist())
            logger.debug(f"Translation result: {len(draft_sub)} draft -> {len(tgt_ids)} target tokens")
            
            # compute drafter logits for *each* drafter subtoken
            # (batch trick: feed once, slice hidden)
            logger.debug("Computing draft logits")
            tiny_logits = tiny_lm(tiny_out).logits[0, tiny_ids.shape[1]:]
            logger.debug(f"Tiny logits shape: {tiny_logits.shape}")
            
            draft_lp    = translator.merge_logits(tiny_logits, spans)
            logger.debug(f"Merged draft log-probs shape: {draft_lp.shape}")

            # 3) draft tree chooses prefix to propose
            logger.debug("Step 3: Building draft tree to choose prefix")
            draft_prefix = build_tree(draft_lp, tokenizer=tiny_tok)
            total_proposed += len(draft_prefix)
            
            if not draft_prefix:
                logger.info("Draft tree returned empty prefix - falling back to large model")
                # drafter not confident – fall back to large model for 1 token
                
                # Get logits to show top-5 predictions before generating
                with torch.no_grad():
                    fallback_logits = large_lm(out_large).logits[0, -1, :]
                    fallback_top5_logprobs, fallback_top5_ids = torch.topk(fallback_logits, k=5)
                    fallback_top5_probs = torch.softmax(fallback_top5_logprobs, dim=0)
                    logger.debug(f"=== Large model fallback predictions ===")
                    logger.debug(f"  Large model top-5 predictions:")
                    for j in range(5):
                        token_id = fallback_top5_ids[j].item()
                        prob = fallback_top5_probs[j].item()
                        token_text = decode_token_safely(large_tok, token_id)
                        logger.debug(f"    {j+1}. {token_text} (id:{token_id}, prob:{prob:.4f})")
                
                next_tok_full = large_lm.generate(out_large, max_new_tokens=1)
                next_tok = next_tok_full[0, -1:].unsqueeze(0)  # Ensure 2D shape [1, 1]
                logger.debug(f"Large model fallback token: {next_tok.squeeze().item()}")
                logger.debug(f"Tensor shapes - out_large: {out_large.shape}, next_tok: {next_tok.shape}")
                out_large = torch.cat([out_large, next_tok], dim=-1)
                # re-sync tiny context
                new_text = large_tok.decode(out_large[0])
                tiny_ids  = tiny_tok(new_text, return_tensors='pt').input_ids.to(DEVICE)
                logger.debug(f"Re-synced tiny context, new length: {tiny_ids.shape[1]}")
                continue

            # 4) Large model: single forward for verification logits
            logger.debug("Step 4: Computing large model verification logits")
            ctx_plus = torch.cat([out_large,
                                  torch.tensor(draft_prefix, device=DEVICE).unsqueeze(0)],
                                 dim=-1)
            logger.debug(f"Context + draft shape: {ctx_plus.shape}")
            
            large_logits = large_lm(ctx_plus).logits[:, -len(draft_prefix):]
            logger.debug(f"Large model logits shape: {large_logits.shape}")
            
            large_lp     = torch.log_softmax(large_logits, dim=-1)[0]
            logger.debug(f"Large model log-probs shape: {large_lp.shape}")

            # 5) SE verification
            logger.debug("Step 5: Speculative ensemble verification")
            accept_len, mix_lp = spec_ensemble_verify(draft_prefix,
                                                      draft_lp[:len(draft_prefix)],
                                                      large_lp,
                                                      alpha)
            total_accepted += accept_len
            logger.info(f"Verification result: {accept_len}/{len(draft_prefix)} tokens accepted")
            
            # 6) commit accepted tokens
            if accept_len > 0:
                accepted = draft_prefix[:accept_len]
                logger.debug(f"Committing {len(accepted)} accepted tokens: {accepted}")
                out_large = torch.cat([out_large,
                                       torch.tensor(accepted,
                                                    device=DEVICE).unsqueeze(0)],
                                      dim=-1)
                logger.debug(f"Output length after acceptance: {out_large.shape[1]}")

            # 7) if rejection occurred, resample 1 token from mixture
            if accept_len < len(draft_prefix):
                logger.debug("Step 7: Rejection occurred, resampling from mixture")
                mix_probs_last = mix_lp[accept_len].softmax(-1)
                logger.debug(f"Mixture probabilities shape at position {accept_len}: {mix_probs_last.shape}")
                
                resample = torch.multinomial(mix_probs_last, 1)
                logger.debug(f"Resampled token: {resample.item()}")
                
                out_large = torch.cat([out_large, resample.unsqueeze(0)], dim=-1)
                logger.debug(f"Output length after resampling: {out_large.shape[1]}")

            # 8) update tiny context by re-encoding **only new chunk**
            logger.debug("Step 8: Updating tiny model context")
            new_txt = large_tok.decode(out_large[0])
            logger.debug(f"Current decoded text length: {len(new_txt)} chars")
            
            tiny_ids = tiny_tok(new_txt, return_tensors='pt').input_ids.to(DEVICE)
            logger.debug(f"Updated tiny context shape: {tiny_ids.shape}")

            # 9) termination
            last_token = out_large[0, -1].item()
            logger.debug(f"Last generated token: {last_token} (EOS: {large_tok.eos_token_id})")
            
            if last_token == large_tok.eos_token_id:
                logger.info("EOS token detected, terminating generation")
                break
                
        # End of generation loop
        logger.info(f"Generation completed after {iteration} iterations")
        logger.info(f"Acceptance rate: {total_accepted}/{total_proposed} = {total_accepted/max(total_proposed,1):.2%}")
        logger.info(f"Final output length: {out_large.shape[1]} tokens")

    final_text = large_tok.decode(out_large[0])
    logger.info(f"Final generated text: '{final_text[:100]}{'...' if len(final_text) > 100 else ''}'")
    return final_text

# ------------------------------------------------------------
# 5.  SIMPLE TEST
# ------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting test execution")
    torch.manual_seed(42)
    logger.debug("Random seed set to 42")
    
    prompt = "In speculative decoding, researchers are exploring"
    logger.info(f"Test prompt: '{prompt}'")
    
    logger.info("Beginning heterogeneous speculative decoding...")
    result = heterogeneous_spec_decode(prompt, max_new_tokens=120)
    
    print("\n" + "="*80)
    print("FINAL RESULT:")
    print("="*80)
    print(result)
    print("="*80)
    
    logger.info("Test execution completed successfully")
