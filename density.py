import numpy as np
import pickle
import torch
import torch.nn.functional as F
import os
from collections import defaultdict
from tqdm import tqdm
from representations.sequentialembedding import SequentialEmbedding

# ==========================================
# 1. Directory Setup
# ==========================================
# Create result directory if it doesn't exist
RESULT_DIR = 'result'
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

RESULTS_PATH = os.path.join(RESULT_DIR, 'density_analysis_results.pkl')

# ==========================================
# 2. Core Logic: Local Density D(w)
# ==========================================

def compute_local_density(word_vectors, neighbor_matrices):
    """
    Implements: D(w) = (1/k) * sum( (v_w · v_ni) / (||v_w|| * ||v_ni||) )
    """
    v_w_norm = F.normalize(word_vectors, p=2, dim=1).unsqueeze(1) # (B, 1, d)
    v_n_norm = F.normalize(neighbor_matrices, p=2, dim=2)        # (B, k, d)
    
    # Batch Matrix Multiplication for Cosine Similarity
    cos_sims = torch.bmm(v_w_norm, v_n_norm.transpose(1, 2))
    
    # Average across k neighbors
    density_scores = cos_sims.mean(dim=2).squeeze(1)
    return density_scores

def compute_semantic_change(v_t, v_next):
    """Measures 1 - Cosine Similarity between two time steps."""
    cos_sim = F.cosine_similarity(v_t, v_next, dim=1)
    return 1 - cos_sim

# ==========================================
# 3. Utility Functions
# ==========================================

def get_stable_target_words(embeddings: SequentialEmbedding):
    years = sorted(embeddings.embeds.keys())
    full_vocab = embeddings.embeds[years[0]].iw 
    stable_words = []
    
    print("Filtering stable target words...")
    for word in tqdm(full_vocab):
        is_alive = True
        for year in years:
            vec = embeddings.get_embed(year, word)
            norm = torch.norm(vec).item() if isinstance(vec, torch.Tensor) else np.linalg.norm(vec)
            if norm < 1e-6:
                is_alive = False
                break
        if is_alive:
            stable_words.append(word)
    return stable_words

def get_neighbor_matrix(embeddings, words, year, k, device="cuda"):
    neighbor_embeds = []
    for w in words:
        neigh = embeddings.get_seq_neighbour_set(w, n=k, year=year)
        M = torch.stack([embeddings.get_embed(year, n) for n in neigh])
        neighbor_embeds.append(M)
    return torch.stack(neighbor_embeds).to(device)

# ==========================================
# 4. Analysis Pipeline
# ==========================================

def run_density_impact_analysis(embeddings, vocab, years, k=100, batch_size=256):
    # Check if results already exist in the result/ folder
    if os.path.exists(RESULTS_PATH):
        print(f"Loading existing results from {RESULTS_PATH}...")
        with open(RESULTS_PATH, 'rb') as f:
            return pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    density_series = defaultdict(list)
    change_series = defaultdict(list)

    for i in range(len(years)):
        y_t = years[i]
        y_next = years[i+1] if i < len(years) - 1 else None
        print(f"\nProcessing Year: {y_t}")

        for start in tqdm(range(0, len(vocab), batch_size)):
            batch_words = vocab[start : start + batch_size]
            v_t = torch.stack([embeddings.get_embed(y_t, w) for w in batch_words]).to(device)
            
            # Density D(w)
            neighbor_matrix = get_neighbor_matrix(embeddings, batch_words, y_t, k=k, device=device)
            densities = compute_local_density(v_t, neighbor_matrix)
            
            for w, d in zip(batch_words, densities.cpu().numpy()):
                density_series[w].append(float(d))

            # Semantic Change (Trajectory Movement)
            if y_next is not None:
                v_next = torch.stack([embeddings.get_embed(y_next, w) for w in batch_words]).to(device)
                changes = compute_semantic_change(v_t, v_next)
                for w, c in zip(batch_words, changes.cpu().numpy()):
                    change_series[w].append(float(c))

    final_results = {
        'density_series': dict(density_series),
        'change_series': dict(change_series),
        'vocab': vocab,
        'years': list(years)
    }

    # Saving to result/density_analysis_results.pkl
    with open(RESULTS_PATH, 'wb') as f:
        pickle.dump(final_results, f)
    print(f"\nResults successfully saved to {RESULTS_PATH}")
    return final_results

def print_summary(results):
    print("\n" + "="*90)
    print(f"{'Period':<15} | {'Density Group':<20} | {'Avg Change (1-Cos)':<20} | {'Count'}")
    print("="*90)

    years = results['years']
    vocab = results['vocab']
    
    for i in range(len(years) - 1):
        y_start, y_end = years[i], years[i+1]
        densities = np.array([results['density_series'][w][i] for w in vocab])
        changes = np.array([results['change_series'][w][i] for w in vocab])
        
        # Split by percentiles
        hi_thresh, lo_thresh = np.percentile(densities, 80), np.percentile(densities, 20)
        hi_mask, lo_mask = (densities >= hi_thresh), (densities <= lo_thresh)
        
        avg_hi, avg_lo = changes[hi_mask].mean(), changes[lo_mask].mean()
        diff = (avg_hi - avg_lo) / avg_lo * 100
        
        print(f"{y_start}->{y_end:<8} | High (Top 20%)      | {avg_hi:<20.6f} | {sum(hi_mask)}")
        print(f"{'':<15} | Low (Bottom 20%)    | {avg_lo:<20.6f} | {sum(lo_mask)}")
        print(f"{'':<15} | -> Impact: {'RESTRAINED' if diff < 0 else 'ACCELERATED'} ({diff:>+6.2f}%)")
        print("-" * 90)

if __name__ == "__main__":
    target_years = range(1850, 2000, 10)
    embeddings = SequentialEmbedding.load("embeddings/eng-all_sgns", target_years)

    # Optimization: Moving to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for year in target_years:
        m_tensor = torch.from_numpy(np.array(embeddings.embeds[year].m, copy=True, dtype=np.float32))
        embeddings.embeds[year].m = m_tensor.to(device).contiguous()

    vocab = get_stable_target_words(embeddings)
    results = run_density_impact_analysis(embeddings, vocab, list(target_years))
    print_summary(results)