import numpy as np
import pickle
import torch
import torch.nn.functional as F
import os
from collections import defaultdict
from tqdm import tqdm
from representations.sequentialembedding import SequentialEmbedding

# ==========================================
# 1. Utility Functions
# ==========================================

def get_stable_target_words(embeddings: SequentialEmbedding):
    """Filter words that exist across all years with a non-zero norm."""
    years = sorted(embeddings.embeds.keys())
    full_vocab = embeddings.embeds[years[0]].iw 
    stable_words = []
    
    print("Filtering stable target words across all years...")
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
    """Extract neighborhood matrix (B, k, d) for a batch of words."""
    neighbor_embeds = []
    for w in words:
        neigh = embeddings.get_seq_neighbour_set(w, n=k, year=year)
        M = torch.stack([embeddings.get_embed(year, n) for n in neigh])
        neighbor_embeds.append(M)
    return torch.stack(neighbor_embeds).to(device)

# ==========================================
# 2. Geometric Calculations
# ==========================================

def compute_anisotropy_and_pc1(neighbor_embeddings):
    """
    Calculate Anisotropy Score $A(w) = \lambda_1 / \sum \lambda_j$ 
    and the corresponding PC1 unit vector.
    """
    # Centering the local neighborhood
    M = neighbor_embeddings - neighbor_embeddings.mean(dim=1, keepdim=True)

    # Batched SVD: Vh contains the principal components (eigenvectors)
    _, S, Vh = torch.linalg.svd(M, full_matrices=False)

    lambdas = S**2
    total_variance = torch.clamp(lambdas.sum(dim=1), min=1e-9)
    
    # Anisotropy Score: Variance explained by the 1st principal component
    a_scores = lambdas[:, 0] / total_variance
    # PC1 Direction (top eigenvector)
    pc1_vectors = Vh[:, 0, :] 

    return a_scores, pc1_vectors

# ==========================================
# 3. Enhanced Analysis Pipeline
# ==========================================

def run_anisotropy_alignment_analysis(embeddings, vocab, years, k=100, batch_size=256, results_path='result/anisotropy_alignment_results_1950.pkl'):
    """
    Checks for existing pkl to avoid redundant GPU computation.
    Otherwise, computes anisotropy and alignment with temporal displacement.
    """
    # Check if results already exist to skip heavy computation
    if os.path.exists(results_path):
        print(f"Loading existing results from {results_path}...")
        with open(results_path, 'rb') as f:
            return pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ani_series = defaultdict(list)
    alignment_series = defaultdict(list)
    period_stats = {}

    for i in range(len(years)):
        y_t = years[i]
        y_next = years[i+1] if i < len(years) - 1 else None
        
        print(f"\nProcessing Year: {y_t}")
        current_period_alignments = []

        for start in tqdm(range(0, len(vocab), batch_size)):
            batch_words = vocab[start : start + batch_size]
            
            # 1. Compute Anisotropy Score and PC1 direction at time t
            neighbor_matrix = get_neighbor_matrix(embeddings, batch_words, y_t, k=k, device=device)
            scores, pc1_vectors = compute_anisotropy_and_pc1(neighbor_matrix)
            
            # Store Anisotropy Scores per word
            for w, s in zip(batch_words, scores.cpu().numpy()):
                ani_series[w].append(float(s))

            # 2. Compute Alignment (Cosine Similarity) with the next decade's drift
            if y_next is not None:
                v_t = torch.stack([embeddings.get_embed(y_t, w) for w in batch_words]).to(device)
                v_next = torch.stack([embeddings.get_embed(y_next, w) for w in batch_words]).to(device)
                
                # Displacement vector Delta_v = v_{t+1} - v_t
                delta_v = v_next - v_t
                
                # Absolute Cosine Similarity: Range [0, 1]
                # We use absolute value as drift can follow PC1 in either direction
                cos_sim = torch.abs(F.cosine_similarity(delta_v, pc1_vectors, dim=1))
                
                # Mask out words with zero movement (noise reduction)
                drift_mask = torch.norm(delta_v, dim=1) > 1e-4
                valid_alignments = cos_sim[drift_mask].cpu().numpy()
                
                current_period_alignments.extend(valid_alignments.tolist())
                
                # Map individual word alignments
                for j, word in enumerate(batch_words):
                    alignment_series[word].append(float(cos_sim[j].cpu().item()))

        # Summary statistics for the transition y_t -> y_next
        if y_next is not None and current_period_alignments:
            avg_align = np.mean(current_period_alignments)
            period_stats[f"{y_t}->{y_next}"] = avg_align
            print(f"Average Alignment Score ({y_t}->{y_next}): {avg_align:.4f}")

    final_output = {
        'anisotropy_series': dict(ani_series),
        'alignment_series': dict(alignment_series),
        'period_averages': period_stats,
        'common_vocab': vocab,
        'years': list(years)
    }

    # Save to file for future use
    with open(results_path, 'wb') as f:
        pickle.dump(final_output, f)

    return final_output

def print_top_bottom_anisotropy(pkl_path, n=10):
    """
    Identifies and prints the words with the highest and lowest 
    Anisotropy scores for each timestamp.
    """
    # Load the results
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    ani_series = data['anisotropy_series']
    vocab = data['common_vocab']
    years = data['years']

    print("\n" + "="*70)
    print(f"TOP AND BOTTOM {n} WORDS BY ANISOTROPY SCORE")
    print("="*70)

    for i, year in enumerate(years):
        # Create list of (word, score) for the current year
        year_scores = []
        for word in vocab:
            score = ani_series[word][i]
            if not np.isnan(score):
                year_scores.append((word, score))
        
        # Sort by score: descending for top, ascending for bottom
        year_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_n = year_scores[:n]
        bottom_n = year_scores[-n:][::-1] # Reverse to show lowest at the very bottom

        print(f"\n>>> Year: {year} <<<")
        
        # Print Top N
        print(f"  Highest Anisotropy (Narrowest Semantic Space):")
        for idx, (word, score) in enumerate(top_n, 1):
            print(f"    {idx:>2}. {word:<15} ({score:.4f})")
        
        print(f"  ---")
        
        # Print Bottom N
        print(f"  Lowest Anisotropy (Broadest Semantic Space):")
        for idx, (word, score) in enumerate(bottom_n, 1):
            # idx is recalculated to show rank within bottom
            print(f"    {n-idx+1:>2}. {word:<15} ({score:.4f})")
        
        print("-" * 40)

def analyze_by_period(pkl_path):
    # Load the results
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    ani_series = data['anisotropy_series']
    align_series = data['alignment_series']
    vocab = data['common_vocab']
    years = data['years']
    
    print("="*70)
    print(f"{'Period':<15} | {'PCA Group':<12} | {'Avg Alignment':<15} | {'Sample Size'}")
    print("="*70)

    # Iterate through each transition (t -> t+1)
    for i in range(len(years) - 1):
        y_start = years[i]
        y_end = years[i+1]
        period_label = f"{y_start} -> {y_end}"
        
        current_ani_scores = []
        current_align_vals = []
        
        for word in vocab:
            # A(w) at year t
            score_t = ani_series[word][i]
            # Alignment for movement t -> t+1
            alignment_t_next = align_series[word][i]
            
            if not np.isnan(score_t) and not np.isnan(alignment_t_next):
                current_ani_scores.append(score_t)
                current_align_vals.append(alignment_t_next)
        
        current_ani_scores = np.array(current_ani_scores)
        current_align_vals = np.array(current_align_vals)
        
        if len(current_ani_scores) == 0:
            continue
            
        # Define High/Low thresholds for THIS SPECIFIC DECADE
        # We use Top 10% and Bottom 10% for maximum contrast
        high_thresh = np.percentile(current_ani_scores, 90)
        low_thresh = np.percentile(current_ani_scores, 10)
        
        high_group = current_align_vals[current_ani_scores >= high_thresh]
        low_group = current_align_vals[current_ani_scores <= low_thresh]
        
        avg_high = np.mean(high_group)
        avg_low = np.mean(low_group)
        
        # Print results for this decade
        print(f"{period_label:<15} | {'High (Top 10%)':<12} | {avg_high:<15.4f} | {len(high_group)}")
        print(f"{'':<15} | {'Low (Bot 10%)':<12} | {avg_low:<15.4f} | {len(low_group)}")
        
        # Calculate improvement
        improvement = ((avg_high - avg_low) / avg_low) * 100
        print(f"{'':<15} | -> Gap: {improvement:>+6.2f}%")
        print("-" * 70)


# ==========================================
# 4. Main Execution
# ==========================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    target_years = range(1850, 2000, 10) 
    results_file = 'result/anisotropy_alignment_results_1950.pkl'
    
    print("Loading sequential embedding models...")
    embeddings = SequentialEmbedding.load("embeddings/eng-all_sgns", target_years)

    # Move embedding matrices to GPU memory for SVD speedup
    for year in target_years:
        m_tensor = torch.from_numpy(np.array(embeddings.embeds[year].m, copy=True, dtype=np.float32))
        embeddings.embeds[year].m = m_tensor.to(device).contiguous()
        embeddings.embeds[year].m = torch.nan_to_num(embeddings.embeds[year].m, nan=0.0)

    # Handle vocab extraction or loading
    # try:
    #     with open('result/anisotropy_alignment_results_1950.pkl', 'rb') as f:
    #         common_vocab = pickle.load(f)['common_vocab']
    # except:
    common_vocab = get_stable_target_words(embeddings)

    # Run Analysis (Skips computation if pkl exists)
    final_results = run_anisotropy_alignment_analysis(
        embeddings, common_vocab, list(target_years), results_path=results_file
    )

    # --- PRINT FINAL OUTPUT ---
    print("\n" + "="*50)
    print("FINAL ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total words analyzed: {len(final_results['common_vocab'])}")
    print(f"Years covered: {final_results['years']}")
    print("\nPeriodic Average Alignment ($|cos(\Delta v, PC1)|$):")
    for period, avg in final_results['period_averages'].items():
        print(f"  {period} : {avg:.4f}")
    print("="*50)
    print(f"Full results stored in: {results_file}")

    print()

    analyze_by_period('result/anisotropy_alignment_results_1950.pkl')

    print()

    print_top_bottom_anisotropy('result/anisotropy_alignment_results_1950.pkl', n=10)