import numpy as np
import pickle
import torch
import torch.nn.functional as F
import os
from collections import defaultdict
from tqdm import tqdm
from representations.sequentialembedding import SequentialEmbedding
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy.stats import spearmanr, pearsonr

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

    ani_series = defaultdict(list)
    alignment_series = defaultdict(list)
    drift_magnitude_series = defaultdict(list)
    period_stats = {}

    for i in range(len(years)):
        y_t = years[i]
        y_next = years[i+1] if i < len(years) - 1 else None
        
        for start in tqdm(range(0, len(vocab), batch_size)):
            batch_words = vocab[start : start + batch_size]

            neighbor_matrix = get_neighbor_matrix(embeddings, batch_words, y_t, k=k, device=device)
            scores, pc1_vectors = compute_anisotropy_and_pc1(neighbor_matrix)
            
            for w, s in zip(batch_words, scores.cpu().numpy()):
                ani_series[w].append(float(s))

            if y_next is not None:
                v_t = torch.stack([embeddings.get_embed(y_t, w) for w in batch_words]).to(device)
                v_next = torch.stack([embeddings.get_embed(y_next, w) for w in batch_words]).to(device)

                cos_sim_raw = F.cosine_similarity(v_t, v_next, dim=1)
                cos_dist = 1 - cos_sim_raw
                
                delta_v = v_next - v_t
                alignment = torch.abs(F.cosine_similarity(delta_v, pc1_vectors, dim=1))
                
                for j, word in enumerate(batch_words):
                    alignment_series[word].append(float(alignment[j].cpu().item()))
                    drift_magnitude_series[word].append(float(cos_dist[j].cpu().item()))

    final_output = {
        'anisotropy_series': dict(ani_series),
        'alignment_series': dict(alignment_series),
        'drift_magnitude_series': dict(drift_magnitude_series), # 存入 pkl
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
    """
    Performs two categorical analyses:
    1. Group by Anisotropy -> Measure Alignment
    2. Group by Alignment  -> Measure Drift Magnitude
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    ani_series = data['anisotropy_series']
    align_series = data['alignment_series']
    drift_series = data.get('drift_magnitude_series', {})
    vocab = data['common_vocab']
    years = data['years']
    
    # Define a shared separator for the table
    line_width = 85
    print("=" * line_width)
    print(f"{'Period':<15} | {'Metric Group':<20} | {'Avg Value':<15} | {'Sample Size'}")
    print("=" * line_width)

    for i in range(len(years) - 1):
        period_label = f"{years[i]} -> {years[i+1]}"
        
        # Collect current lists for all words at index i
        scores_t = []   # Anisotropy
        aligns_t = []   # Alignment
        drifts_t = []   # Drift Magnitude
        
        for word in vocab:
            # Check indices to avoid errors
            if i < len(ani_series[word]) and i < len(align_series[word]):
                scores_t.append(ani_series[word][i])
                aligns_t.append(align_series[word][i])
                drifts_t.append(drift_series[word][i])
        
        scores_t = np.array(scores_t)
        aligns_t = np.array(aligns_t)
        drifts_t = np.array(drifts_t)

        # --- ANALYSIS 1: Group by Anisotropy, measure Alignment ---
        ani_high_thresh = np.percentile(scores_t, 90)
        ani_low_thresh = np.percentile(scores_t, 10)
        
        mask_h_ani = scores_t >= ani_high_thresh
        mask_l_ani = scores_t <= ani_low_thresh
        
        avg_align_h_ani = np.mean(aligns_t[mask_h_ani])
        avg_align_l_ani = np.mean(aligns_t[mask_l_ani])
        ani_gap = ((avg_align_h_ani - avg_align_l_ani) / avg_align_l_ani) * 100

        # --- ANALYSIS 2: Group by Alignment, measure Drift Magnitude ---
        align_high_thresh = np.percentile(aligns_t, 90)
        align_low_thresh = np.percentile(aligns_t, 10)
        
        mask_h_align = aligns_t >= align_high_thresh
        mask_l_align = aligns_t <= align_low_thresh
        
        avg_drift_h_align = np.mean(drifts_t[mask_h_align])
        avg_drift_l_align = np.mean(drifts_t[mask_l_align])
        align_gap = ((avg_drift_h_align - avg_drift_l_align) / avg_drift_l_align) * 100

        # --- PRINTING RESULTS ---
        # Part 1: Anisotropy -> Alignment
        print(f"{period_label:<15} | {'High Ani (Top 10%)':<20} | {avg_align_h_ani:<15.4f} | {sum(mask_h_ani)}")
        print(f"{'':<15} | {'Low Ani (Bot 10%)':<20} | {avg_align_l_ani:<15.4f} | {sum(mask_l_ani)}")
        print(f"{'':<15} | -> Align Gap: {ani_gap:>+6.2f}%")
        
        print(f"{'':<15} | {'-'*45}") # Sub-separator

        # Part 2: Alignment -> Drift Magnitude
        print(f"{'':<15} | {'High Align (10%)':<20} | {avg_drift_h_align:<15.4f} | {sum(mask_h_align)}")
        print(f"{'':<15} | {'Low Align (10%)':<20} | {avg_drift_l_align:<15.4f} | {sum(mask_l_align)}")
        print(f"{'':<15} | -> Drift Gap: {align_gap:>+6.2f}%")
        
        print("-" * line_width)


def visualize_word_detailed_shift_arrow(pkl_path, embeddings, target_word="to", k=100):
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found.")
        return

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    years = data['years']
    ani_series = data['anisotropy_series']

    if target_word not in ani_series:
        print(f"Error: Word '{target_word}' not found in analyzed vocabulary.")
        return

    print(f"Generating sequence for '{target_word}'...")

    for i, year in enumerate(years):
        current_score = ani_series[target_word][i]
        next_year = years[i + 1] if i < len(years) - 1 else None

        # --- Extract vectors ---
        v_t = embeddings.get_embed(year, target_word)
        if torch.is_tensor(v_t): v_t = v_t.detach().cpu().numpy()

        neighbors = embeddings.get_seq_neighbour_set(target_word, n=k, year=year)
        neighbor_vecs, neighbor_labels = [], []
        for nw in neighbors:
            v_nw = embeddings.get_embed(year, nw)
            if torch.is_tensor(v_nw): v_nw = v_nw.detach().cpu().numpy()
            neighbor_vecs.append(v_nw)
            neighbor_labels.append(nw)

        neighbor_vecs = np.array(neighbor_vecs)
        all_vectors = np.vstack([v_t.reshape(1, -1), neighbor_vecs])

        # --- PCA ---
        pca = PCA(n_components=2)
        coords = pca.fit_transform(all_vectors)

        target_coord = coords[0]
        neighbor_coords = coords[1:]

        # --- Compact the neighbor cloud ---
        cloud_center = neighbor_coords.mean(axis=0)
        COMPACT_SCALE = 0.6
        neighbor_coords_compact = cloud_center + (neighbor_coords - cloud_center) * COMPACT_SCALE

        # --- Semantic drift vector ---
        drift_v_projected = np.zeros(2)
        if next_year is not None:
            v_next = embeddings.get_embed(next_year, target_word)
            if torch.is_tensor(v_next): v_next = v_next.detach().cpu().numpy()
            drift_v = v_next - v_t
            drift_v_projected = np.dot(pca.components_, drift_v)
   
        fig = plt.figure(figsize=(12, 13))

        ax_header = fig.add_axes([0.0, 0.88, 1.0, 0.12])
        ax_header.axis('off')

        ax = fig.add_axes([0.05, 0.05, 0.90, 0.82])
        ax.set_aspect('equal')
        ax.axis('off')

        ax_header.text(
            0.5, 0.78,
            f"Figure {i + 1}.  Semantic neighbourhood of '{target_word}'  ({year})",
            fontsize=14, fontweight='bold', color='#1a1a2e',
            ha='center', va='top', transform=ax_header.transAxes
        )
        ax_header.text(
            0.5, 0.38,
            f"Anisotropy  A(w) = {current_score:.4f}",
            fontsize=10, color='#555555',
            ha='center', va='top', transform=ax_header.transAxes,
            style='italic'
        )

        legend_x = 0.02
        legend_y = 0.08

        ax_header.annotate(
            '', xy=(legend_x + 0.055, legend_y), xytext=(legend_x, legend_y),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='<->', color='#4472C4', lw=1.8, mutation_scale=10)
        )
        ax_header.text(
            legend_x + 0.065, legend_y,
            "PC1 axis (bidirectional, sign-invariant)",
            fontsize=8.5, color='#4472C4',
            va='center', transform=ax_header.transAxes
        )

        if next_year is not None:
            drift_legend_x = legend_x + 0.40
            ax_header.annotate(
                '', xy=(drift_legend_x + 0.055, legend_y), xytext=(drift_legend_x, legend_y),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color='#2ca02c', lw=1.8, mutation_scale=10)
            )
            ax_header.text(
                drift_legend_x + 0.065, legend_y,
                f"Semantic drift → {next_year}",
                fontsize=8.5, color='#2ca02c',
                va='center', transform=ax_header.transAxes
            )

        ax_header.axhline(y=0.0, color='#cccccc', linewidth=0.8, xmin=0.02, xmax=0.98)

        for j, label in enumerate(neighbor_labels):
            ax.text(
                neighbor_coords_compact[j, 0],
                neighbor_coords_compact[j, 1],
                f" {label}",
                fontsize=7.5, alpha=0.65,
                color='#2c4a7c',
                zorder=2
            )

        ax.scatter(
            target_coord[0], target_coord[1],
            c='#d62728', edgecolors='#8b0000',
            s=220, zorder=5, linewidths=1.2
        )
        ax.text(
            target_coord[0], target_coord[1],
            f"  {target_word.upper()}",
            fontsize=18, fontweight='bold', color='#d62728',
            zorder=6,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=2),
            va='bottom'
        )

        x_vals = neighbor_coords_compact[:, 0]
        axis_reach = np.percentile(np.abs(x_vals), 90) * 1.3

        pc1_arrow = FancyArrowPatch(
            (-axis_reach, 0), (axis_reach, 0),
            arrowstyle='<->',
            color='#4472C4',
            linewidth=2.0,
            mutation_scale=14,
            zorder=1,
            alpha=0.75
        )
        ax.add_patch(pc1_arrow)

        ax.text(
            axis_reach + 0.01, 0,
            "PC1+", fontsize=8, color='#4472C4', alpha=0.8,
            ha='left', va='center'
        )
        ax.text(
            -axis_reach - 0.01, 0,
            "PC1−", fontsize=8, color='#4472C4', alpha=0.8,
            ha='right', va='center'
        )
        ax.text(
            0, -np.percentile(np.abs(neighbor_coords_compact[:, 1]), 95) * 0.15,
            "(sign-invariant axis)",
            fontsize=7.5, color='#4472C4', alpha=0.55,
            ha='center', va='top', style='italic'
        )

        if next_year is not None and np.linalg.norm(drift_v_projected) > 1e-6:
            drift_arrow = FancyArrowPatch(
                (target_coord[0], target_coord[1]),
                (target_coord[0] + drift_v_projected[0],
                 target_coord[1] + drift_v_projected[1]),
                arrowstyle='-|>',
                color='#2ca02c',
                linewidth=2.2,
                mutation_scale=16,
                zorder=10,
                alpha=0.9
            )
            ax.add_patch(drift_arrow)
            ax.text(
                target_coord[0] + drift_v_projected[0],
                target_coord[1] + drift_v_projected[1],
                f"  → {next_year}",
                fontsize=9, color='#2ca02c',
                fontweight='bold', alpha=0.9, zorder=11,
                ha='left', va='center'
            )

        all_x = np.concatenate([neighbor_coords_compact[:, 0], [target_coord[0]]])
        all_y = np.concatenate([neighbor_coords_compact[:, 1], [target_coord[1]]])
        margin_x = (all_x.max() - all_x.min()) * 0.15 + 0.02
        margin_y = (all_y.max() - all_y.min()) * 0.15 + 0.02
        ax.set_xlim(all_x.min() - margin_x, all_x.max() + margin_x)
        ax.set_ylim(all_y.min() - margin_y, all_y.max() + margin_y)

        save_name = f"Figure {i + 1}.png"
        plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Saved {save_name}  [{year}]")

    print(f"\nDone. Generated Figure 1 to Figure {len(years)}.")

def calculate_alignment_drift_correlation(pkl_path):
    """
    Calculates the statistical correlation between Alignment and Drift Magnitude
    for every word across all periods.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    align_series = data['alignment_series']
    drift_series = data.get('drift_magnitude_series', {})
    vocab = data['common_vocab']
    years = data['years']
    
    print("="*85)
    print(f"{'Period':<15} | {'Spearman Rho':<15} | {'P-Value':<15} | {'Interpretation'}")
    print("="*85)

    for i in range(len(years) - 1):
        period_label = f"{years[i]} -> {years[i+1]}"
        
        all_aligns = []
        all_drifts = []
        
        for word in vocab:
            # Ensure data exists for this word at this specific decade index
            if i < len(align_series[word]) and i < len(drift_series[word]):
                a_val = align_series[word][i]
                d_val = drift_series[word][i]
                
                # Filter out NaNs or extremely small noise
                if not np.isnan(a_val) and not np.isnan(d_val):
                    all_aligns.append(a_val)
                    all_drifts.append(d_val)
        
        if len(all_aligns) < 2:
            continue

        # Calculate Spearman Correlation
        rho, p_val = spearmanr(all_aligns, all_drifts)
        
        # Determine significance
        significance = "Significant" if p_val < 0.05 else "Not Sig."
        strength = "Positive" if rho > 0 else "Negative"
        
        print(f"{period_label:<15} | {rho:<15.4f} | {p_val:<15.2e} | {strength} ({significance})")

    print("="*85)


    generate_alignment_drift_scatter_plots(pkl_path)



def generate_alignment_drift_scatter_plots(pkl_path):
    """
    Generates scatter plots ONLY for words in the top 50th percentile of Anisotropy A(w).
    Saves as plot_top50_1.png, plot_top50_2.png, etc.
    """
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found.")
        return

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    ani_series = data['anisotropy_series']
    align_series = data['alignment_series']
    drift_series = data.get('drift_magnitude_series', {})
    vocab = data['common_vocab']
    years = data['years']

    print(f"Filtering for Top 50% Anisotropy and generating plots...")

    for i in range(len(years) - 1):
        period_label = f"{years[i]} -> {years[i+1]}"
        
        # 1. Collect all A(w) scores for this year to find the median
        year_ani_scores = []
        for word in vocab:
            if i < len(ani_series[word]):
                val = ani_series[word][i]
                if not np.isnan(val):
                    year_ani_scores.append(val)
        
        if not year_ani_scores: continue
        
        # Calculate 50th percentile (Median) threshold
        threshold = np.percentile(year_ani_scores, 95)
        
        # 2. Filter words based on threshold
        filtered_aligns = []
        filtered_drifts = []
        
        for word in vocab:
            if (i < len(ani_series[word]) and i < len(align_series[word]) and 
                i < len(drift_series[word])):
                
                a_score = ani_series[word][i]
                al_val = align_series[word][i]
                dr_val = drift_series[word][i]
                
                # Check if word is in the top 50% of Anisotropy
                if not np.isnan(a_score) and a_score >= threshold:
                    if not np.isnan(al_val) and not np.isnan(dr_val):
                        filtered_aligns.append(al_val)
                        filtered_drifts.append(dr_val)
        
        if len(filtered_aligns) < 10: continue

        filtered_aligns = np.array(filtered_aligns)
        filtered_drifts = np.array(filtered_drifts)

        # 3. Calculate Correlation for the filtered set
        rho, p_val = spearmanr(filtered_aligns, filtered_drifts)

        # 4. Plotting
        plt.figure(figsize=(10, 7))
        # Use a different color (Dark Orange) to indicate filtered data
        plt.scatter(filtered_aligns, filtered_drifts, alpha=0.4, color='darkorange', s=12, label='Words (Top 50% A(w))')
        
        # Add Trend Line
        m, b = np.polyfit(filtered_aligns, filtered_drifts, 1)
        plt.plot(filtered_aligns, m*filtered_aligns + b, color='black', linewidth=2, label='Trend Line')

        plt.title(f"Period: {period_label} (Filtered: Top 50% Anisotropy)\nSpearman Rho: {rho:.4f} (p: {p_val:.2e})", fontsize=13)
        plt.xlabel("Alignment with PC1 ($|cos|$)", fontsize=11)
        plt.ylabel("Drift Magnitude (Cosine Distance)", fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()

        save_name = f"plot_top50_{i+1}.png"
        plt.savefig(save_name, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {save_name} for {period_label} (Words included: {len(filtered_aligns)})")

# ==========================================
# 4. Main Execution
# ==========================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    target_years = range(1950, 2000, 10) 
    results_file = 'result/anisotropy_alignment_results_1950_2.pkl'
    
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

    analyze_by_period(results_file)
    print()

    # print_top_bottom_anisotropy(results_file, n=10)
    # print()

    calculate_alignment_drift_correlation(results_file)

    # visualize_word_detailed_shift_arrow(
    #     pkl_path=results_file, 
    #     embeddings=embeddings, 
    #     target_word="broadcast", 
    #     k=100
    # )