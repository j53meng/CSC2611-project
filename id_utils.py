from representations.sequentialembedding import SequentialEmbedding
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import torch
from scipy.stats import pearsonr

def get_stable_target_words(embeddings: SequentialEmbedding):
    years = sorted(embeddings.embeds.keys())
    full_vocab = embeddings.embeds[years[0]].iw 
    stable_words = []
    
    for word in full_vocab:
        is_alive = True
        for year in years:
            vec = embeddings.get_embed(year, word)
            
            # the norm is zero, discard
            if np.linalg.norm(vec) < 1e-6:
                is_alive = False
                break
        
        if is_alive:
            stable_words.append(word)
            
    return stable_words

def get_neighbor_matrix(embeddings, words, year, k):

    neighbor_embeds = []

    for w in words:
        neigh = embeddings.get_seq_neighbour_set(w, n=k, year=year)
        M = torch.stack([embeddings.get_embed(year, n) for n in neigh])
        neighbor_embeds.append(M)

    return torch.stack(neighbor_embeds).to("cuda")

def get_id_pr(embeddings: SequentialEmbedding, word: str, year: int, k=10):
    neighbors = embeddings.get_seq_neighbour_set(word, n=k, year=year)
    M = np.array([embeddings.get_embed(year, n) for n in neighbors])
    
    M_centered = M - np.mean(M, axis=0)
    _, sigmas, _ = np.linalg.svd(M_centered, full_matrices=False)
    
    # PR ratio
    lambdas = sigmas**2
    V_total = np.sum(lambdas)
    denominator = np.sum((lambdas / V_total)**2)
    
    return 1 / denominator

def get_neighbor_matrix(embeddings, words, year, k):

    neighbor_embeds = []

    for w in words:
        neigh = embeddings.get_seq_neighbour_set(w, n=k, year=year)
        M = torch.stack([embeddings.get_embed(year, n) for n in neigh])
        neighbor_embeds.append(M)

    return torch.stack(neighbor_embeds).to("cuda")

def id_pr_gpu(neighbor_embeddings):
    """
    neighbor_embeddings: (B, k, d)
    B = batch size (#words)
    k = neighbors
    d = embedding dim
    """

    # center
    M = neighbor_embeddings - neighbor_embeddings.mean(dim=1, keepdim=True)

    # batched SVD
    _, S, _ = torch.linalg.svd(M, full_matrices=False)

    lambdas = S**2
    V_total = lambdas.sum(dim=1, keepdim=True)

    denom = ((lambdas / V_total) ** 2).sum(dim=1)

    return 1 / denom

def get_velocity(embeddings: SequentialEmbedding, words: str, year: int, year_next: int):
    idx = torch.tensor([embeddings.embeds[year].wi[w] for w in words],device=device)
    M_t = embeddings.embeds[year].m[idx]         # (num_words, dim)
    M_t_next = embeddings.embeds[year_next].m[idx]
    
    cos_dist = 1 - torch.sum(M_t * M_t_next, dim=1)
    return cos_dist

def get_id_series(embeddings, vocab, years):
    batch_size = 256
    for i, year in enumerate(years):
        print(f"Processing {year}")

        for start in tqdm(range(0, len(vocab), batch_size)):

            batch_words = vocab[start:start+batch_size]

            neighbor_matrix = get_neighbor_matrix(
                embeddings, batch_words, year, k=100
            )

            pr_vals = id_pr_gpu(neighbor_matrix)

            for w, val in zip(batch_words, pr_vals.cpu().numpy()):
                id_series[w].append(val)
    return id_series

def get_velocity_series(embeddings, vocab, years):
    all_dists = []

    for i in range(len(years) - 1):
        y, y_next = years[i], years[i+1]

        cos_dist = get_velocity(embeddings, vocab, y, y_next)

        all_dists.append(cos_dist)
    all_dists = torch.stack(all_dists, dim=1)
    velocity_series = {
        vocab[i]: all_dists[i].tolist()
        for i in range(len(vocab))
    }
    return velocity_series

def save_results(id_series, velocity_series, vocab):
    with open('manifold_analysis_results.pkl', 'rb') as f:
        results = pickle.load(f)
        id_series = results['id_series']


    data_to_save = {
        'id_series': dict(id_series),
        'velocity_series': dict(velocity_series),
        'vocab': vocab
    }

    with open('manifold_analysis_results.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)

    print("Results successfully pickled.")

# --------------id bursts----------------
def find_id_bursts(id_series, threshold=2.0):
    """
    Identifies years of structural instability for a word.
    """
    # 1. Handle NaNs from missing/dead years (like 'judean')
    series = np.array(id_series)
    mu = np.nanmean(series)
    sigma = np.nanstd(series)
    print(f"Mean: {mu}, Std: {sigma}")
    
    if sigma == 0: return [] # Structurally locked words

    # 2. Find peaks
    z_scores = (series - mu) / sigma
    burst_years = np.where(z_scores > threshold)[0]
    
    return burst_years

def find_significant_bursts(id_series, sigma_percentile=70):
    """
    Only finds bursts for words that are in the top % of restlessness.
    """
    # Find the threshold for the 70th percentile of restlessness
    restlessness_scores = {}
    for word, series in id_series.items():
        restlessness_scores[word] = np.nanstd(series)
    min_restlessness = np.percentile(list(restlessness_scores.values()), sigma_percentile)
    
    significant_bursts = {}
    for word, series in id_series.items():
        if restlessness_scores[word] < min_restlessness:
            continue
            
        series = np.array(series)
        mu = np.nanmean(series)
        sigma = np.nanstd(series)
        
        z_scores = (series - mu) / sigma
        
        burst_years = np.where(z_scores > 1.5)[0]
        if len(burst_years) > 0:
            significant_bursts[word] = burst_years
            
    return significant_bursts

# --------------id pr vs velocity correlation----------------
def calculate_predictability_index(id_series, velocity_series, years):
    x_vals = []
    y_vals = []
    
    for word in id_series.keys():
        ids = id_series[word]
        vels = velocity_series[word]
        
        # We need a lag: ID at time t vs Velocity at time t+10
        # If IDs[0] is 1850, Vels[1] is the drift from 1860 to 1870
        for i in range(len(vels) - 1):
            val_id = ids[i]
            val_vel = vels[i + 1] # This is the 't+10' kinetic outcome 
            
            # Skip NaNs (dead years like 'judean')
            if not np.isnan(val_id) and not np.isnan(val_vel):
                x_vals.append(val_id)
                y_vals.append(val_vel)

    # Calculate Pearson correlation 
    r_coeff, p_value = pearsonr(x_vals, y_vals)
    return r_coeff, p_value

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def calculate_trajectory_alignment(embeddings: SequentialEmbedding, word, year_t, year_next, k_neighbors=100):
    # 1. Get the word's neighborhood at time t
    # (Assuming you have a function to get neighbor vectors)
    neighbors_matrix = get_neighbor_matrix(embeddings, [word], year_t, k=k_neighbors).detach().cpu().numpy()[0]
    print(neighbors_matrix.shape)
    
    # 2. Cluster the neighbors to find 'Semantic Gravity'
    kmeans = KMeans(n_clusters=3, n_init=10).fit(neighbors_matrix)
    
    # Find the largest cluster (highest density)
    counts = np.bincount(kmeans.labels_)
    dense_cluster_idx = np.argmax(counts)
    mu_dense = kmeans.cluster_centers_[dense_cluster_idx]
    
    # 3. Define the Vectors
    w_t = embeddings.get_embed(year_t, word).detach().cpu().numpy()
    w_next = embeddings.get_embed(year_next, word).detach().cpu().numpy()
    
    actual_displacement = w_next - w_t
    predicted_target = mu_dense - w_t
    
    # 4. Calculate Alignment Score
    alignment = cosine_similarity(
        actual_displacement.reshape(1, -1), 
        predicted_target.reshape(1, -1))[0][0]
    
    return alignment

if __name__ == "__main__":
    # find common vocab across all decades
    embeddings = SequentialEmbedding.load("embeddings/eng-all_sgns", range(1950, 2000, 10))
    # common_vocab = embeddings.get_common_vocab()
    # print(common_vocab[:10])
    # target_words = get_stable_target_words(embeddings)
    # print(f"Selected {len(target_words)} stable words for analysis.")

    id_series = defaultdict(list)

    ### gpu
    years = sorted(embeddings.embeds.keys())

    device = "cuda"

    for year in years:
        embeddings.embeds[year].m = (
            torch.from_numpy(np.array(embeddings.embeds[year].m, copy=True, dtype=np.float32))
            .float()
            .to(device)
            .contiguous()
        )
        embeddings.embeds[year].m = torch.nan_to_num(embeddings.embeds[year].m,
                                                    nan=0.0, posinf=0.0, neginf=0.0)

    # ------id pr------
    
    # vocab = target_words
    with open('manifold_analysis_results.pkl', 'rb') as f:
        results = pickle.load(f)
        id_series = results['id_series']
        velocity_series = results['velocity_series']
        common_vocab = results['common_vocab']

    # Execute the analysis

    # align = calculate_trajectory_alignment(embeddings, 'gay', 1970, 1980)
    # print(f"alignment score is {align:.4f}")
    bursts = find_significant_bursts(id_series)
    print(len(bursts))
    restless_words =list(bursts.keys())
    restless_id = {w: id_series[w] for w in restless_words}
    restless_vel = {w: velocity_series[w] for w in restless_words}
    r, p = calculate_predictability_index(restless_id, restless_vel, years)
    print(f"Predictability Index (r): {r:.4f}")
    print(f"P-value: {p:.4e}")

