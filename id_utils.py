from representations.sequentialembedding import SequentialEmbedding
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import torch
from scipy.stats import pearsonr, spearmanr

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

def id_pr_gpu(neighbor_embeddings):
    M = neighbor_embeddings - neighbor_embeddings.mean(dim=1, keepdim=True)

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

def save_results():
    # find common vocab across all decades
    embeddings = SequentialEmbedding.load("embeddings/eng-all_sgns", range(1950, 2000, 10))
    common_vocab = embeddings.get_common_vocab()
    print(common_vocab[:10])
    target_words = get_stable_target_words(embeddings)
    # print(f"Selected {len(target_words)} stable words for analysis.")

    years = sorted(embeddings.embeds.keys())

    for year in years:
        embeddings.embeds[year].m = (
            torch.from_numpy(np.array(embeddings.embeds[year].m, copy=True, dtype=np.float32))
            .float()
            .to(device)
            .contiguous()
        )
        embeddings.embeds[year].m = torch.nan_to_num(embeddings.embeds[year].m,
                                                    nan=0.0, posinf=0.0, neginf=0.0)

    # Get results for id series and velocity series
    id_series = get_id_series(embeddings, target_words, years) # ------id pr------
    velocity_series = get_velocity_series(embeddings, target_words, years) # ------velocity------

    data_to_save = {
        'id_series': dict(id_series),
        'velocity_series': dict(velocity_series),
        'vocab': target_words
    }

    with open('manifold_analysis_results_1950.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)

    print("Results successfully pickled.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    id_series = defaultdict(list)
    velocity_series = defaultdict(list)

    embeddings = SequentialEmbedding.load("embeddings/eng-all_sgns", range(1950, 2000, 10))
    years = sorted(embeddings.embeds.keys())
    
    save_results()
    


