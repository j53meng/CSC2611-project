from representations.sequentialembedding import SequentialEmbedding
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import torch
from scipy.stats import pearsonr, spearmanr
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt

# --------------id bursts----------------
def find_significant_bursts(id_series, years, sigma_percentile=70):
    """
    Only finds bursts for words that are in the top % of restlessness.
    """
    # Find the threshold for the 70th percentile of restlessness (std dev of id series)
    restlessness_scores = {}
    for word, series in id_series.items():
        restlessness_scores[word] = np.nanstd(series)
    min_restlessness = np.percentile(list(restlessness_scores.values()), sigma_percentile)
    
    bursts = []
    for word, series in id_series.items():
        if restlessness_scores[word] < min_restlessness:
            continue
            
        series = np.array(series)
        mu = np.nanmean(series)
        sigma = np.nanstd(series)
        for i, val in enumerate(series):
        
            z_score = (val - mu) / sigma
            if z_score >= 2.0:
                bursts.append({
                        'word': word,
                        'year': years[i],
                        'z_score': z_score,
                        'id_val': val
                    })

    sorted_words = sorted(restlessness_scores.items(), key=lambda x: x[1])
    n = len(sorted_words)
    q = n // 4 # 25% index

    bottom_25_words, bottom_score = [w for w, score in sorted_words[:q]], [score for w, score in sorted_words[:q]]
    top_25_words, top_score = [w for w, score in sorted_words[-q:]], [score for w, score in sorted_words[-q:]]
    print("Mean std for top 25 is: ", np.mean(top_score))
    print("Mean std for bottom 25 is: ", np.mean(bottom_score))
    print(bottom_25_words[-10:])
            
    return bursts, top_25_words, bottom_25_words

# --------------id pr vs velocity correlation----------------
def calculate_predictability_index_all(id_series, velocity_series, years):
    x_vals = []
    y_vals = []
    
    for word in id_series.keys():
        ids = id_series[word]
        vels = velocity_series[word]
        
        for i in range(len(vels) - 1):
            val_id = ids[i]
            val_vel = vels[i]
            
            if not np.isnan(val_id) and not np.isnan(val_vel):
                x_vals.append(val_id)
                y_vals.append(val_vel)

    r_coeff, p_value = pearsonr(x_vals, y_vals)
    return r_coeff, p_value

def calculate_predictability_index(id_series, velocity_series, years, top25, bottom25):
    print(f"{'Decade':<10} | {'Top 25%':<20} | {'Bot 25%':<20}")
    print("-" * 60)
    corr = []

    for i in range(len(years) - 1):
        year_t = years[i]
    
        def get_xy(group_words, idx):
            x, y = [], []
            for w in group_words:
                val_id = id_series[w][idx]
                val_vel = velocity_series[w][idx]
                if np.isfinite(val_id) and np.isfinite(val_vel):
                    x.append(val_id)
                    y.append(val_vel)
            return x, y
        
        x_top, y_top = get_xy(top25, i)
        x_bottom, y_bottom = get_xy(bottom25, i)
        
        r_top, p_top = pearsonr(x_top, y_top) if len(x_top) > 50 else (0, 1)
        r_bottom, p_bottom = pearsonr(x_bottom, y_bottom) if len(x_bottom) > 50 else (0, 1)

        sig = "***" if p_top < 0.001 else "**" if p_top < 0.01 else "*" if p_top < 0.05 else ""
        
        print(f"{year_t:<10} | {r_top:>10.4f} {'':<8} | {r_bottom:>10.4f} {sig:<3}")
        corr.append(r_top)
    return corr

def get_freqs_series(velocity_series):
    with open('freqs.pkl', 'rb') as f:
        freqs = pickle.load(f, encoding="bytes")
    freqs_series = {}
    for word in velocity_series:
        freqs_series[word] = [freqs[word][y] for y in freqs[word]]
    return freqs_series


def plot_id_vs_velocity(word, id_series, vel_series, years):
    ids = np.array(id_series[word])
    vels = np.array(vel_series[word]) 
    
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # ID
    color = 'tab:blue'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('ID_PR (Instability)', color=color)
    ax1.plot(years, ids, color=color, marker='s', label='Geometry (ID)')
    ax1.tick_params(axis='y', labelcolor=color)

    # Velocity
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Velocity (Movement)', color=color)
    ax2.plot(years[:-1], vels, color=color, marker='o', label='Velocity (V)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"Geometric Precursor vs. Semantic Drift: '{word}'")
    fig.tight_layout()
    plt.savefig(f"./figures/{word}_comparison.png")

def plot_id_vs_freq(id, freq, years):
    plt.figure(figsize=(12, 6))

    # ID
    plt.plot(years, id, marker='o', color='royalblue', linewidth=2.5, label=r'Geometric Force ($ID_{PR}$ vs. $V_{t+1}$)')

    # Frequency
    plt.plot(years, freq, marker='s', color='tab:red', linestyle='--', linewidth=2, alpha=0.8, label=r'Frequency Baseline ($Freq$ vs. $V_{t+1}$)')

    plt.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)
    plt.title("Lexical Dynamics: The 1900 Phase Transition", fontsize=16, fontweight='bold')
    plt.xlabel("Decade", fontsize=13)
    plt.ylabel("pearson Correlation ($r$)", fontsize=13)
    plt.ylim(-0.5, 0.7)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    plt.savefig('./figures/correlation_pearson_trends.png')


# get neighbours before and after burst for a word
def analyze_burst(word, embeddings:SequentialEmbedding, year):
    results = {}
    
    for y in [year, year - 10]:
        neighbor_words = embeddings.get_seq_neighbour_set(word, n=10, year=y)
        
        results[y] = {
            'words': neighbor_words,
        }
    
    print(f"--- Analysis for: {word} ---")
    for label, y in [('Before', year - 10), ('After', year)]:
        print(f"\n[{label} Burst]")
        print(f"Top Neighbors: {', '.join(results[y]['words'])}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    id_series = defaultdict(list)
    velocity_series = defaultdict(list)

    embeddings = SequentialEmbedding.load("embeddings/eng-all_sgns", range(1850, 2000, 10))
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
    
    # load results
    with open('manifold_analysis_results.pkl', 'rb') as f:
        results = pickle.load(f)
        id_series = results['id_series']
        velocity_series = results['velocity_series']
        vocab = results['vocab']
    freq_series = get_freqs_series(velocity_series)

    # Execute the analysis

    # get words with bursts
    bursts, top25, bottom25 = find_significant_bursts(id_series, years)
    print(len(bursts))
    top_bursts = sorted(bursts, key=lambda x: x['z_score'], reverse=True)

    # see neighbours of bursts
    for b in top_bursts[:15]:
        analyze_burst(b['word'], embeddings, b['year'])

    # get info for top 15 bursts (id, velocity rank, frequency rank)
    print(f"{'Word':<15} | {'Year':<6} | {'Z-Score':<8} | {'ID Value':<8} | {'Velocity Rank':<8} | {'Frequency Rank':<8}")
    print("-" * 70)
    for b in top_bursts[:15]:
        cur_year = years.index(b['year'])
        rank = np.nan
        if b['year'] != 1990:
            all_v_scores = [velocity_series[w][cur_year] for w in velocity_series]
            word_v_score = velocity_series[b['word']][cur_year]
            rank = percentileofscore(all_v_scores, word_v_score) // 100.0
        all_freqs = [freq_series[w][cur_year] for w in freq_series]

        freq = freq_series[b['word']][cur_year]
        freq_rank = percentileofscore(all_freqs, freq) // 100.0
        
        print(f"{b['word']:<15} | {b['year']:<6} | {b['z_score']:>7.2f} | {b['id_val']:>8.2f} | {rank:>8.2f}| {freq_rank:>8.2f}")

    # plot_id_vs_velocity('monday', id_series, velocity_series, years)
    
    # correlation analysis for all words
    # restless_words =list(bursts.keys())
    # restless_id = {w: id_series[w] for w in restless_words}
    # restless_vel = {w: velocity_series[w] for w in restless_words}
    # r, p = calculate_predictability_index(restless_id, restless_vel, years)
    # print(f"Predictability Index (r): {r:.4f}")
    # print(f"P-value: {p:.4e}")


    # correlation analysis for top 25% vs bottom 25% restless words for each decade
    print(" id correlation with velocity:")
    id_corr = calculate_predictability_index(id_series, velocity_series, years, top25, bottom25)

    print(" frequency correlation with velocity:")
    freq_corr = calculate_predictability_index(freq_series, velocity_series, years, top25, bottom25)

    # plot trends
    plot_id_vs_freq(id_corr, freq_corr, years[:-1])



