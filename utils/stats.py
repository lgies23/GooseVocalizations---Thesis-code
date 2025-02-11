import numpy as np
import random
import matplotlib.pyplot as plt

from math import isnan
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.cluster import v_measure_score, completeness_score, homogeneity_score
from scipy.stats import bootstrap
from scipy.stats import kruskal
from sklearn.metrics import silhouette_samples, adjusted_rand_score


def silhouette_against_chance(y_pred, embeddings):
    # https://github.com/timsainb/avgn_paper/blob/63e25ca535a230f96b7fb1017728ead6ee0bf36b/notebooks/02.5-make-projection-dfs/indv-id/marmoset-make-umap-get-silhouette-vs-chance.ipynb#L4
    emb_coefficients = silhouette_samples(embeddings, labels=y_pred)
    chance_coefficients = silhouette_samples(embeddings, labels=np.random.permutation(y_pred))
    krusk_res = kruskal(emb_coefficients, chance_coefficients)
    return krusk_res


def silhouette_across_embeddings(y_pred_a, y_pred_b, embeddings):
    # https://github.com/timsainb/avgn_paper/blob/63e25ca535a230f96b7fb1017728ead6ee0bf36b/notebooks/02.5-make-projection-dfs/indv-id/marmoset-make-umap-get-silhouette-vs-chance.ipynb#L4
    a_coefficients = silhouette_samples(embeddings, labels=y_pred_a)
    b_coefficients = silhouette_samples(embeddings, labels=y_pred_b)
    krusk_res = kruskal(a_coefficients, b_coefficients)
    return krusk_res


def hopkins(X, random_seed):
    """
    Computes Hopkins statistic to avaluate clustering tendency of the dataset.
    
    Parameters:
        X (np.ndarray): The dataset to assess (n_samples, n_features).
        random_seed (int): Seed for random number generation for permutation.
    
    Returns:
        float: The Hopkins statistic (0 to 1). Values closer to 1 indicate clusters, 
               and values closer to 0 indicate uniform distribution.
    
    adapted from https://datascience.stackexchange.com/questions/14142/cluster-tendency-using-hopkins-statistic-implementation-in-python and https://matevzkunaver.wordpress.com/2017/06/20/hopkins-test-for-cluster-tendency/ accessed November 2024
    """
    # set seeds again to ensure 
    np.random.seed(random_seed)
    random.seed(random_seed)

    d = X.shape[1]
    n = len(X) # rows
    m = int(0.07 * n) # proportion of random subset
    nbrs = NearestNeighbors(n_neighbors=1).fit(X)
 
    rand_X = random.sample(range(0, n, 1), m)
 
    uid = []
    wid = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(np.random.uniform(np.amin(X,axis=0), np.amax(X,axis=0), d).reshape(1, -1), 2, return_distance=True)
        uid.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X[rand_X[j]].reshape(1, -1), 2, return_distance=True)
        wid.append(w_dist[0][1])
 
    H = sum(uid) / (sum(uid) + sum(wid))
    if isnan(H):
        print(uid, wid)
        H = 0

    return H


def v_measure_against_chance(y_true, y_pred, n_permutations=1000, plot=True):
    n_permutations = 1000

    # true V-measure
    res_bootstrap_v = bootstrap((y_true, y_pred), v_measure_score, n_resamples=n_permutations, paired=True)

    true_v_measure = np.mean(res_bootstrap_v.bootstrap_distribution)
    ci_low, ci_high = res_bootstrap_v.confidence_interval.low, res_bootstrap_v.confidence_interval.high
    st_err = res_bootstrap_v.standard_error
    distribution_v = res_bootstrap_v.bootstrap_distribution

    # null distribution of V-measure scores
    null_v_measures = []
    for _ in range(n_permutations):
        permuted_labels = np.random.permutation(y_pred)
        null_v_measures.append(v_measure_score(y_true, permuted_labels))

    # compute chance level as the mean of the null distribution
    chance_level = np.mean(null_v_measures)

    # compute p-value for the true V-measure
    p_value = np.mean([v >= true_v_measure for v in null_v_measures])

    # print(f"V-measure: {true_v_measure}")
    # print(f"Standard error: {st_err}")
    # print(f"Chance level: {chance_level}")
    # print(f"P-value: {p_value:.6f}")

    if plot:
        fig, ax = plt.subplots()
        ax.hist(null_v_measures, bins=20, density=True)
        ax.axvline(true_v_measure, ls="--", color="r")
        score_label = f"V-measure:: {true_v_measure:.3f}±{st_err:.4f}\n(p-value: {p_value:.6f})"
        ax.text(true_v_measure*1.1, 10, score_label, fontsize=12)
        ax.set_xlabel("Accuracy score")
        ax.set_ylabel("Probability density")
        plt.show()

    return true_v_measure, st_err, ci_low, ci_high, p_value, chance_level, distribution_v


def calculate_metrics_clustering(y_true, y_pred, data, representation_type, n_clusters, random_seed=42, plot=True):
    # https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.v_measure_score.html
    
    result = {}
    result['n clusters'] = n_clusters

    if "embedding" in representation_type:
        hopkins = hopkins(random_seed, data)
        result['Hopkins'] = hopkins
    else: 
        result['Hopkins'] = None
    
    sil_coeffs = silhouette_samples(data, y_pred)
    sil_score = np.mean(sil_coeffs)

    result['Silhouette score/Modularity'] = sil_score
    result['Silhouette coefficients'] = sil_coeffs

    res_bootstrap_h = bootstrap((y_true, y_pred), homogeneity_score, n_resamples=1000, paired=True)
    result['Homogeneity'] = np.mean(res_bootstrap_h.bootstrap_distribution)
    result['Homogeneity CI Low'] = res_bootstrap_h.confidence_interval.low
    result['Homogeneity CI High'] = res_bootstrap_h.confidence_interval.high
    result['Homogeneity Standard Error'] = res_bootstrap_h.standard_error

    res_bootstrap_c = bootstrap((y_true, y_pred), completeness_score, n_resamples=1000, paired=True)
    result['Completeness'] = np.mean(res_bootstrap_c.bootstrap_distribution)
    result['Completeness CI Low'] = res_bootstrap_c.confidence_interval.low
    result['Completeness CI High'] = res_bootstrap_c.confidence_interval.high
    result['Completeness Standard Error'] = res_bootstrap_c.standard_error

    true_v_measure, st_err, ci_low, ci_high, p_value, chance_level, distribution = v_measure_against_chance(y_true, y_pred, plot=plot)
    result['V-measure'] = true_v_measure
    result['V-measure CI Low'] = ci_low
    result['V-measure CI High'] = ci_high
    result['V-measure Standard Error'] = st_err
    result['V-measure p-value'] = p_value
    result['V-measure Chance Level'] = chance_level
    result['V-measure distribution'] = distribution

    res_bootstrap_rand = bootstrap((y_true, y_pred), adjusted_rand_score, n_resamples=1000, paired=True)
    result['Adjusted Rand Score'] = np.mean(res_bootstrap_rand.bootstrap_distribution)
    result['Adjusted Rand Score CI Low'] = res_bootstrap_rand.confidence_interval.low
    result['Adjusted Rand Score CI High'] = res_bootstrap_rand.confidence_interval.high
    result['Adjusted Rand Score Standard Error'] = res_bootstrap_rand.standard_error
    result['Adjusted Rand Score distribution'] = res_bootstrap_rand.bootstrap_distribution

    return result

