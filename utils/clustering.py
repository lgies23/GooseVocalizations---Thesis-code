import numpy as np
import random
import itertools

import hdbscan
import leidenalg as la
import igraph as ig

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from utils import plotting

from sklearn.utils import resample



RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def silhouette_analysis_kmeans(data, embeddings=None, range_n_clusters=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11], plot_every_step=False, plot_best_clustering=True):
    """
    adapted from https://scikit-learn.org/1.5/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py

    Performs silhouette analysis using kMeans clustering to find number of clusters with highest silhouette score.

    Parameters:
        data (ndarray of shape (n_samples, n_features)): data to cluster and analyze
        optional:
            embeddings (2D ndarray): embeddings of analyzed data to plot, if not given, data is assumed to be embeddings # TODO switch data and embeddings so it actually makes sense
            plot_every_step (bool): plot results of every analysis step. default=False
            plot_best_clustering (bool): plot clustering with highest silhouette score
            
    Returns:
        max_score (float): highest silhouette score
        n (int): number of clusters where silhouette score is highest
        labels (ndarray of shape (n_samples,)): cluster labels for every data point
    """
    plot_any=False

    if embeddings is None:
        embeddings = data
        plot_any=True

    max_score, n = -1, 0
    
    for n_clusters in range_n_clusters:
        # Initialize clusterer with n_clusters value
        clusterer = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)

        # print(
        #     "For n_clusters =",
        #     n_clusters,
        #     "The silhouette_score is:",
        #     silhouette_avg,
        # )

        if silhouette_avg > max_score:
            max_score, n, labels, centers = silhouette_avg, n_clusters, cluster_labels, clusterer.cluster_centers_
        
        if plot_every_step and plot_any:
            plotting._plot_silhouette_analysis(data, n_clusters, cluster_labels, silhouette_avg, embeddings, clusterer.cluster_centers_)
    
    if plot_best_clustering and plot_any:
        plotting._plot_silhouette_analysis(data, n, labels, max_score, embeddings, centers)
        plotting.plot_embeddings_with_colorcoded_label(data, embeddings, labels, "kMeans", "Cluster")

    #print(f'Best score with {n} clusters: {max_score}')
    return max_score, n, labels


def silhouette_analysis_hdbscan(data, embeddings=None, min_sample_numbers=[1, 2, 3, 4, 5, 15, 25], plot_every_step=False, plot_best_clustering=True):
    """
    adapted from https://scikit-learn.org/1.5/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py

    Performs silhouette analysis using HDBSCAN to find parameter values of min_cluster_sizes, min_sample_numbers with highest silhouette score result.

    Parameters:
        data (ndarray of shape (n_samples, n_features)): data to cluster and analyze
        optional:
            embeddings (2D ndarray): embeddings of analyzed data to plot, if not given, data is assumed to be embeddings # TODO switch data and embeddings so it actually makes sense
            plot_every_step (bool): plot results of every analysis step. default=False
            plot_best_clustering (bool): plot clustering with highest silhouette score
            
    Returns:
        max_score (float): highest silhouette score
        n (int): number of clusters where silhouette score is highest
        labels (ndarray of shape (n_samples,)): cluster labels for every data point
    """

    plot_any = False

    if embeddings is None:
        embeddings = data
        plot_any = True

    max_score, best_n, best_labels, best_params = -1, 0, None, None

    min_cluster_sizes = [int(len(data) * 0.01)]
    
    for min_size, min_samples in itertools.product(min_cluster_sizes, min_sample_numbers):
        # Initialize the clusterer with n_clusters value and a random generator
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_size, 
            min_samples=min_samples, 
            prediction_data=True
            )
        cluster_labels = clusterer.fit_predict(data)

        n_clusters = clusterer.labels_.max()+1
        if n_clusters > 1:
            sil_score = silhouette_score(data, cluster_labels)

            # print(
            #     f'For min_cluster_size={min_size}, min_samples={min_samples}, n_clusters={n_clusters}: \
            #     average silhouette_score={sil_score}'
            # )

            # no centers because clusters may not be convex
            if sil_score > max_score:
                max_score = sil_score
                best_n = n_clusters
                best_labels = cluster_labels
                best_params = (min_size, min_samples)

            if plot_every_step and plot_any:
                plotting._plot_silhouette_analysis(data, n_clusters, cluster_labels, sil_score, embeddings)


    # print(f'\nBest parameter combination: min_cluster_size={best_params[0]}, min_samples={best_params[1]} \
    #       with {best_n} clusters and silhouette score: {max_score}')

    if plot_best_clustering and plot_any:
        plotting._plot_silhouette_analysis(data, best_n, best_labels, max_score, embeddings)
        plotting.plot_embeddings_with_colorcoded_label(data, embeddings, best_labels, "HDBSCAN", "Cluster")
        #clusterer.condensed_tree_.plot()
        #sns.displot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=False) 

    return max_score, best_n, best_labels


def leiden_cluster_from_nn_graph(graph, partition_type=la.ModularityVertexPartition, resolution_parameter=1, seed=RANDOM_SEED):
    # TODO documentation
    # Adapted from Tim Sainburg's workshop code (personal communication).
    knn_indices, knn_dists, _ = graph

    # Extract edges and weights
    edges = []
    weights = []

    for i in range(knn_indices.shape[0]):
        for j in range(1, knn_indices.shape[1]):
            edges.append((i, knn_indices[i, j]))
            weights.append(knn_dists[i, j])

    # Create an igraph graph
    g = ig.Graph(edges=edges, directed=False)
    g.es['weight'] = weights

    # Perform Leiden clustering
    try:
        partition = la.find_partition(g, partition_type, weights=g.es['weight'], seed=seed, resolution_parameter=resolution_parameter)
    except:
        partition = la.find_partition(g, partition_type, weights=g.es['weight'], seed=seed)
    #partition = la.CPMVertexPartition(g, weights=g.es['weight'], resolution_parameter=resolution_parameter)
    #partition = la.RBConfigurationVertexPartition(g, weights=g.es['weight'], resolution_parameter=resolution_parameter) # optimizes Modularity-"inspired" function, includes resolution parameter, compares against random graph
    optimiser = la.Optimiser()

    diff = optimiser.optimise_partition(partition, n_iterations=10)
    # Extract cluster labels
    labels = np.array(partition.membership)

    return labels, partition, g

def modularity_analysis_leiden(graph, calls_df, labels_column, embeddings=None, resolution_parameters=None, plot_every_step=False, plot_best_clustering=True, seed=RANDOM_SEED):
    """
    TODO
    """
    if resolution_parameters == None:
        resolution_parameters = np.linspace(0, 1, 20) # gamma below 1 to find more global group structure
    max_score, n, res_param_max = -1, 0, 0
    
    for res_param in resolution_parameters:
        labels, partition, _ = leiden_cluster_from_nn_graph(graph, res_param, seed=seed)
        modularity_score = partition.modularity
        n_clusters = np.max(labels)+1

        # print(
        #     "For resolution =",
        #     res_param,
        #     "The average modularity is:",
        #     modularity_score,
        #     "N clusters:",
        #     n_clusters
        # )

        if modularity_score > max_score:
            max_score, n, labels_best, res_param_max, best_partition = modularity_score, n_clusters, labels, res_param, partition
            
        if plot_every_step:
            plotting.plot_clusters_on_embeddings(labels, embeddings, title=None)
    
    if plot_best_clustering:
        plotting.plot_clusters_on_embeddings(labels_best, embeddings, title=None)
        plotting.plot_embeddings_with_colorcoded_label(calls_df, embeddings, labels_best, "Leiden", "Cluster")

    calls_df[labels_column] = np.asarray(labels_best)
    #print(f'Best score with {n} clusters and resolution parameter {res_param_max}: {max_score}')
    return max_score, n, labels_best, best_partition

def preset_leiden(graph, calls_df, labels_column, class_number, embeddings=None, resolution_parameters=None, plot_every_step=False, plot_best_clustering=True, seed=RANDOM_SEED):
    """
    TODO
    """
    if resolution_parameters == None:
        resolution_parameters = np.linspace(0, 1, 40) # gamma below 1 to find more global group structure
    max_score, n, min_distance = -1, 0, 30
    best_match = None

    for res_param in resolution_parameters:
        labels, partition, _ = leiden_cluster_from_nn_graph(graph, res_param, seed=seed)
        modularity_score = partition.modularity
        n_clusters = np.max(labels)+1
            
        if plot_every_step:
            plotting.plot_clusters_on_embeddings(labels, embeddings, title=None)
        
        if n_clusters == class_number:
            best_match = (modularity_score, n_clusters, labels, min_distance, partition)
            break
        
        distance = np.abs(n_clusters-class_number)
        if distance < min_distance:
            min_distance = distance
            best_match = (modularity_score, n_clusters, labels, min_distance, partition)
         
    if best_match:
        max_score, n, labels_best, res_param_max, best_partition = best_match

        if plot_best_clustering:
            plotting.plot_clusters_on_embeddings(labels_best, embeddings, title=None)
            plotting.plot_embeddings_with_colorcoded_label(calls_df, embeddings, labels_best, "Leiden", "Cluster")

        calls_df[labels_column] = np.asarray(labels_best)
        #print(f'Best score with {n} clusters and resolution parameter {res_param_max}: {max_score}')
        return max_score, n, labels_best, best_partition
    
    else:
        print("No valid clustering found. Check resolution parameters or input data.")
        return None, None, None, None


def bootstrap_classes_of_size(df, class_size=95, random_seed=42):
    return df.groupby('call_type', group_keys=False).apply(
        lambda x: resample(x, replace=True, n_samples=class_size, random_state=random_seed)
    )