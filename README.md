# Investigating the structure of the greylag goose vocal repertoire: what can unsupervised methods tell us?

Code to the Masters Thesis "Investigating the structure of the greylag goose vocal repertoire: what can unsupervised methods tell us?". 

## Description

This repositore holds all scripts to reproduce the results in the Masters Thesis "Investigating the structure of the greylag goose vocal repertoire: what can unsupervised methods tell us?".
Dependencies are listed in *requirements.txt*.

## Executing code

1. *cut_files_from_selection_tables.ipynb* was used to **cut the call clips from the orignal tracks**, using timestamps and storing goose ID, track ID, clip ID and call type annotation in the filename.
   The resulting clips are stored on the KLF share and were used for all further analysis. This script is optional - you could also start with step two, using the ready clips.
   Use this script if you want to add more data from tracks.

2. *feature_extraction.ipynb* holds the code to extract **acoustic feature vectors**, **spectrograms**, **LFCCs** and (optional) **MFCCs** as well as some **data exploration**.
    The extracted features are exported to *features_and_spectrograms.csv* (not in the repo because the file is > 100 MB.

3. *VAE_embedding_extraction.ipynb* is used to **train the VAE and export the latent vectors**. Weights to the trained VAE weights are in *vae_weights/convolutional_vae_weights_adam_3.pth*.
    Please set retrain to False to just rerun the extraction with the already trained VAE. Latent vectors are exported to *latent_representations_CVAE_adam.csv*.
   
5. *resample_pipeline.ipynb* can be used to embed and cluster the features using **UMAP**, **k-Means**, **HDBSCAN** and **Leiden**. The pipeline is run several times with different subsets and produces all **figures** used in the thesis.
   The sript also prints some raw values and stores a **binomial logit link GLM** that analyzes the influence of subset size, clustering method and representation type on V-measure and Adj. RS.

The *utils* directory stores all utility functions for plotting, embedding, etc. 
   
## Authors

Lena Gies (a12113965@unet.univie.ac.at)

## Acknowledgments

* Sueur, J. (2018). Sound Analysis and Synthesis with R (Springer International Publishing). https://doi.org/10.1007/978-3-319-77647-7.
* Tim Sainburg, personal correspondence
* Traag, V.A., Waltman. L., Van Eck, N.-J. (2018). From Louvain to Leiden: guaranteeing well-connected communities. arXiv:1810.08473
* McInnes L, Healy J, Melville J. 2020 UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. doi:https://doi.org/10.48550/arXiv.1802.03426
* McInnes L, Healy J, Astels S. 2017 hdbscan: Hierarchical density based clustering. JOSS 2, 205. doi:10.21105/joss.00205
* references in code
