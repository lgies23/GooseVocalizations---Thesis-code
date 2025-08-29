import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib import gridspec
from scipy.spatial import Voronoi
from scipy.spatial import cKDTree
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch, ConnectionPatch
from sklearn.metrics import silhouette_samples

from .helpers import _min_max_scale


# For plotting
sns.set_theme(style="ticks", font_scale=1.6)
CALL_TYPE_COLORS = {'alarm': (0.2823529411764706, 0.47058823529411764, 0.8156862745098039), 
                    'contact': (0.9333333333333333, 0.5215686274509804, 0.2901960784313726), 
                    'departure': (0.41568627450980394, 0.8, 0.39215686274509803), 
                    'distance': (0.8392156862745098, 0.37254901960784315, 0.37254901960784315), 
                    'recruitment': (0.5843137254901961, 0.4235294117647059, 0.7058823529411765), 
                    'triumph': (0.5490196078431373, 0.3803921568627451, 0.23529411764705882)}
DOT_SIZE = 10
ALPHA = .6
RANDOM_SEED = 42


def scatter_projections(
    syllables=None,
    projection=None,
    labels=None,
    ax=None,
    figsize=(10, 10),
    alpha=0.1,
    s=1,
    color="k",
    color_palette="tab20",
    categorical_labels=True,
    show_legend=True,
    tick_pos="bottom",
    tick_size=16,
    cbar_orientation="vertical",
    log_x=False,
    log_y=False,
    grey_unlabelled=True,
    fig=None,
    colornorm=False,
    rasterized=True,
    equalize_axes=True,
    print_lab_dict=False,  # prints color scheme
    lab_dict=CALL_TYPE_COLORS,
):
    """ 
    creates a scatterplot of syllables using some projection
    From Sainburg et al. 2020: https://github.com/timsainb/avgn_paper/blob/V2/avgn/visualization/projections.py
    """
    if projection is None and syllables is None:
            raise ValueError("Either syllables or projections must be passed")

    # color labels
    if labels is not None:
        if categorical_labels:
            if lab_dict is None:
                if (color_palette == "tab10") & (len(np.unique(labels)) < 10):
                    pal = sns.color_palette(color_palette, n_colors=10)
                    pal = np.array(pal)[
                        np.linspace(0, 9, len(np.unique(labels))).astype("int")
                    ]
                else:
                    pal = sns.color_palette(color_palette, n_colors=len(np.unique(labels)))
                lab_dict = {lab: pal[i] for i, lab in enumerate(np.unique(labels))}

            if grey_unlabelled and -1 in lab_dict:
                lab_dict[-1] = [0.95, 0.95, 0.95, 1.0]

            if print_lab_dict:
                print(lab_dict)

            colors = np.array([lab_dict[i] for i in labels])
        else:
            colors = labels  # continuous case
    else:
        colors = color

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if colornorm:
        norm = LogNorm()
    else:
        norm = None

    if categorical_labels or labels is None:
        ax.scatter(
            projection[:, 0],
            projection[:, 1],
            rasterized=rasterized,
            alpha=alpha,
            s=s,
            color=colors,
            norm=norm,
        )

    else:
        cmin = np.quantile(labels, 0.01)
        cmax = np.quantile(labels, 0.99)
        sct = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            vmin=cmin,
            vmax=cmax,
            cmap=plt.get_cmap(color_palette),
            rasterized=rasterized,
            alpha=alpha,
            s=s,
            c=labels,
        )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    if labels is not None:
        if categorical_labels == True:
            legend_elements = [
                Line2D([0], [0], marker="o", color=value, label=key)
                for key, value in lab_dict.items()
            ]
        if show_legend:
            if not categorical_labels:
                if cbar_orientation == "horizontal":
                    axins1 = inset_axes(
                        ax,
                        width="50%",  # width = 50% of parent_bbox width
                        height="5%",  # height : 5%
                        loc="upper left",
                    )
                    # cbar = fig.colorbar(sct, cax=axins1, orientation=cbar_orientation

                else:
                    axins1 = inset_axes(
                        ax,
                        width="5%",  # width = 50% of parent_bbox width
                        height="50%",  # height : 5%
                        loc="lower right",
                    )
                cbar = fig.colorbar(sct, cax=axins1, orientation=cbar_orientation)
                cbar.ax.tick_params(labelsize=tick_size)
                axins1.xaxis.set_ticks_position(tick_pos)
            else:
                ax.legend(handles=legend_elements)
    if equalize_axes:
        ax.axis("equal")
    return ax

def scatter_spec(
    z,
    specs,
    column_size=10,
    pal_color="hls",
    matshow_kwargs={"cmap": plt.cm.Greys},
    scatter_kwargs={"alpha": 0.5, "s": 1},
    line_kwargs={"lw": 1, "ls": "dashed", "alpha": 1},
    color_points=False,
    figsize=(10, 10),
    range_pad=0.1,
    x_range=None,
    y_range=None,
    enlarge_points=0,
    draw_lines=True,
    n_subset=-1,
    ax=None,
    show_scatter=True,
    border_line_width=1,
    img_origin="lower",
):
    n_columns = column_size * 4 - 4
    pal = sns.color_palette(pal_color, n_colors=n_columns)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(column_size, column_size)
        main_ax = fig.add_subplot(gs[1 : column_size - 1, 1 : column_size - 1])

    else:
        fig = ax.figure

       # get subplot bbox in figure coordinates
        bbox = ax.get_position()
        width = bbox.width
        height = bbox.height
        side = min(width, height)

        # center a square in subplot
        cx = bbox.x0 + width/2
        cy = bbox.y0 + height/2
        square_bbox = [cx - side/2, cy - side/2, side, side]

        # main_ax remains the user-passed ax
        main_ax.set_position(square_bbox)

        # now build the ring of small axes manually in figure coordinates
        n_thumbs = len(specs)  # or column_size*4-4
        for i, spec in enumerate(specs):
            # compute row/col in ring as before
            row, col = ...  # same as original logic
            x0 = square_bbox[0] + (col / column_size) * side
            y0 = square_bbox[1] + (row / column_size) * side
            ax_small = fig.add_axes([x0, y0, side/column_size, side/column_size])
            ax_small.matshow(spec, origin="lower", cmap=plt.cm.Greys)
            ax_small.set_xticks([])
            ax_small.set_yticks([])

    fig.patch.set_alpha(0)

    # Determine plotting range
    if x_range is None and y_range is None:
        z_arr = np.vstack(z)
        x_sorted = np.sort(z_arr[:, 0])
        y_sorted = np.sort(z_arr[:, 1])
        xmin, xmax = x_sorted[[int(len(z_arr) * 0.01), int(len(z_arr) * 0.99)]]
        ymin, ymax = y_sorted[[int(len(z_arr) * 0.01), int(len(z_arr) * 0.99)]]
        xmin -= (xmax - xmin) * range_pad
        xmax += (xmax - xmin) * range_pad
        ymin -= (ymax - ymin) * range_pad
        ymax += (ymax - ymin) * range_pad
    else:
        xmin, xmax = x_range
        ymin, ymax = y_range

    x_block = (xmax - xmin) / column_size
    y_block = (ymax - ymin) / column_size

    # Filter to points within range
    z = np.array(z)
    mask = (z[:, 0] > xmin) & (z[:, 1] > ymin) & (z[:, 0] < xmax) & (z[:, 1] < ymax)

    if "labels" in scatter_kwargs:
        scatter_kwargs["labels"] = np.array(scatter_kwargs["labels"])[mask]
    specs = np.array(specs)[mask]
    z = z[mask]

    # Respect n_subset only when positive
    if n_subset is not None and n_subset > 0:
        z = z[:n_subset]
        specs = specs[:n_subset]
        if "labels" in scatter_kwargs:
            scatter_kwargs["labels"] = scatter_kwargs["labels"][:n_subset]

    # # Prepare the main axis
    # main_ax = fig.add_subplot(gs[1 : column_size - 1, 1 : column_size - 1])
    if show_scatter and len(z) > 0:
        scatter_projections(projection=z, ax=main_ax, fig=fig, **scatter_kwargs)

    # Build the ring of small axes
    axs = {}
    for column in range(n_columns):
        # decide row/col on the ring
        if column < column_size:
            row = 0
            col = column
        elif (column >= column_size) and (column < (column_size * 2) - 1):
            row = column - column_size + 1
            col = column_size - 1
        elif (column >= ((column_size * 2) - 1)) and (column < (column_size * 3 - 2)):
            row = column_size - 1
            col = column_size - 3 - (column - column_size * 2)
        else:
            row = n_columns - column
            col = 0

        axs[column] = {"ax": fig.add_subplot(gs[row, col]), "col": col, "row": row}

        # geometric center for this small axis (in data coords)
        xpos = xmin + x_block * col + x_block / 2
        ypos = ymax - y_block * row - y_block / 2
        axs[column]["xpos"] = xpos
        axs[column]["ypos"] = ypos

    main_ax.set_xlim([xmin, xmax])
    main_ax.set_ylim([ymin, ymax])
    main_ax.patch.set_alpha(0)

    if len(z) == 0:
        gs.update(wspace=0, hspace=0)
        return fig  # nothing to connect

    # Voronoi over the ring cell centers, then assign each z-point to nearest center
    points = np.array([[axs[i]["xpos"], axs[i]["ypos"]] for i in axs.keys()])
    voronoi_kdtree = cKDTree(points)
    # index of closest cell center for each z point
    _, point_regions = voronoi_kdtree.query(z)

    # Track which z points have been used so we don't reuse the same one
    used = np.zeros(len(z), dtype=bool)
    lines_list = []

    # Loop through cells on the ring and pick a representative z point
    for key in axs.keys():
        ax_small = axs[key]["ax"]

        # pick a z point assigned to this region that hasn't been used
        in_region = np.where((point_regions == key) & (~used))[0]

        if in_region.size:
            chosen_point = np.random.choice(in_region)
        else:
            # fallback: choose nearest unused to the cell center
            center = np.array([axs[key]["xpos"], axs[key]["ypos"]])
            avail = np.where(~used)[0]
            dists = np.linalg.norm(z[avail] - center, axis=1)
            chosen_point = avail[np.argmin(dists)]

        used[chosen_point] = True

        # draw the spectrogram in the small axis
        ax_small.matshow(
            specs[chosen_point],
            origin=img_origin,
            interpolation="none",
            aspect="auto",
            **matshow_kwargs,
        )
        ax_small.set_xticks([])
        ax_small.set_yticks([])
        ax_small.patch.set_alpha(0)

        # colored borders if requested
        if color_points:
            plt.setp(ax_small.spines.values(), color=pal[key])
        for spine in ax_small.spines.values():
            spine.set_linewidth(border_line_width)

        # optionally enlarge and color the chosen main point
        if enlarge_points > 0:
            dot_color = pal[key] if color_points else "k"
            main_ax.scatter(
                [z[chosen_point, 0]],
                [z[chosen_point, 1]],
                color=dot_color,
                s=enlarge_points,
                zorder=3,
            )

        # draw dashed connection using robust transforms
        if draw_lines:
            # pick an anchor on the edge of the small subplot (in Axes coords)
            line_end_pos = [0.5, 0.5]
            if axs[key]["row"] == 0:
                line_end_pos[1] = 0
            if axs[key]["row"] == column_size - 1:
                line_end_pos[1] = 1
            if axs[key]["col"] == 0:
                line_end_pos[0] = 1
            if axs[key]["col"] == column_size - 1:
                line_end_pos[0] = 0

            con = ConnectionPatch(
                xyA=(z[chosen_point, 0], z[chosen_point, 1]),
                coordsA=main_ax.transData,          # data coords on main axis
                xyB=tuple(line_end_pos),
                coordsB=ax_small.transAxes,         # axes coords on small subplot
                axesA=main_ax,
                axesB=ax_small,
                **line_kwargs,
            )
            fig.add_artist(con)
            lines_list.append(con)

    if isinstance(gs, gridspec.GridSpec):
        gs.update(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=0, hspace=0)
    return fig


def plot_embeddings_with_colorcoded_label(df, embeddings, label_column, plot_title, legend_title=None, show_legend=True):
    """
    TODO 
    """
    if type(label_column) != str:
        unique_labels = np.unique(label_column)
        colors = cm.nipy_spectral(label_column.astype(float) / (np.max(label_column)+1))
        #colors = sns.color_palette("muted", len(unique_labels))
        legend_patches = [Patch(color=color, label=label) for label, color in zip(unique_labels, colors)]
        
    else:
        unique_labels = df[label_column].unique()
        color_palette = sns.color_palette("muted", len(unique_labels))
        label_to_color = {label: color for label, color in zip(unique_labels, color_palette)}
        if "call" in label_column:
            legend_patches = [Patch(color=color, label=label) for label, color in CALL_TYPE_COLORS.items()]
            colors = [CALL_TYPE_COLORS[label] for label in df[label_column]]
        else:
            legend_patches = [Patch(color=color, label=label) for label, color in label_to_color.items()]
            # Assign colors to points
            colors = [label_to_color[label] for label in df[label_column]]

    # Scatter plot
    plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=colors,
        s=DOT_SIZE, 
        alpha=ALPHA
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(plot_title, fontsize=14)

    if type(label_column) == str:
        if show_legend:
            if legend_title is None:
                legend_title = label_column

            # Legend
            plt.legend(
                handles=legend_patches,
                title=legend_title,
                fontsize=8,
                title_fontsize=10,
                loc="center left",
                bbox_to_anchor=(1.0, 0.7)
            )

    plt.show()


def plot_embedding_per_call_type(embeddings, df):
    """
    Plots subplots (scatterplots) for each call type with the datapoints of the call type colored on grey embeddings.

    Parameters:
        embeddings (2D ndarray): Embeddings of the data to plot.
        df (DataFrame): corresponding DataFrame containing the call type label.
    """
    unique_call_types = df["call_type"].unique()
    color_palette = sns.color_palette("muted", len(unique_call_types))
    #call_type_to_color = {call_type: color for call_type, color in zip(unique_call_types, color_palette)}

    cols = 3
    rows = 2

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
    axes = axes.flatten()

    # Determine the global limits to set equal axes
    x_min, x_max = embeddings[:, 0].min()-.8, embeddings[:, 0].max()+.8
    y_min, y_max = embeddings[:, 1].min(), embeddings[:, 1].max()

    for i, call_type in enumerate(unique_call_types):
        ax = axes[i]

        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c="grey",
            alpha=0.1,
            s=DOT_SIZE
        )

        # Filter embeddings for current call type
        mask = df["call_type"] == call_type
        filtered_embeddings = embeddings[mask]
        color = CALL_TYPE_COLORS[call_type]
        
        ax.scatter(
            filtered_embeddings[:, 0],
            filtered_embeddings[:, 1],
            c=color,
            alpha=0.7,
            s=DOT_SIZE
        )
        ax.set_title(call_type, fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', 'datalim')

    plt.show()


def plot_colorcoded_features(feature_columns, feature_titles, embeddings_list, embeddings_titles, calls_df):
    """
    Plots subplots for each feature column and each embedding in a grid layout.

    Parameters:
        feature_columns (list): List of feature column names to visualize.
        feature_titles (list): List of titles corresponding to the feature columns.
        embeddings_list (list): List of embeddings (each as a 2D numpy array).
        embeddings_titles (list): List of titles for each embedding type.
        calls_df (DataFrame): DataFrame containing the feature columns.
    """
    num_features = len(feature_columns)
    num_embeddings = len(embeddings_list)

    fig, axes = plt.subplots(num_features, num_embeddings, figsize=(5 * num_embeddings, 4 * num_features), squeeze=False, gridspec_kw={"width_ratios": [1] * (num_embeddings - 1) + [1.2]})

    colormap = plt.cm.viridis

    for i, (feature_column, feature_title) in enumerate(zip(feature_columns, feature_titles)):
        normalized_values = _min_max_scale(calls_df[feature_column])
        for j, (embeddings, embedding_title) in enumerate(zip(embeddings_list, embeddings_titles)):
            ax = axes[i, j]
            colors = colormap(normalized_values)

            ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                c=colors,
                s=DOT_SIZE
            )
            ax.set_aspect('equal', 'datalim')

            # Only add the embedding title at the top of each column
            if i == 0:
                ax.set_title(embedding_title, fontsize=20)

            # Add a colorbar only for the rightmost column
            if j == num_embeddings - 1:
                cbar = fig.colorbar(
                    plt.cm.ScalarMappable(
                        cmap=colormap, 
                        norm=plt.Normalize(
                            vmin=calls_df[feature_column].min(), 
                            vmax=calls_df[feature_column].max()
                        )
                    ), 
                    ax=ax
                )
                cbar.set_label(feature_title, fontsize=20)

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust layout to fit feature titles on the right
    plt.show()


def _plot_silhouette_analysis(data, n_clusters, cluster_labels, silhouette_score, embeddings, centers=None):
    """
    adapted from https://scikit-learn.org/1.5/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py

    Plots silhouette plot and scatterplots of data with colorcoded clusters from silhouette analysis.
    Parameters:
        data ()
        n_clusters (int): number of clusters
        cluster_labels (ndarray of shape (n_samples,)): array of cluster labels for every data point
        silhouette_score (float):
        embeddings (2D ndarray): embeddings of analyzed data to plot
        optional:
            centers (2D ndarray): centers of the clusters. If given, plotted on the scatter plot. default=None
    """
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1 to 1
    ax1.set_xlim([-0.1, 1])

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette plot for the various clusters.")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_score, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        embeddings[:, 0], embeddings[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    ax2.set_title("Visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
    plt.show()


def display_nearest_neighbours(nn_graph, df, n_examples=5, n_neighbors=15):
    fig, axs = plt.subplots(nrows=n_examples, ncols=n_neighbors, figsize=(n_neighbors, n_examples))

    for i in range(n_examples):
        for j in range(n_neighbors):
            axs[i, j].matshow(df.iloc[nn_graph[0][i][j]]["log_padded_spectrogram"], origin="lower", cmap="magma", aspect="auto")
            axs[i, j].axis("off")
    plt.tight_layout()
    plt.show()

def plot_clusters_on_embeddings(labels, embeddings, title=None):
    unique_labels = np.unique(labels)
    # set colors for labels
    colors = sns.color_palette("muted", len(unique_labels))
    color_dict = {label: color for label, color in zip(unique_labels, colors)}
    c = [color_dict[i] for i in labels]

    plt.scatter(embeddings[:,0], embeddings[:,1], s=1, c=c)
    if title:
        plt.title(title)

    # Create legend elements
    legend_elements = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=color_dict[label],
            markersize=8,
            label=f'Cluster {label}',
        )
        for label in unique_labels
    ]

    # Add legend to the plot
    plt.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize="small",
        title="Clusters",
        frameon=True,
    )
    
    plt.show()
    
def plot_cluster_examples(column_name, unique_labels, calls_df, examples_per_cluster=6):
    num_clusters = len(unique_labels)
    fig, axs = plt.subplots(num_clusters, examples_per_cluster, figsize=(examples_per_cluster * 4, num_clusters * 4))

    # Ensure axs is a 2D array even if there's only one cluster
    if num_clusters == 1:
        axs = np.expand_dims(axs, axis=0)

    # Loop through each cluster
    for cluster_idx, cluster in enumerate(unique_labels):
        # Filter spectrograms belonging to the current cluster
        
        cluster_spectrograms = calls_df[calls_df[column_name] == cluster]

        # Randomly sample spectrograms from the cluster
        sampled_spectrograms = cluster_spectrograms.sample(min(examples_per_cluster, len(cluster_spectrograms)), random_state=RANDOM_SEED)

        # Loop through each selected spectrogram
        for example_idx, (index, row) in enumerate(sampled_spectrograms.iterrows()):
            # Get the spectrogram data
            spectrogram = row["log_padded_spectrogram"]  

            # Plot the spectrogram
            ax = axs[cluster_idx, example_idx]
            ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f"Cluster {cluster}, Call_type: {row['call_type']}")
            ax.axis('off')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def clusterability_plot(clusterability_results):
    fig, axs = plt.subplots(ncols=2, figsize=(20, 8))

    columns = ["Silhouette Score", "Modularity"]
    for i, column in enumerate(columns):
        # Raw data points
        sns.stripplot(
            x="Representation", 
            y=column, 
            data=clusterability_results, 
            ax=axs[i], 
            jitter=True, 
            size=10, 
            hue="Algorithm", 
            legend=(True if i == len(columns)-1 else False)
        )
        # Mean points
        sns.pointplot(
            x="Representation", 
            y=column, 
            data=clusterability_results, 
            ax=axs[i], 
            errorbar=None,
            markers="o", 
            color="grey",
            alpha=0.3
        )

        axs[i].set_xticks(range(len(['PAFs', 'LFCCs', 'Spectrograms', 'VAE representations'])))
        axs[i].set_xticklabels(['PAFs', 'LFCCs', 'Spectrograms', 'VAE representations'], rotation=45, ha='right')
        axs[i].set_title(column)
        axs[i].set_xlabel('')
        axs[i].set_ylabel('')

    plt.tight_layout()
    plt.show()

