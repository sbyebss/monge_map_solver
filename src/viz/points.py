import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

# pylint: disable=too-many-locals


def plot_histogram(samples, num_class=100, path="hist.jpg"):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), facecolor="w")
    ax.hist(
        np.array(samples),
        bins=num_class,
        range=(0, num_class),
        density=True,
        color="gold",
    )
    fig.savefig(path, bbox_inches="tight", dpi=50)


def compare_highd_kde_scatter(
    data_list, fig_path, plot_size=0.5, levels=10, figsize=(12, 3.5)
):
    pca = PCA(n_components=2)
    pca.fit_transform(data_list[0].detach().cpu()[:4000])

    n_col = len(data_list)
    fig, ax = plt.subplots(nrows=1, ncols=n_col, figsize=figsize, facecolor="w")
    colors = ["mediumaquamarine", "darkturquoise", "limegreen"]
    titles = ["Target embedding", "Ours", "Perrot et al. 2016"]
    for idx, color, title in zip(range(n_col), colors, titles):
        data = data_list[idx]
        data = data.detach().cpu()
        data_pca = pca.transform(data)
        sns.kdeplot(
            x=data_pca[:, 0],
            y=data_pca[:, 1],
            ax=ax[idx],
            color=color,
            linewidths=3.0,
            levels=levels,
            alpha=0.7,
        )
        xlims = (-plot_size, plot_size)
        ylims = (-plot_size, plot_size)
        ax[idx].set_xlim(xlims)
        ax[idx].set_ylim(ylims)
        ax[idx].set_title(title, fontsize=18)
        ax[idx].scatter(
            data_pca[:, 0], data_pca[:, 1], color="darkslategray", alpha=0.1
        )
        ax[idx].grid()

    fig.savefig(fig_path, bbox_inches="tight", dpi=200)
