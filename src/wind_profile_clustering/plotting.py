"""
Plotting functions for wind profile clustering visualization.

This module contains functions for visualizing wind profiles, clusters, and
their statistical properties.
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import copy


# Plot limits and labels
xlim_pc12 = [-1.1, 1.1]
ylim_pc12 = [-1.1, 1.1]
x_lim_profiles = [-0.8, 1.25]


def plot_wind_profile_shapes(altitudes, wind_prl, wind_prp, wind_mag=None, n_rows=2, savePlots=False):
    """Plot wind profile shapes showing parallel and perpendicular components.

    Args:
        altitudes (ndarray): Height levels in meters.
        wind_prl (ndarray): Parallel wind speed components for each cluster.
        wind_prp (ndarray): Perpendicular wind speed components for each cluster.
        wind_mag (ndarray): Wind speed magnitudes. Defaults to None.
        n_rows (int): Number of rows in the plot grid. Defaults to 2.
        savePlots (bool): If True, save the plot as PDF. Defaults to False.

    Returns:
        None: Displays the plot.
    """
    n_profiles = len(wind_prl)
    x_label0 = r"$\tilde{v}$ [-]"
    x_label1 = r"$\tilde{v}_{\parallel}$ [-]"
    y_label1 = r"$\tilde{v}_{\bot}$ [-]"

    n_cols = int(np.ceil(n_profiles/n_rows))
    figsize = (n_cols*2+.8, n_rows*4.8+.4)
    height_ratios = np.ones(2*n_rows)
    for j in range(n_rows):
        height_ratios[2*j] = 1.8
    fig, ax = plt.subplots(2*n_rows, n_cols, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': height_ratios}, squeeze=False)
    wspace = 0.2
    plt.subplots_adjust(top=0.955, bottom=0.05, left=0.08, right=0.98, hspace=0.2, wspace=wspace)

    for j in range(n_rows):
        ax[0+2*j, 0].set_xlim(x_lim_profiles)
        ax[0+2*j, 0].set_ylabel("Height [m]")
        ax[1+2*j, 0].set_ylabel(y_label1)

    for i in range(n_profiles):
        prl, prp = wind_prl[i], wind_prp[i]
        k = i%n_cols
        j = i//n_cols*2
        ax[j, k].plot(prl, altitudes, label="Parallel", color='#ff7f0e')
        ax[j, k].plot(prp, altitudes, label="Perpendicular", color='#1f77b4')
        ax[j, k].sharey(ax[0, 0])

        wind_dir = []
        for v_prl, v_prp in zip(wind_prl[i, :], wind_prp[i, :]):
            wind_dir.append(np.arctan2(v_prp, v_prl))

        if wind_mag is not None:
            ax[j, k].plot(wind_mag[i], altitudes, '--', label='Magnitude', color='#2ca02c')

        txt = '${}$'.format(i+1)
        ax[j, k].plot(0.1, 0.1, 'o', mfc="white", alpha=1, ms=14, mec='k', transform=ax[j, k].transAxes)
        ax[j, k].plot(0.1, 0.1, marker=txt, alpha=1, ms=7, mec='k', transform=ax[j, k].transAxes)

        ax[j, k].grid(True)
        ax[j, k].set_xlabel(x_label0)

        ax[j+1, k].plot(prl, prp, color='#7f7f7f')
        ax[j+1, k].plot([0, prl[0]], [0, prp[0]], ':', color='#7f7f7f')
        ax[j+1, k].grid(True)
        ax[j+1, k].axes.set_aspect('equal')
        ax[j+1, k].set_ylim([-.7, .7])
        ax[j+1, k].set_xlabel(x_label1)

        if k > 0:
            ax[j, k].set_yticklabels([])
            ax[j+1, k].set_yticklabels([])

    ax[0, 0].legend(bbox_to_anchor=(-1.5+n_cols*.5, 1.05, 3.+wspace*(n_cols-1), 0.2), loc="lower left", mode="expand",
                    borderaxespad=0, ncol=4)
    
    if savePlots:
        fig.savefig('results/wind_profile_shapes.pdf', bbox_inches='tight')
        print("Saved: results/wind_profile_shapes.pdf")


def plot_bars(array2d, bars_labels=None, ax=None, legend_title="", xticklabels=None):
    """
    Plot bar charts for comparing cluster properties.

    Args:
        array2d (ndarray): 2D array with bars to plot (n_bars x n_cols).
        bars_labels (list): Labels for each bar series. Defaults to None.
        ax (matplotlib.axes.Axes): Axes to plot on. Defaults to None.
        legend_title (str): Title for the legend. Defaults to "".
        xticklabels (list): Labels for x-axis ticks. Defaults to None.

    Returns:
        None: Modifies the axes or creates a new plot.
    """
    n_bars = array2d.shape[0]
    n_cols = array2d.shape[1]
    plot_legend = True
    if bars_labels is None:
        plot_legend = False
        bars_labels = [None]*n_bars
    w_group = .8
    w_bar = w_group/n_bars
    dx = np.linspace(-w_group/2+w_bar/2, w_group/2-w_bar/2, n_bars)
    x0 = np.linspace(0, n_cols-1, n_cols)

    for i_bar, l in enumerate(bars_labels):
        x = x0 + dx[i_bar]
        if ax is None:
            ax = plt
        ax.bar(x, array2d[i_bar, :], width=w_bar, align='center', label=l)
    ax.grid(True)
    if plot_legend:
        if n_bars > 5:
            ncol = int(np.ceil(n_bars/4))
        else:
            ncol = 1
        ax.legend(bbox_to_anchor=(1.01, 1.07), loc="upper left", ncol=ncol, title=legend_title)

    if xticklabels:
        ax.set_xticks(x0)
        ax.set_xticklabels(xticklabels)


def visualise_patterns(n_clusters, wind_data, sample_labels, frequency_clusters, savePlots=False):
    """Visualize temporal and meteorological patterns of clusters.

    Creates bar plots showing cluster frequency distributions across:
    - Years
    - Months
    - Hours of day
    - Wind speed bins
    - Wind direction bins

    Args:
        n_clusters (int): Number of clusters.
        wind_data (dict): Dictionary containing wind data with keys:
            - reference_vector_speed: Wind speed at reference height.
            - years: Tuple of (start_year, end_year).
            - datetime: Array of datetime64 objects.
            - reference_vector_direction: Wind direction in radians.
        sample_labels (ndarray): Cluster labels for each sample.
        frequency_clusters (ndarray): Overall frequency of each cluster.
        savePlots (bool): If True, save the plot as PDF. Defaults to False.

    Returns:
        None: Displays the plot.
    """
    wind_speed_100m = wind_data['reference_vector_speed']
    n_samples = len(wind_speed_100m)

    # Create figure.
    plot_frame_cfg = {'top': 0.99, 'bottom': 0.060, 'left': 0.1, 'right': 0.695, 'hspace': 0.259, 'wspace': 0.2}
    fig_bars, ax_bars = plt.subplots(5, 1, sharex=True, figsize=(10, 8.5))
    fig_bars.subplots_adjust(**plot_frame_cfg)
    ax_bars[0].set_xticks(np.array(range(n_clusters)))
    ax_bars[0].set_xticklabels(range(1, n_clusters+1))
    ax_bars[-1].set_xlabel('Cluster label')

    viridis = copy(plt.get_cmap('viridis'))
    viridis.set_bad(color='white')

    # Study yearly (grouped) variation.
    n_years_group = 1
    year_range = range(wind_data['years'][0], wind_data['years'][1] + 2, n_years_group)
    years = wind_data['datetime'].astype('datetime64[Y]').astype(int) + 1970
    freq2d_year_bin = np.zeros((len(year_range), n_clusters))
    year_bin_labels = []
    for k, (y0, y1) in enumerate(zip(year_range[:-1], year_range[1:])):
        mask_year_bin = (years >= y0) & (years < y1)
        n_samples_yr = np.sum(mask_year_bin)
        lbls_yr = sample_labels[mask_year_bin]

        if n_years_group == 1:
            lbl = "'{}".format(str(y0)[2:])
        else:
            lbl = "'{}-'{}".format(str(y0)[2:], str(y1)[2:])
        year_bin_labels.append(lbl)
        for i_c in range(n_clusters):
            freq2d_year_bin[k, i_c] = np.sum(lbls_yr == i_c)/n_samples_yr * 100.
    plot_bars(freq2d_year_bin, year_bin_labels, ax=ax_bars[0], legend_title="Year bins")
    ax_bars[0].set_ylabel("Frequency [%]")

    # Study seasonal variation.
    month_bin_lims = list(range(1, 13))
    month_bin_lbls = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
                        'October', 'November', 'December']
    month_bin_lbls = [m[:3] for m in month_bin_lbls]
    months = wind_data['datetime'].astype('datetime64[M]').astype(int) % 12 + 1

    freq2d_month_bin = np.zeros((len(month_bin_lims), n_clusters))
    for i_b, (lbl, m0, m1) in enumerate(zip(month_bin_lbls, month_bin_lims, month_bin_lims[1:]+[month_bin_lims[0]])):
        if m0 < m1:
            mask_month_bin = (months >= m0) & (months < m1)
        else:
            mask_month_bin = ((months >= m0) & (months <= 12)) | ((months >= 1) & (months < m1))

        lbls_m = sample_labels[mask_month_bin]

        for i_c in range(n_clusters):
            freq2d_month_bin[i_b, i_c] = np.sum(lbls_m == i_c)/n_samples * 100.
            freq2d_month_bin[i_b, i_c] = freq2d_month_bin[i_b, i_c] / frequency_clusters[i_c] * 100.
    plot_bars(freq2d_month_bin, month_bin_lbls, ax=ax_bars[1], legend_title="Month bins")
    ax_bars[1].set_ylabel("Within-cluster\nfrequency [%]")

    # Study diurnal variation.
    hour_bin_lims = list(range(0, 24, 2))
    hour_bin_lbls = ["{}-{}".format(h0, h1) for h0, h1 in zip(hour_bin_lims, hour_bin_lims[1:]+[24])]
    hours = wind_data['datetime'].astype('datetime64[h]').astype(int) % 24

    freq2d_hour_bin = np.zeros((len(hour_bin_lims), n_clusters))
    for i_b, (lbl, h0, h1) in enumerate(zip(hour_bin_lbls, hour_bin_lims, hour_bin_lims[1:]+[hour_bin_lims[0]])):
        if h0 < h1:
            mask_hour_bin = (hours >= h0) & (hours < h1)
        else:
            mask_hour_bin = ((hours >= h0) & (hours < 24)) | ((hours >= 0) & (hours < h1))

        lbls_hr = sample_labels[mask_hour_bin]

        for i_c in range(n_clusters):
            freq2d_hour_bin[i_b, i_c] = np.sum(lbls_hr == i_c)/n_samples * 100.
            freq2d_hour_bin[i_b, i_c] = freq2d_hour_bin[i_b, i_c] / frequency_clusters[i_c] * 100.
    plot_bars(freq2d_hour_bin, hour_bin_lbls, ax=ax_bars[2], legend_title="UTC hour bins")
    ax_bars[2].set_ylabel("Within-cluster\nfrequency [%]")

    # Study wind speed distributions.
    n_wind_speed_bins = 4
    v_bin_limits = np.percentile(wind_speed_100m, np.linspace(0, 100, n_wind_speed_bins+1))

    freq2d_vbin = np.zeros((n_wind_speed_bins, n_clusters))
    v_bin_labels = []
    for i_v_bin, (v0, v1) in enumerate(zip(v_bin_limits[:-1], v_bin_limits[1:])):
        v_bin_labels.append("{:.1f}".format(v0) + " < $v_{100m}$ <= " + "{:.1f}".format(v1))
        mask_wind_speed_bin = (wind_speed_100m > v0) & (wind_speed_100m <= v1)
        labels_in_v_bin = sample_labels[mask_wind_speed_bin]

        for i_c in range(n_clusters):
            freq2d_vbin[i_v_bin, i_c] = np.sum(labels_in_v_bin == i_c)/n_samples * 100.
            freq2d_vbin[i_v_bin, i_c] = freq2d_vbin[i_v_bin, i_c] / frequency_clusters[i_c] * 100.
    v_bin_labels[0] = "$v_{100m}$ <= " + "{:.1f}".format(v_bin_limits[1])
    v_bin_labels[-1] = "$v_{100m}$ > " + "{:.1f}".format(v_bin_limits[-2])

    plot_bars(freq2d_vbin, v_bin_labels, ax=ax_bars[3], legend_title="Wind speed 100 m bins [m s$^{-1}$]")
    ax_bars[3].set_ylabel("Within-cluster\nfrequency [%]")

    # Study wind direction variation. Downwind direction: + CCW w.r.t. East
    wind_dir_bin_lims = list(np.arange(-180+45/2, 180, 45))  #list(range(-135, 135+1, 90))
    wind_dir_bin_lbls = ['NE', 'N', 'NW', 'W', 'SW', 'S', 'SE', 'E']  #['South', 'East', 'North', 'West']

    wind_dir = wind_data['reference_vector_direction'] * 180./np.pi
    assert wind_dir.min() >= -180. and wind_dir.max() <= 180.

    freq2d_wind_dir_bin = np.zeros((len(wind_dir_bin_lims), n_clusters))
    for i_b, (lbl, dir0, dir1) in enumerate(zip(wind_dir_bin_lbls, wind_dir_bin_lims,
                                                wind_dir_bin_lims[1:]+[wind_dir_bin_lims[0]])):
        if dir0 < dir1:
            mask_wind_dir_bin = (wind_dir > dir0) & (wind_dir <= dir1)
        else:
            mask_wind_dir_bin = ((wind_dir > dir0) & (wind_dir <= 180)) | ((wind_dir >= -180) & (wind_dir <= dir1))

        lbls_wd = sample_labels[mask_wind_dir_bin]

        for i_c in range(n_clusters):
            freq2d_wind_dir_bin[i_b, i_c] = np.sum(lbls_wd == i_c)/n_samples * 100.
            freq2d_wind_dir_bin[i_b, i_c] = freq2d_wind_dir_bin[i_b, i_c] / frequency_clusters[i_c] * 100.

    # Rearrange bin order
    freq2d_wind_dir_bin = np.vstack((freq2d_wind_dir_bin[2:, :], freq2d_wind_dir_bin[:2, :]))
    freq2d_wind_dir_bin = freq2d_wind_dir_bin[::-1, :]
    wind_dir_bin_lbls = wind_dir_bin_lbls[2:] + wind_dir_bin_lbls[:2]
    wind_dir_bin_lbls = wind_dir_bin_lbls[::-1]

    plot_bars(freq2d_wind_dir_bin, wind_dir_bin_lbls, ax=ax_bars[4], legend_title="Upwind direction 100 m bins")
    ax_bars[4].set_ylabel("Within-cluster\nfrequency [%]")
    
    if savePlots:
        fig_bars.savefig('results/cluster_patterns.pdf', bbox_inches='tight')
        print("Saved: results/cluster_patterns.pdf")


def projection_plot_of_clusters(training_data_reduced, labels, clusters_pc, savePlots=False):
    """Plot scatter plot of data projected onto first two principal components.

    Shows the distribution of samples in PC space with cluster centers marked.

    Args:
        training_data_reduced (ndarray): Training data in PC space.
        labels (ndarray): Cluster labels for each sample.
        clusters_pc (ndarray): Cluster centers in PC space.
        savePlots (bool): If True, save the plot as PDF. Defaults to False.

    Returns:
        None: Displays the plot.
    """
    plt.figure(figsize=(4.2, 2.5))
    plt.subplots_adjust(top=0.975, bottom=0.178, left=0.18, right=0.94)
    if len(labels) > 5e4:
        alpha = .01
    else:
        alpha = .05

    n_clusters = len(clusters_pc)
    cmap = plt.get_cmap('Dark2')
    clrs = cmap(np.linspace(0., 1., n_clusters))

    for i, c in enumerate(clrs):
        # Draw all data points belonging to the respective cluster.
        mask_cluster = labels == i
        plt.scatter(training_data_reduced[mask_cluster, 0], training_data_reduced[mask_cluster, 1], marker='.', s=15,
                    c=[c]*np.sum(mask_cluster), alpha=alpha)

        # Draw white circles at cluster centers
        plt.plot(clusters_pc[i, 0], clusters_pc[i, 1], 'o', mfc="white", alpha=1, ms=14, mec='k')
        c = clusters_pc[i, :]
        plt.plot(c[0], c[1], marker='${}$'.format(i+1), alpha=1, ms=7, mec='k')
    plt.xlim(xlim_pc12)
    plt.ylim(ylim_pc12)
    plt.grid()

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    if savePlots:
        plt.savefig('results/pc_projection.pdf', bbox_inches='tight')
        print("Saved: results/pc_projection.pdf")


def plot_all_results(processed_data, res, processed_data_full, labels_full, frequency_clusters_full, n_clusters, savePlots=False):
    """Create all plots for clustering results.

    This is a convenience function that creates all standard visualizations:
    - Wind profile shapes
    - Pattern visualization (temporal and meteorological distributions)
    - PC projection scatter plot
    - Cluster frequency comparison (filtered vs full dataset)

    Args:
        processed_data (dict): Preprocessed wind data (filtered).
        res (dict): Clustering results dictionary.
        processed_data_full (dict): Preprocessed wind data (full dataset).
        labels_full (ndarray): Cluster labels for full dataset.
        frequency_clusters_full (ndarray): Cluster frequencies for full dataset.
        n_clusters (int): Number of clusters.
        savePlots (bool): If True, save all plots as PDF files. Defaults to False.

    Returns:
        None: Displays all plots.
    """
    prl, prp = res['clusters_feature']['parallel'], res['clusters_feature']['perpendicular']
    
    # Plot wind profile shapes
    plot_wind_profile_shapes(processed_data['altitude'], prl, prp, (prl ** 2 + prp ** 2) ** .5, savePlots=savePlots)
    
    # Visualize patterns
    visualise_patterns(n_clusters, processed_data, res['sample_labels'], res['frequency_clusters'], savePlots=savePlots)
    
    # Plot PC projection
    projection_plot_of_clusters(res['training_data_pc'], res['sample_labels'], res['clusters_pc'], savePlots=savePlots)
    
    # Compare cluster frequencies
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    plt.subplots_adjust(top=.9, hspace=.3)
    ax[0].set_title('Filtered dataset')
    plot_bars(res['frequency_clusters'].reshape((1, -1)), ax=ax[0], xticklabels=range(1, n_clusters+1))
    ax[1].set_title('Full dataset')
    plot_bars(frequency_clusters_full.reshape((1, -1)), ax=ax[1], xticklabels=range(1, n_clusters+1))
    for a in ax:
        a.set_ylabel('Cluster frequency [%]')
    
    if savePlots:
        fig.savefig('results/cluster_frequencies_comparison.pdf', bbox_inches='tight')
        print("Saved: results/cluster_frequencies_comparison.pdf")
