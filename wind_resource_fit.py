import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from scipy.stats import gaussian_kde
from principal_component_analysis import add_inside_panel_labels, plot_frequency_projection
from dimensionality_reduction import *
from read_data.dowa import read_data
import pandas as pd
from utils import add_panel_labels


def eval_loc(loc='mmij'):
    from read_data.dowa import read_data
    wind_data = read_data({'name': loc})
    wind_speed = (wind_data['wind_speed_east']**2 + wind_data['wind_speed_north']**2)**.5
    n_samples = wind_speed.shape[0]

    if loc == 'mmca':
        rl = .03
    else:
        rl = .0002
    pcs = np.load("pcs_{}.npy".format(loc))
    shape_modes = np.insert(pcs, 0, log_law_wind_profile2(wind_data['altitude'], rl, 1, 200, 0), axis=0)

    x_sol = np.empty((n_samples, 3))
    for i in range(n_samples):
        v = wind_speed[i:i+1, :]
        x0 = np.array([
            np.amax(v),
            0,
            0
        ])
        x_sol[i, :] = least_squares(fit_err, x0, args=(v, shape_modes), bounds=((0, -np.inf, -np.inf), np.inf)).x
        print("{}/{}: {}".format(i, n_samples, x_sol[i, :]))
    np.save("x_sol_{}.npy".format(loc), x_sol)


def fit_err_cluster(x, wind_speed, profile_shape, weights=1):
    v_err = wind_speed - x*profile_shape
    v_err = v_err*weights
    return v_err


def assign_to_cluster(loc, include_shapes='all', weighted_fit=True):
    if loc == 'mmca':
        rl = .03
    else:
        rl = .0002

    wind_data = read_data({'name': loc})
    wind_speed = (wind_data['wind_speed_east'] ** 2 + wind_data['wind_speed_north'] ** 2) ** .5
    n_samples = wind_speed.shape[0]

    if weighted_fit:
        weights = {
            10.: 0,
            20.: 0,
            40.: 0,
            60.: 0,
            80.: 0,
            100.: .1+.35,
            120.: .2+.1,
            140.: .15+.05,
            150.: .1,
            160.: .15,
            180.: .2,
            200.: .2,
            220.: .25,
            250.: .4,
            300.: 1.25,
            500.: 1.5,
            600.: 0.5+0.5,
        }
        weights = np.array(list(weights.values()))
    else:
        weights = 1.

    j_200m = np.argmax(wind_data['altitude'] == 200)
    cluster_shapes = np.load("cluster_shapes_{}.npy".format(loc))
    for i in range(len(cluster_shapes)):
        cluster_shapes[i, :] = cluster_shapes[i, :]/cluster_shapes[i, j_200m]
    shapes = np.insert(cluster_shapes, 0, log_law_wind_profile2(wind_data['altitude'], rl, 1, 200, 0), axis=0)
    if include_shapes != 'all':
        shapes = shapes[include_shapes, :]
    n_shapes = shapes.shape[0]

    x_cluster = np.empty((n_samples, 2))
    for i in range(n_samples):
        print("{}/{}".format(i, n_samples))
        v = wind_speed[i, :]
        costs = np.empty(n_shapes)
        magnitudes = np.empty(n_shapes)
        for j in range(n_shapes):
            res = least_squares(fit_err_cluster, v[j_200m], args=(v, shapes[j, :], weights), bounds=(0, np.inf))
            costs[j] = res.cost
            magnitudes[j] = res.x
        i_min = np.argmin(costs)
        x_cluster[i, 0] = i_min
        x_cluster[i, 1] = magnitudes[i_min]

    if include_shapes != 'all':
        add_str = ''.join([str(i) for i in include_shapes])
    else:
        add_str = ''
    if not weighted_fit:
        add_str += '_notweighted'
    file_name = "wind_resource_fit/x_cluster{}_{}.npy".format(add_str, loc)
    np.save(file_name, x_cluster)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_3d_results(loc='mmij'):
    x_sol = np.load("x_sol_{}.npy".format(loc))
    cluster_shapes = np.load("cluster_shapes_{}.npy".format(loc))

    altitudes = np.array([10., 20., 40., 60., 80., 100., 120., 140., 150., 160., 180., 200., 220., 250., 300., 500., 600.])
    if loc == 'mmca':
        rl = .03
    else:
        rl = .0002
    pcs = np.load("pcs_{}.npy".format(loc))
    pc_data = np.load("pc_data_{}.npy".format(loc))
    shape_modes = np.insert(pcs, 0, log_law_wind_profile2(altitudes, rl, 1, 200, 0), axis=0)
    x_pc_plane = log_law_wind_profile2(100, rl, 1, 600, 0)

    # ax1 = plt.figure().gca()
    # ax1.set_xlabel('Log profile [m/s]')
    # ax1.set_ylabel('Scaling PC1 [m/s]')
    # sns.kdeplot(x=x_sol[:, 0], y=x_sol[:, 1], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Blues"), 0.2, 1.))
    # ax2 = plt.figure().gca()
    # ax2.set_xlabel('Log profile [m/s]')
    # ax2.set_ylabel('Scaling PC2 [m/s]')
    # sns.kdeplot(x=x_sol[:, 0], y=x_sol[:, 2], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Blues"), 0.2, 1.))
    # ax1.axvline(x_pc_plane, color='grey')
    # ax2.axvline(x_pc_plane, color='grey')
    # for i, pc_prfl in enumerate(hand_picked_shapes_pc):
    #     ax1.plot(x_pc_plane, pc_prfl[0], 's')
    #     ax2.plot(x_pc_plane, pc_prfl[1], 's')

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # subset = np.random.choice(x_sol.shape[0], x_sol.shape[0] // 10, replace=False)
    # ax.scatter(x_sol[subset, 0], x_sol[subset, 1], x_sol[subset, 2], c=x_cluster[subset, 0], alpha=.02)
    # ax.set_xlabel('Magnitude')

    fig = plt.figure(figsize=[8, 8])
    plt.subplots_adjust(top=1.04, bottom=.065, wspace=0.12, hspace=.08, left=.07, right=.97)
    spec = fig.add_gridspec(ncols=3, nrows=3, height_ratios=[4, 1, 1])
    ax3d = fig.add_subplot(spec[0, :], projection='3d')
    ax3d.set_xlabel(r'$v_{\rm log,200m}$ [m/s]')
    ax3d.set_xlim([0, 30])
    ax3d.set_ylabel(r'$k_{\rm PC1}$ [-]')
    ax3d.set_zlabel(r'$k_{\rm PC2}$ [-]')

    n_bins = 6
    bin_edges = np.linspace(0, 26, n_bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    hand_picked_shapes_points = np.zeros((n_bins, cluster_shapes.shape[0], 3))
    for i, prfl in enumerate(cluster_shapes):
        hand_picked_shape_lines = np.zeros((2, 3))
        r = 25
        v_prfl = prfl*r
        x0 = np.array([
            np.amax(v_prfl),
            0,
            0
        ])
        hand_picked_shape_lines[1, :] = least_squares(fit_err, x0, args=(v_prfl, shape_modes), bounds=((0, -np.inf, -np.inf), np.inf)).x
        ax3d.plot(hand_picked_shape_lines[:, 0], hand_picked_shape_lines[:, 1], hand_picked_shape_lines[:, 2])

        for j, vwl200m in enumerate(bin_centers):
            hand_picked_shapes_points[j, i, 0] = vwl200m
            hand_picked_shapes_points[j, i, 1] = vwl200m/hand_picked_shape_lines[1, 0] * hand_picked_shape_lines[1, 1]
            hand_picked_shapes_points[j, i, 2] = vwl200m/hand_picked_shape_lines[1, 0] * hand_picked_shape_lines[1, 2]
    ax3d.plot([0, r], [0, 0], [0, 0], 'k')

    n_total = x_sol.shape[0]
    # ax2 = plt.subplots(2, n_bins//2, sharex=True, sharey=True)[1].reshape(-1)
    # ax3 = plt.subplots(2, n_bins//2, sharex=True, sharey=True)[1].reshape(-1)

    wind_data = read_data({'name': loc})
    wind_speed = (wind_data['wind_speed_east'] ** 2 + wind_data['wind_speed_north'] ** 2) ** .5
    ax_means = plt.subplots(1, 2)[1]

    ax = []
    for i, (le, ue) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (x_sol[:, 0] >= le) & (x_sol[:, 0] < ue)
        n_bin = np.sum(mask)
        print("Percentage of points in bins: {:.1f}%".format(n_bin/n_total*100))
        kernel = gaussian_kde(x_sol[mask, 1:].T)

        x, y = np.mgrid[-20:20:50j, -10:10:50j]
        positions = np.vstack([x.ravel(), y.ravel()])
        z = np.reshape(kernel(positions).T, x.shape)*n_bin/n_total*100

        print(np.amax(z))

        clevels = np.array([0.03, 0.05, 0.1, 0.25, 0.5, 0.75, 1., 2.])*3/n_bins  #, 4., 6., 8.
        print(clevels)


        # ax = plt.figure().gca()
        # cp = ax.contour(x, y, z, cmap=plt.cm.gist_earth_r)
        # print(cp.cvalues)
        # ax.clabel(cp, inline=True, fontsize=10)

        # ax[0].imshow(np.rot90(z), cmap=plt.cm.gist_earth_r, extent=[-20, 20, -10, 10])

        a = fig.add_subplot(spec[1+i//3, i % 3])
        if i % 3 != 0:
            a.set_yticklabels([])
        else:
            a.set_ylabel(r'$k_{\rm PC2}$ [-]')
        if i // 3 == 0:
            a.set_xticklabels([])
        else:
            a.set_xlabel(r'$k_{\rm PC1}$ [-]')
        cset = a.contour(x, y, z, clevels, cmap=truncate_colormap(plt.cm.gist_earth_r, 0.05, 1.), norm=colors.LogNorm())  #, (le+ue)/2, zdir='x'
        for point in hand_picked_shapes_points[i, :, :]:
            a.plot(point[1], point[2], '*', ms=6)
        a.plot(0, 0, 'k*', ms=6)
        a.text(.2, .75, '{:.1f} - {:.1f} m/s\n{:.1f}%'.format(le, ue, n_bin / n_total * 100), transform=a.transAxes)
        a.grid()
        a.clabel(cset, inline=True, fontsize=10)
        ax.append(a)
        # sns.kdeplot(x=x_sol[:, 1], y=x_sol[:, 2], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Blues"), 0.2, 1.), ax=ax[2])

        # clevels = np.array([1, 5, 10])/n_bins

        ax3d.contour(z, x, y, clevels, offset=(le+ue)/2, zdir='x', cmap=truncate_colormap(plt.cm.gist_earth_r, 0.05, 1.), norm=colors.LogNorm())

        # plot_frequency_projection(pc_data[mask, 0], pc_data[mask, 1], ax=ax2[i])
        # ax2[i].text(.2, .8, '{:.1f}%'.format(n_bin/n_total*100), transform=ax2[i].transAxes)
        # ax2[i].plot(hand_picked_shapes_pc[:, 0], hand_picked_shapes_pc[:, 1], '*')
        # plot_frequency_projection(pc_data[mask, 0], pc_data[mask, 1], ax=ax3[i], kde_plot=True)
        # ax3[i].text(.2, .8, '{:.1f}%'.format(n_bin/n_total*100), transform=ax3[i].transAxes)
        # ax3[i].plot(hand_picked_shapes_pc[:, 0], hand_picked_shapes_pc[:, 1], '*')

        mean_profile = np.mean(wind_speed[mask], axis=0)
        ax_means[0].plot(mean_profile, altitudes, label='{:.1f} - {:.1f} m/s'.format(le, ue))
        ax_means[1].plot(mean_profile / max(mean_profile), altitudes, label='{:.1f} - {:.1f} m/s'.format(le, ue))
    ax_means[1].legend()

    ax[0].get_shared_y_axes().join(*ax)
    ax[0].set_xlim([-14, 20])
    ax[0].set_ylim([-3.5, 8])

    ax.insert(0, ax3d)
    add_inside_panel_labels(np.array(ax))


def export_distributions_to_csv(altitudes, mean_profiles, rl, loc='mmca'):
    mean_profiles_200m = np.zeros(mean_profiles.shape)
    j_200 = np.argmax(altitudes == 200)
    for i, v in enumerate(mean_profiles):
        wind_speed_ref = v[j_200]
        mean_profiles_200m[i] = v / wind_speed_ref

    df = pd.DataFrame(mean_profiles_200m.T[1:, :], columns=[f'Cluster{i}' for i in range(1, len(mean_profiles)+1)])
    df.insert(0, 'Neutral log', log_law_wind_profile2(altitudes[1:], rl, 1., 200, 0.))
    df.insert(0, 'Heights', altitudes[1:])
    csv_file_path = f'cluster_shapes_{loc}.csv'  # Specify the file path where you want to save the CSV file
    df.to_csv(csv_file_path, index=False)  # Export the DataFrame as CSV, without writing the index


def plot_cluster_frequency(loc='mmca', clusters='', year=None, plot=False):
    if loc == 'mmca':
        rl = .03
    else:
        rl = .0002

    wind_data = read_data({'name': loc})
    wind_speed = (wind_data['wind_speed_east'] ** 2 + wind_data['wind_speed_north'] ** 2) ** .5
    if year is not None:
        mask_year = wind_data['datetime'].astype('datetime64[Y]').astype(int) + 1970 == year
        wind_speed = wind_speed[mask_year, :]
    n_samples = wind_speed.shape[0]
    print("No. of hourly samples {}".format(n_samples))
    j_200m = np.argmax(wind_data['altitude'] == 200)

    n_bins = 30
    vw_200m_bin_edges_log = np.linspace(0, 30, n_bins + 1)
    vw_200m_bin_centers_log = (vw_200m_bin_edges_log[1:] + vw_200m_bin_edges_log[:-1]) / 2
    # density_vw200m = np.histogram(wind_speed[:, j_200m], vw_200m_bin_edges_log, density=True)[0]
    # np.save('density_vw200m.npy', density_vw200m)

    # for j, file_name in enumerate(["x_cluster_weighted_{}.npy".format(loc)]):  #"x_cluster_{}.npy".format(loc),
    file_name = "wind_resource_fit/x_cluster{}_{}.npy".format(clusters, loc)
    x_cluster = np.load(file_name)
    x_cluster_log = np.load("wind_resource_fit/x_cluster0_{}.npy".format(loc))
    if year is not None:
        x_cluster = x_cluster[mask_year, :]
        x_cluster_log = x_cluster_log[mask_year, :]

    n_clusters = int(np.amax(x_cluster[:, 0]))+1
    cluster_labels = ['Log'] + ['Cluster {}'.format(i) for i in range(1, n_clusters)]

    # vw_200m_bin_edges_export = np.linspace(0, 30, 60 + 1)
    vw_200m_bin_edges_export = np.arange(0, 30, .6)
    vw_200m_bin_edges_export = np.round(vw_200m_bin_edges_export, 1)
    vw_200m_bin_centers_export = (vw_200m_bin_edges_export[1:] + vw_200m_bin_edges_export[:-1]) / 2

    if plot:
        plt.figure(figsize=[7, 7])
        plt.subplots_adjust(top=0.92, bottom=0.07, left=0.125, right=0.925, hspace=0.17, wspace=0.13)
        ax0 = plt.subplot(421)
        ax0.set_xlim([0, 27.5])

        n = ax0.hist(wind_speed[:, j_200m], vw_200m_bin_edges_log, alpha=.2, weights=[100/wind_speed.shape[0]]*wind_speed.shape[0])[0]
        ax0.plot(vw_200m_bin_centers_log, n, color='C0', label='Wind atlas')
        n = ax0.hist(x_cluster[:, 1], vw_200m_bin_edges_log, alpha=.2, weights=[100/wind_speed.shape[0]]*wind_speed.shape[0])[0]
        ax0.plot(vw_200m_bin_centers_log, n, color='C1', label='Aggregate of clusters')
        n = ax0.hist(x_cluster_log[:, 1], vw_200m_bin_edges_log, alpha=.2, weights=[100/wind_speed.shape[0]]*wind_speed.shape[0])[0]
        ax0.plot(vw_200m_bin_centers_log, n, color='C2', label='Log fit')
        
        ax0.set_ylabel('Bin frequency [%]')
        # ax0.legend()
        ax0.legend(bbox_to_anchor=(0, 1.05, 1., .2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
        ax0.xaxis.set_visible(False)

        ax_hist = []
        ax_power_curve = []
        frequency = []
        frequency_effective = []
    p_avg = 0

    bin_freq_export = np.zeros((vw_200m_bin_edges_export.shape[0]-1, n_clusters))
    for i in range(n_clusters):
        # Load optimized power curves and set the wind speed distribution limits accordingly
        df = pd.read_csv('opt_res_{}/opt_res_{}{}.csv'.format(loc, loc, i+1))
        vw_200m_bin_edges = np.linspace(df['vw200'].iloc[0], df['vw200'].iloc[-1], n_bins+1)
        vw_200m_bin_centers = (vw_200m_bin_edges[1:] + vw_200m_bin_edges[:-1])/2
        mask = x_cluster[:, 0] == i
        bin_freq = np.histogram(x_cluster[mask, 1], vw_200m_bin_edges)[0]

        # the weights argument makes sure that each bin quantifies the frequency of the total population
        bin_freq_export[:, i] = np.histogram(x_cluster[mask, 1], vw_200m_bin_edges_export)[0]*100/wind_speed.shape[0]

        if plot:
            ax = plt.subplot(4, 2, i + 2)
            ax.xaxis.set_visible(False)
            ax.set_ylim([0, 2])
            ax.set_xlim([0, 27.5])

            # in-between cut-in and cut-out
            ax.hist(x_cluster[mask, 1], vw_200m_bin_edges, alpha=.5, weights=[100/wind_speed.shape[0]]*np.sum(mask))
            ax.axvline(df['vw200'].iloc[0], ls='--', color='grey')
            ax.axvline(df['vw200'].iloc[-1], ls='--', color='grey')

            frequency.append(np.sum(mask)/n_samples*100)
            frequency_effective.append(np.sum(bin_freq)/n_samples*100)

            width_bin = (df['vw200'].iloc[-1]-df['vw200'].iloc[0])/n_bins
            vw_200m_bin_edges_below = np.arange(df['vw200'].iloc[0], 0, -width_bin)[::-1]
            ax.hist(x_cluster[mask, 1], vw_200m_bin_edges_below, color='C1', alpha=.5, weights=[100/wind_speed.shape[0]]*np.sum(mask))
            vw_200m_bin_edges_above = np.arange(df['vw200'].iloc[-1], np.amax(x_cluster[mask, 1]), width_bin)
            ax.hist(x_cluster[mask, 1], vw_200m_bin_edges_above, color='C1', alpha=.5, weights=[100/wind_speed.shape[0]]*np.sum(mask))
            ax.text(0.03, 0.7, cluster_labels[i]+'\n{:.1f}%'.format(sum(mask)/x_cluster.shape[0]*100), transform=ax.transAxes)
            axt = ax.twinx()
            if i % 2 == 1:
                axt.yaxis.set_visible(False)
                ax.set_ylabel('Bin frequency [%]')
            else:
                ax.yaxis.set_visible(False)
                axt.set_ylabel('Power [kW]')
            axt.plot(df['vw200'], df['mcp']*1e-3, ls='-')

            ax_power_curve.append(axt)
            ax_hist.append(ax)

            for a in ax_hist[5:]:
                a.set_xlabel(r'$v_{\rm w,200m}$ [m/s]')
                a.xaxis.set_visible(True)
        
        bin_power = np.interp(vw_200m_bin_centers, df['vw200'], df['mcp'], left=0., right=0.)

        p_avg_bin = np.sum(bin_freq * bin_power / n_samples)
        p_avg += p_avg_bin
    print("Average power {:.2f}".format(p_avg*1e-3))

    df = pd.DataFrame(bin_freq_export, columns=cluster_labels)
    df.insert(0, 'Lower', vw_200m_bin_edges_export[:-1])
    df.insert(1, 'Upper', vw_200m_bin_edges_export[1:])
    df.columns = ['Lower', 'Upper', 'Log', 'Cluster1', 'Cluster2', 'Cluster3', 'Cluster4', 'Cluster5', 'Cluster6']

    csv_file_path = f'cluster_frequency_{loc}.csv'  # Specify the file path where you want to save the CSV file
    df.to_csv(csv_file_path, index=False)

    if plot:
        ax_power_curve[0].get_shared_y_axes().join(*ax_power_curve)
        ax_hist[0].get_shared_x_axes().join(*ax_hist)
        ax_hist.insert(0, ax0)
        add_panel_labels(np.array(ax_hist), [.33, .11]*4)

        plt.figure()
        plt.bar(range(1, n_clusters+1), frequency, color=['k']+['C{}'.format(i) for i in range(n_clusters-1)], alpha=.4)
        plt.bar(range(1, n_clusters+1), frequency_effective, color=['k']+['C{}'.format(i) for i in range(n_clusters-1)])

    return p_avg


def plot_distr_pc12(loc='mmij'):
    x_sol = np.load("x_sol_{}.npy".format(loc))
    sns.kdeplot(x=x_sol[:, 1], y=x_sol[:, 2], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Blues"), 0.2, 1.)) # levels=lvls, cmap=cmap_grey, ax=ax_col[0])


def plot_results_pc1(loc='mmij'):
    x_sol = np.load("x_sol_{}.npy".format(loc))
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    sns.kdeplot(x=x_sol[:, 0], y=x_sol[:, 1], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Blues"), 0.2, 1.), ax=ax[0, 0]) # levels=lvls, cmap=cmap_grey, ax=ax_col[0])
    mask = x_sol[:, 2] > 1
    sns.kdeplot(x=x_sol[mask, 0], y=x_sol[mask, 1], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Oranges"), 0.2, 1.), ax=ax[1, 0]) # levels=lvls, cmap=cmap_grey, ax=ax_col[0])
    sns.kdeplot(x=x_sol[~mask, 0], y=x_sol[~mask, 1], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Oranges"), 0.2, 1.), ax=ax[1, 1]) # levels=lvls, cmap=cmap_grey, ax=ax_col[0])
    # plt.xlim([0, 27])
    # plt.ylim([-5, 8])


def plot_results_pc2(loc='mmij'):
    x_sol = np.load("x_sol_{}.npy".format(loc))
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    sns.kdeplot(x=x_sol[:, 0], y=x_sol[:, 2], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Blues"), 0.2, 1.), ax=ax[0, 0]) # levels=lvls, cmap=cmap_grey, ax=ax_col[0])
    mask = x_sol[:, 1] > 0
    sns.kdeplot(x=x_sol[mask, 0], y=x_sol[mask, 2], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Oranges"), 0.2, 1.), ax=ax[1, 0]) # levels=lvls, cmap=cmap_grey, ax=ax_col[0])
    sns.kdeplot(x=x_sol[~mask, 0], y=x_sol[~mask, 2], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Oranges"), 0.2, 1.), ax=ax[1, 1]) # levels=lvls, cmap=cmap_grey, ax=ax_col[0])
    # plt.xlim([0, 27])
    # plt.ylim([-5, 8])


def aep_convergence(loc='mmij', ax=None):
    if loc == 'mmca':
        cluster_combinations = ['0', '0123', '01234', '012346', '']  #['0', '012', '03', '034', '0134', '01345', '']
    else:
        cluster_combinations = ['0', '0123', '01235', '012356', '']

    li_s = []
    for i in range(4):
        s = pd.read_csv('opt_res_{}/succeeded{}.csv'.format(loc, i))
        li_s.append(s)

    s = pd.read_csv('opt_res_{}/succeeded_att2.csv'.format(loc))
    li_s.append(s)
    succeeded = pd.concat(li_s, axis=0, ignore_index=True)
    brute_force_p_2008 = succeeded['mcp'].sum() / 8784

    n_combs = len(cluster_combinations)
    p_avg_convergence = np.zeros((11, n_combs))
    for i, clstr_combi in enumerate(cluster_combinations):
        p_avg = plot_cluster_frequency(loc, clstr_combi)
        p_avg_convergence[0, i] = p_avg
        for j, y in enumerate(range(2008, 2018)):
            p_avg = plot_cluster_frequency(loc, clstr_combi, y)
            p_avg_convergence[j+1, i] = p_avg

    if ax is None:
        add_legend = True
        ax = plt.figure().gca()
    else:
        add_legend = False
    ax.plot(p_avg_convergence[0, :]*8760*1e-6, 's-', label='2008-2017', ms=3)
    for j, y in enumerate(range(2008, 2018)):
        if y == 2008:
            alpha = 1
            ls = '.-'
        else:
            alpha = .5
            ls = '-'
        ax.plot(p_avg_convergence[j+1, :]*8760*1e-6, ls, label=y, alpha=alpha)
    ax.set_xticks(range(n_combs))
    ax.set_xticklabels(['Log', 'Set #1', 'Set #2', 'Set #3', 'All'], fontdict={'fontsize': 'small'})

    ax.axhline(brute_force_p_2008*8760*1e-6, ls='--', color='C1')
    ax.set_xlabel('Wind profile shapes')

    if add_legend:
        ax.legend()

    fig, ax = plt.subplots(2, 1)
    steps = (p_avg_convergence[:, 1:] - p_avg_convergence[:, :-1])*8760*1e-6
    steps_perc = (p_avg_convergence[:, 1:] - p_avg_convergence[:, :-1])/p_avg_convergence[:, 1:] * 100
    ax[0].plot(steps[0, :], 's')
    ax[0].plot(steps[1:, :].T)
    ax[1].plot(steps_perc[0, :], 's')
    ax[1].plot(steps_perc[1:, :].T)
    ax[0].set_ylabel('Step in AEP [MWh]')
    ax[1].set_ylabel('Step in AEP [%]')

    plt.figure()
    plt.plot(p_avg_convergence[1, :]/brute_force_p_2008*100)


def plot_aep():
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[7, 4])
    wspace = .2
    plt.subplots_adjust(left=.115, bottom=.121, right=.983, top=.75, hspace=.274, wspace=wspace)

    for j, loc in enumerate(['mmca', 'mmij']):
        aep_convergence(loc, ax[j])

    ax[0].legend(bbox_to_anchor=(.3, 1.13, 1.6, .2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
    ax[0].set_ylabel('Annual energy production [MWh]')
    ax[0].set_title('Onshore')
    ax[1].set_title('Offshore')
    for a in ax.reshape(-1):
        a.grid()
        a.set_ylim([30, 70])
    add_panel_labels(ax, [.29, .13])

def assign_to_cluster_for_all_combinations():
    from multiprocessing import Process
    cluster_combinations = {
        'mmca': [],  #['0', '0123', '01234', '012346', 'all']  #
        'mmij': ['01235', '012356'],
    }
    processes = []
    for loc in ['mmca', 'mmij']:
        for set in cluster_combinations[loc]:
            if set != 'all':
                set = [int(v) for v in list(set)]
            p = Process(target=assign_to_cluster, args=(loc, set))
            processes.append(p)
            p.start()
    for p in processes:
        p.join()


if __name__ == '__main__':
    # assign_to_cluster_for_all_combinations()
    # plot_aep()
    loc = 'mmij'
    plot_cluster_frequency(loc, plot=False)
    # plot_cluster_frequency(loc, '0')
    # plot_cluster_frequency(loc, '012')

    # eval_loc(loc)
    # plot_3d_results(loc)
    # plot_distr_pc12()
    # plot_results_pc1(loc)
    # plot_results_pc2(loc)
    plt.show()