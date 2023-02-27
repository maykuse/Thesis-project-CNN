import numpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import string
import scipy


params = ["t2m", "u10", "v10", 'z', 't', "tcc", "tp"]
# params_x = ["t2m", "v10", 'z', 't', "tcc", "tp"]
params_C = ["T2M", "U10", "V10", 'Z', 'T', "TCC", "TP"]

if (draw_world_map):
    for l in range(6):
        fig = plt.figure(figsize=(10, 10))
        sns.set(font_scale=2.2)
        sns.heatmap(results[0][l], cmap="RdBu", xticklabels=False, yticklabels=False, center=0.00, vmin=-lrp_max[k],
                    vmax=lrp_max[k],
                    cbar_kws=dict(use_gridspec=False, orientation="horizontal"))
        plt.title("\n".join(wrap(
            "{param} Prediction Heatmap LRP with respect to input parameter {par}".format(param=params_C[k],
                                                                                          par=all_labels[k][l]), 35)))
        # plt.title("{param} Prediction model World heatmap of Gradients wrt. p_{par} when p_all ~ 1".format(param=params_C[k], par=all_labels[k][l]))
        plt.show()
        plt.tight_layout()
        fig.savefig(
            '/home/ge75tis/Desktop/LRP180/{param}_world_heatmap_{par}'.format(param=params_C[k], par=all_labels[k][l]))

if (draw_grad_bar):
    x = np.arange(len(all_labels[k]))

    width = 0.3
    fig3, ax = plt.subplots()
    std = torch.std(std_tensor, dim=1)
    # std = torch.mul(std, 1000)
    std = torch.div(std, 10000)
    print(results)
    print(std)

    # print(std)
    rects1 = ax.bar(x, results, width, yerr=std, capsize=4)
    ax.set_ylabel('Avg. Gradient over Validation data')
    ax.set_xlabel('parameter x')
    ax.set_title('{param} Loss Gradients w.r.t. p_{par} ~ 0 and p_x ~ 0 when p_others ~ 1'.format(param=params_C[k],
                                                                                                  par=params[
                                                                                                      set_to_zero[k]]),
                 fontsize=16)
    ax.set_xticks(x, all_labels[k])
    ax.tick_params(axis='x', which='major', labelsize=16)
    # ax.bar_label(rects1, padding=3)
    fig3.tight_layout()
    plt.show()
    # fig3.savefig('/home/ge75tis/Desktop/{param}_gradient_bar_chart'.format(param=params_C[k]))

    if (graph):
        if (month):
            x_axis = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
            labels = all_labels[k]
            fig = plt.figure()
            for i in range(6):
                plt.errorbar(avg_val_loss_gridded_m[k][i], yerr=0, label=labels[i])
            fig.suptitle('{param} analysis, p_x ~ 0, p_other ~ 1 per month'.format(param=params_C[k]))
            fig.errorbar
            plt.legend()
            plt.xlabel('months')
            plt.ylabel('Average loss')
            fig.savefig(
                "/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/per_month_test/{param}_Dropout_per_month_p_x_0".format(
                    param=params_C[k]))
        # fig = plt.figure(g, figsize=(10,10))
        # sns.heatmap(avg_val_loss_gridded[g], linewidths=.5, cmap="Greens", annot=True, xticklabels=grid, yticklabels=grid, norm=LogNorm(), fmt=".3f")
        # labels = ["t2m", "u10", "v10", 'z', 't', "tcc"]
        # plt.title('Avg Validation loss of TP for different dropout rates of {par} and other parameters'.format(par=labels[g]))
        # plt.xlabel('other parameters dropout rate p')
        # plt.ylabel('{par} dropout rate p'.format(par=labels[g]))
        # fig.savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/DROPOUT_ANALYSIS/tp_analysis_{label}_heatmap'.format(label=labels[g]))

    # results = np.array(p_gradients.cpu())
results = rec_losses[k][0]

if (gradient_bars):
    # print(f' {labels[k]}: p_grads, {all_labels[k]} p ~ 0, others ~ 1: {results}')
    x = np.arange(len(all_labels[k]))
    width = 0.3
    fig3, ax = plt.subplots()
    rects1 = ax.bar(x, results, width, label='p_{param} ~ 0 and p_x ~ 0, p_others ~ 1')
    ax.set_ylabel('Avg. Validation Loss')
    ax.set_xlabel
    ax.set_title(
        '{param} Dropout Analysis when p_{par} ~ 0 and p_x ~ 0, p_others ~ 1'.format(param=params_C[k],
                                                                                     par=params[params_zero[k]]))
    ax.set_xticks(x, all_labels[K])
    ax.legend()
    # ax.bar_label(rects1, padding=3)
    fig3.tight_layout()
    fig3.savefig(
        '/home/ge75tis/Desktop/{param}_dropout_bar_chart'.format(
            param=params_C[k]))





















params_g = [['u10', 'v10', 'z', 'tcc', 'tp'], ['t2m', 'z', 't', 'tcc', 'tp'],['t2m', 'z', 't', 'tcc', 'tp'],
              ['t2m', 'u10', 't', 'tcc', 'tp'], ['u10', 'v10', 'z', 'tcc', 'tp'], ['t2m', 'u10', 'z', 't', 'tp'], ['t2m', 'u10', 'z', 't', 'tcc']]

bar_chart_std = False
params_zer = [4, 2, 1, 2, 0, 2, 2]
if(bar_chart_std):
    for i in range(7):
        grads = [[70, 65, 105, 41, 25], [26, 391, 232, 83, 282], [100, 341, 248, 181, 461],  [98, 227, 216, 50, 38],
                 [118, 115, 169, 64, 20], [37, 234, 116, 117, 241], [0.25, 31, 5, 0.15, 27]]

        stds = [[0.1091, 0.1478, 0.2207, 0.0671, 0.0525], [0.0333, 0.5026, 0.3446, 0.1797, 0.4037], [0.1337,  0.4858, 0.3542, 0.2951, 0.6118]
                , [1.8661e-01, 5.7324e-01, 4.5843e-01, 7.0172e-02, 1.7575e-01], [0.1302, 0.1501, 0.2760, 0.0931, 0.0278], [0.0214, 0.1180,0.0981, 0.0634, 0.1965],
                [1.2607e-03, 1.7842e-02, 6.5283e-03, 1.8910e-03, 1.5993e-02]]

        clm_losses = [0.1061, 0.5411, 0.6394, 0.2035, 0.1897, 0.6936, 0.354]

        x = np.arange(5)
        width = 0.3
        plt.rcParams.update({'font.size': 25})
        fig, ax = plt.subplots(figsize=(11,8))
        # low = min(unet_losses)
        # high = max(unet_losses)
        # plt.ylim(0.24, 0.28)
        rects1 = ax.bar(x, grads[i], width, yerr=stds[i], capsize=6, label="Gradient of {param} loss for p_{best} ~ 0 and p_others ~ 1".format(param=params_C[i], best=params_C[params_zer[i]]))
        # rects2 = ax.bar(x + width/2, clm_losses, width, label="Climatology")
        ax.set_ylabel('Average Gradient')
        ax.set_xticks(x, params_g[i])
        ax.set_xlabel('input parameters')
        ax.set_title('UNet(5-to-1) prediction when one additional input is ignored')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
                  ncol=3, fancybox=True, shadow=True)

        # ax.bar_label(rects1, padding=3)
        # ax.bar_label(rects2, padding=3)

        fig.tight_layout()
        plt.show()
        fig.savefig('/home/ge75tis/Desktop/higher_order_grad_{param}'.format(param=params_C[i]))

    # set_to_zeros = [4, 2, 1, 2, 0, 2, 2]
    # for i in range(7):
    #     loss_p = rec_losses[i][0]
    #     std_p = rec_losses[i][1]
    #     loss1_p = rec_losses1[i][0]
    #     std1_p = rec_losses1[i][1]
    #     x = np.arange(len(all_labels[i]))
    #     width = 0.3
    #     fig3, ax = plt.subplots()
    #     rects1 = ax.bar(x - width/2, loss_p, width, yerr=std_p, capsize=4, label='p_{aa} ~ 0 and p_y ~ 0, p_other ~ 1'.format(param=params_C[i]), aa=params[set_to_zeros[i]])
    #     # rects2 = ax.bar(x + width/2, loss1_p, width, yerr=std1_p, capsize=4, label='p_x ~ 1, p_other ~ 0'.format(param=params_C[i]))
    #     ax.set_ylabel('Average validation losses')
    #     ax.set_xlabel('parameter y')
    #     ax.set_title('{param} Dropout Analysis noise comparison when p_{aa} ~ 0'.format(param=params_C[i]), aa=params[set_to_zeros[i]])
    #     ax.set_xticks(x, all_labels[i])
    #     ax.legend()
    #     # ax.bar_label(rects1, padding=3)
    #     fig3.tight_layout()
    #     plt.show()
    #     fig3.savefig('/home/ge75tis/Desktop/double_dropout_bar_chart_with_stds_{param}'.format(param=params_C[i]))




distance_cluster = False
if(distance_cluster):
    comb_heatmap_percentage = [[0, 0.2372, 0.3441, 0.5068, 0.5326, 0.1715, 0.0742], [0.0130, 0, 0.6495, 0.4286, 0.1589, 0.1426, 0.2390],
                                   [0.0774, 0.7039, 0, 0.4368, 0.1069, 0.1861, 0.1784], [0.1943, 0.3980, 0.2435, 0, 0.5723, 0.0394, 0.0959],
                                   [0.7112, 0.0709, 0.1207, 0.5341, 0, 0.0857, 0.0063], [0.0351, 0.1879, 0.3602, 0.1391, 0.2310, 0, 0.1881],
                                   [0.0210, 0.4211, 0.4670, 0.0889, 0.0503, 0.2109, 0]]

    grad_heatmap = [[0, 37.1591, 53.27599, 23.071262, 369.16653, 10, 10],
                    [18.599178, 0, 506.08084, 239.3966, 252.6608, 99.09486, 122.54916],
                    [89.44, 419.50064, 0, 200.04277, 229.186, 160.98209, 173.02988],
                    [69.64928, 222.90337, 303.3208, 0, 192.48601, 38.59495, 34.078014],
                    [243.56503, 70.133736, 118.08318, 162.23822, 0, 48.33932, 16.240177],
                    [20.356546, 117.38441, 187.36948, 116.907524, 100.78966, 0, 154.77966],
                    [1, 10.919211, 24.24271, 3.0331063, 1, 16.8139, 0]]
    # how to deal with the really low gradients of tcc and tp? the distances become out of scale compared to others

    dist_matr = np.empty([7,7])
    dist_matr = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(comb_heatmap_percentage))
    # for i in range(7):
    #     for j in range(7):
    #         if(i == j):
    #             dist_matr[i][j] = 0
    #         else:
    #             dist_matr[i][j] *= (1 / (grad_heatmap[i][j] + grad_heatmap[j][i]) ) * 1000

    print(dist_matr)

    dt = [('len', float)]
    dist_matr = dist_matr.view(dt)

    G = nx.from_numpy_matrix(dist_matr)
    G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),string.ascii_uppercase)))
    G = nx.drawing.nx_agraph.to_agraph(G)
    G.node_attr.update(color="red", style="filled")
    G.edge_attr.update(color="white", width="0.0")
    fig = plt.figure()
    G.draw('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/a_cluster2.png', format='png', prog='neato')
    # plt.savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/cluster2.png', format="PNG")


heatmap = False
if(heatmap):
    new_param_order = ['t', 't2m', 'z', 'v10', 'u10', 'tp', 'tcc']
    fig1 = plt.figure(figsize=(12,12))

    comb_heatmap_percentage = [[1, 0.2372, 0.3441, 0.5068, 0.5326, 0.1715, 0.0742], [0.0130, 1, 0.6495, 0.4286, 0.1589, 0.1426, 0.2390],
                               [0.0774, 0.7039, 1, 0.4368, 0.1069, 0.1861, 0.1784], [0.1943, 0.3980, 0.2435, 1, 0.5723, 0.0394, 0.0959],
                               [0.7112, 0.0709, 0.1207, 0.5341, 1, 0.0857, 0.0063], [0.0351, 0.1879, 0.3602, 0.1391, 0.2310, 1, 0.1881],
                               [0.0210, 0.4211, 0.4670, 0.0889, 0.0503, 0.2109, 1]]
    comb_heatmap = [[0, 4.30, 3.65, 2.77, 2.60, 4.65, 5.20], [11.23, 0, 3.94, 6.46, 9.55, 9.75, 8.53],
                    [16.08, 5.06, 0, 9.83, 15.68, 14.13, 14.36],
                    [10.56, 7.81, 9.87, 0, 5.57, 12.53, 11.78], [2.75, 8.91, 8.42, 4.52, 0, 8.77, 9.50],
                    [4.77, 4.06, 3.15, 4.23, 3.77, 0, 3.98], [1.09, 0.65, 0.59, 1.00, 1.05, 0.87, 0]]
    #, -18.684557, -9.69056 ]
    grad_heatmap = [[0, 37.1591, 53.27599, 23.071262, 369.16653, -18.684557, -9.69056], [18.599178, 0, 506.08084, 239.3966, 252.6608, 99.09486, 122.54916 ],
                    [89.44, 419.50064, 0, 200.04277, 229.186, 160.98209, 173.02988], [69.64928, 222.90337, 303.3208, 0, 192.48601, 38.59495, 34.078014],
                    [243.56503, 70.133736, 118.08318, 162.23822, 0, 48.33932, 16.240177], [20.356546, 117.38441, 187.36948, 116.907524, 100.78966, 0, 154.77966],
                    [0.11764815, 10.919211, 24.24271, 3.0331063, 0.18807021, 16.8139, 0]]

    grad_heatmap2 = [[0, 37.1591, 53.27599, 23.071262, 369.16653, 1, 1], [18.599178, 0, 506.08084, 239.3966, 252.6608, 99.09486, 122.54916 ],
                    [89.44, 419.50064, 0, 200.04277, 229.186, 160.98209, 173.02988], [69.64928, 222.90337, 303.3208, 0, 192.48601, 38.59495, 34.078014],
                    [243.56503, 70.133736, 118.08318, 162.23822, 0, 48.33932, 16.240177], [20.356546, 117.38441, 187.36948, 116.907524, 100.78966, 0, 154.77966],
                    [0.11764815, 10.919211, 24.24271, 3.0331063, 0.18807021, 16.8139, 0]]

    grad_heatmap_new = [[0, 0.36795822, 0.43980718, 0.61094254, 0.66767836, 0.25484586, 0.15239869], [0.09645872, 0, 3.8159883,  1.0963811, 1.3558255,  0.1850218, 0.49045599],
                        [0.97554576, 6.065276,  0, 2.6048186,  1.7355888,  1.7813991,  0.6647806], [1.5544034, 3.0572004, 3.1909885, 0, 4.1371703, 0.4340603, 0.7173816],
                        [4.782727,   0.28787705, 0.8700743,  2.5207248, 0, 0.62985164, 0.08375259], [0.1033018,  0.3755193, 0.7591337,  0.58453566, 0.48831362, 0, 0.65765136],
                        [0.00877767, 0.09491006, 0.12384017, 0.04167771,  0.02347502, 0.07231351, 0]]

    a = numpy.array(grad_heatmap_new)
    row_sums = a.sum(axis=1)
    new_matrix = a / row_sums[:, numpy.newaxis]
    # print(new_matrix)

    # grad_heatmap_new =

    heat_norm = plt.Normalize(0,1)
    dist_matr = np.empty([7, 7])
    for i in range(7):
        for j in range(7):
            if (i == j):
                dist_matr[i][j] = 0
            else:
                dist_matr[i][j] = (1 / (new_matrix[i][j] + new_matrix[j][i]))

    # print(dist_matr)

    sns.set(font_scale=1.2)
    # sns.heatmap(grad_heatmap, linewidths=.5, cmap="magma", annot=True, xticklabels=params_C, yticklabels=params_C, norm=LogNorm(), fmt='.3g')
    # fig1 = sns.clustermap(comb_heatmap_percentage, cbar_kws={"shrink": 0.5}, method='average', linewidths=1, linecolor='white', row_linkage=scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dist_matr)), col_linkage=scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dist_matr)), cmap="magma",  annot=True, xticklabels=params, yticklabels=params, norm=heat_norm, fmt='.3g')
    # fig1.ax_heatmap.set_xticklabels(fig1.ax_heatmap.get_xmajorticklabels(), fontsize = 22)
    # fig1.ax_heatmap.set_yticklabels(fig1.ax_heatmap.get_ymajorticklabels(), fontsize = 22)

    # Why does the clustermap change tcc and tp's position, even though they should lie at the end
    # plt.title('Gradient of prediction loss wrt. input parameters when p_all ~ 1')
    # plt.xlabel('input parameters (gradient)')
    # plt.ylabel('predicted parameter')
    # plt.title('Gradient of prediction loss with respect to p of each parameter when p_all ~ 1', loc='center', wrap=True)
    # plt.xlabel('input parameters')
    # plt.ylabel('predicted parameter')
    fig = plt.figure()
    # plt.tight_layout()
    # plt.show()
    fig1.savefig('/home/ge75tis/Desktop/Thesis-project-CNN/b')


    dist = scipy.spatial.distance.squareform(dist_matr)
    links = scipy.cluster.hierarchy.linkage(dist, "average")
    scipy.cluster.hierarchy.dendrogram(links, labels=params_C)

    plt.title("Hierarchical Clustering of Parameters using Gradients (avg.)")
    plt.ylabel("distance")
    # plt.show()

    # plt.savefig('/home/ge75tis/Desktop/newgrad')

# row_linkage=scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dist_matr))




