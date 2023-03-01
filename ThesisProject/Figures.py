import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import string
import scipy
from textwrap import wrap


params_C = ["T2M", "U10", "V10", 'Z', 'T', "TCC", "TP"]
params = ["t2m", "u10", "v10", 'z', 't', "tcc", "tp"]

all_labels = [["u10", "v10", 'z', 't', "tcc", "tp"], ["t2m", "v10", 'z', 't', "tcc", "tp"],
              ["t2m", "u10", 'z', 't', "tcc", "tp"], ["t2m", "u10", "v10", 't', "tcc", "tp"],
              ["t2m", "u10", "v10", 'z', "tcc", "tp"], ["t2m", "u10", "v10", 'z', 't', "tp"],
              ["t2m", "u10", "v10", 'z', 't', "tcc"]]


def Loss_Comparison(results):
    clm = [0.1061, 0.5411, 0.6394, 0.2035, 0.1897, 0.6936, 0.3540]

    x = np.arange(7)
    width = 0.2
    fig, ax = plt.subplots()

    rects0 = ax.bar(x - width, results[0], width, label="Many-to-one Loss with no dropout")
    rects1 = ax.bar(x, results[1], width, label="Many-to-one Loss with dropout")
    rects2 = ax.bar(x + width, clm, width, label="Climatology Loss")

    ax.set_ylabel('Average Loss')
    ax.set_xticks(x, params_C)
    ax.set_title('Many-to-one Losses with/without dropout vs. Climatology')

    ax.legend()
    fig.tight_layout()
    plt.show()
    fig.savefig('/home/ge75tis/Desktop/a')


#     a = numpy.array(grad_heatmap_new)
#     row_sums = a.sum(axis=1)
#     new_matrix = a / row_sums[:, numpy.newaxis]
#     # print(new_matrix)

#     # plt.title('Gradient of prediction loss wrt. input parameters when p_all ~ 1')
#     # plt.xlabel('input parameters (gradient)')
#     # plt.ylabel('predicted parameter')
#     # plt.title('Gradient of prediction loss with respect to p of each parameter when p_all ~ 1', loc='center', wrap=True)
#     # plt.xlabel('input parameters')
#     # plt.ylabel('predicted parameter')


def Similarity_Graphs(results, fig_type):
    fig1 = plt.figure(figsize=(12, 12))

    heatmap = np.empty([7,7])
    heat_norm = plt.Normalize(0, 1)

    for i in range(7):
        count = 0
        for j in range(7):
            if(i == j):
                heatmap[i][j] = 1
            else:
                if(fig_type == 0):
                    heatmap[i][j] = (results[0][i][6] - results[0][i][count]) / results[0][i][6]
                else:
                    heatmap[i][j] = results[i][count]
                count += 1

    # a = numpy.array(heatmap)
    # row_sums = a.sum(axis=1)
    # new_matrix = a / row_sums[:, numpy.newaxis]

    dist_matr = np.empty([7, 7])

    for i in range(7):
        for j in range(7):
            if (i == j):
                dist_matr[i][j] = 0
            else:
                dist_matr[i][j] = (1 / (heatmap[i][j] + heatmap[j][i])) * 10000




    if(fig_type == 0):
        sns.set(font_scale=1.2)
        fig1 = sns.clustermap(heatmap, cbar_kws={"shrink": 0.5}, linewidths=1, linecolor='white', row_linkage=scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dist_matr), "average"),
            col_linkage=scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dist_matr), "average"), cmap="magma",  annot=True, xticklabels=params, yticklabels=params, norm=heat_norm, fmt='.3g')
        fig1.ax_heatmap.set_xticklabels(fig1.ax_heatmap.get_xmajorticklabels(), fontsize=22)
        fig1.ax_heatmap.set_yticklabels(fig1.ax_heatmap.get_ymajorticklabels(), fontsize=22)

        # # plt.title('Percentage Decrease (compared to p_all ~ 1) in prediction loss when p_x ~ 0, p_others ~ 1')
        # plt.xlabel('input parameter x')
        # plt.ylabel('predicted parameter')
        plt.tight_layout()
        fig1.savefig('/home/ge75tis/Desktop/Heatmap')


    dist = scipy.spatial.distance.squareform(dist_matr)
    links = scipy.cluster.hierarchy.linkage(dist, "average")
    scipy.cluster.hierarchy.dendrogram(links, labels=params_C)

    if(fig_type == 0):
        plt.title("Hierarchical Clustering (Loss based avg. distance)")
    else:
        plt.title("Hierarchical Clustering (Gradient based avg. distance)")
    plt.ylabel("distance")
    plt.savefig('/home/ge75tis/Desktop/dendrogram')


    G = nx.from_numpy_matrix(dist_matr)
    G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), string.ascii_uppercase)))
    G = nx.drawing.nx_agraph.to_agraph(G)
    G.node_attr.update(color="red", style="filled")
    G.edge_attr.update(color="white", width="0.0")
    G.draw('/home/ge75tis/Desktop/distance', format='png', prog='neato')



def Grad_Bar_Chart(results):
    for k in range(7):
        x = np.arange(len(all_labels[k]))
        width = 0.3
        fig2, ax = plt.subplots()

        rects0 = ax.bar(x, results[k], width)

        ax.set_ylabel('Avg. Gradient')
        ax.set_xlabel('Input parameters')
        ax.set_title('Gradient of {par} Loss w.r.t. p of each input parameter when p_all ~ 1'.format(par=params_C[k]))
        ax.set_xticks(x, all_labels[k])
        ax.legend()
        fig2.tight_layout()
        fig2.savefig('/home/ge75tis/Desktop/{param}_grad_bar_chart'.format(param=params_C[k]))


def Grad_Bar_Chart_Multi(results):
    for k in range(7):
        for l in range(6):
            x = np.arange(len(all_labels[k]))
            width = 0.3
            fig2, ax = plt.subplots()

            rects0 = ax.bar(x, results[k], width)

            ax.set_ylabel('Avg. Gradient')
            ax.set_xlabel('Input parameters')
            ax.set_title(
                'Gradient of {par} Loss w.r.t. p of each input parameter when p_{x} ~ 0 and p_others ~ 1'.format(par=params_C[k], x=all_labels[k][l]))
            ax.set_xticks(x, all_labels[k])
            ax.legend()
            fig2.tight_layout()
            fig2.savefig('/home/ge75tis/Desktop/{param}_grad_bar_chart_p_{par}0'.format(param=params_C[k], par=all_labels[k][l]))


def Grad_World_Maps(results):
    maxes = [0]
    for k in range(7):
        for l in range(6):
            fig = plt.figure(figsize=(10, 10))
            sns.set(font_scale=2.2)
            sns.heatmap(results[k][l], cmap="RdBu", xticklabels=False, yticklabels=False, center=0.00, vmin=-maxes[k],
                        vmax=maxes[k], cbar_kws=dict(use_gridspec=False, orientation="horizontal"))
            plt.title("{param} Prediction model World heatmap of Gradients wrt. p_{par} when p_all ~ 1".format(param=params_C[k], par=all_labels[k][l]))
            plt.tight_layout()
            fig.savefig('/home/ge75tis/Desktop/LRP180/{param}_world_heatmap_{par}'.format(param=params_C[k], par=all_labels[k][l]))


def LRP_World_Maps(results):
    maxes = [0]
    for k in range(7):
        for l in range(6):
            fig = plt.figure(figsize=(10, 10))
            sns.set(font_scale=2.2)
            sns.heatmap(results[k][l], cmap="RdBu", xticklabels=False, yticklabels=False, center=0.00, vmin=-maxes[k],
                        vmax=maxes[k], cbar_kws=dict(use_gridspec=False, orientation="horizontal"))
            plt.title("\n".join(wrap("{param} Prediction Heatmap LRP with respect to input parameter {par}".
                                     format(param=params_C[k],par=all_labels[k][l]), 35)))
            plt.tight_layout()
            fig.savefig('/home/ge75tis/Desktop/LRP180/{param}_world_heatmap_{par}'.format(param=params_C[k], par=all_labels[k][l]))



# params = ["t2m", "u10", "v10", 'z', 't', "tcc", "tp"]
# # params_x = ["t2m", "v10", 'z', 't', "tcc", "tp"]
# params_C = ["T2M", "U10", "V10", 'Z', 'T', "TCC", "TP"]
#
# if (draw_world_map):
#     for l in range(6):
#
# if (draw_grad_bar):
#     x = np.arange(len(all_labels[k]))
#
#     width = 0.3
#     fig3, ax = plt.subplots()
#     std = torch.std(std_tensor, dim=1)
#     # std = torch.mul(std, 1000)
#     std = torch.div(std, 10000)
#     print(results)
#     print(std)
#
#     # print(std)
#     rects1 = ax.bar(x, results, width, yerr=std, capsize=4)
#     ax.set_ylabel('Avg. Gradient over Validation data')
#     ax.set_xlabel('parameter x')
#     ax.set_title('{param} Loss Gradients w.r.t. p_{par} ~ 0 and p_x ~ 0 when p_others ~ 1'.format(param=params_C[k],
#                                                                                                   par=params[
#                                                                                                       set_to_zero[k]]),
#                  fontsize=16)
#     ax.set_xticks(x, all_labels[k])
#     ax.tick_params(axis='x', which='major', labelsize=16)
#     # ax.bar_label(rects1, padding=3)
#     fig3.tight_layout()
#     plt.show()
#     # fig3.savefig('/home/ge75tis/Desktop/{param}_gradient_bar_chart'.format(param=params_C[k]))
#
#     if (graph):
#         if (month):
#             x_axis = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
#             labels = all_labels[k]
#             fig = plt.figure()
#             for i in range(6):
#                 plt.errorbar(avg_val_loss_gridded_m[k][i], yerr=0, label=labels[i])
#             fig.suptitle('{param} analysis, p_x ~ 0, p_other ~ 1 per month'.format(param=params_C[k]))
#             fig.errorbar
#             plt.legend()
#             plt.xlabel('months')
#             plt.ylabel('Average loss')
#             fig.savefig(
#                 "/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/per_month_test/{param}_Dropout_per_month_p_x_0".format(
#                     param=params_C[k]))
#         # fig = plt.figure(g, figsize=(10,10))
#         # sns.heatmap(avg_val_loss_gridded[g], linewidths=.5, cmap="Greens", annot=True, xticklabels=grid, yticklabels=grid, norm=LogNorm(), fmt=".3f")
#         # labels = ["t2m", "u10", "v10", 'z', 't', "tcc"]
#         # plt.title('Avg Validation loss of TP for different dropout rates of {par} and other parameters'.format(par=labels[g]))
#         # plt.xlabel('other parameters dropout rate p')
#         # plt.ylabel('{par} dropout rate p'.format(par=labels[g]))
#         # fig.savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/DROPOUT_ANALYSIS/tp_analysis_{label}_heatmap'.format(label=labels[g]))
#
#     # results = np.array(p_gradients.cpu())
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# params_g = [['u10', 'v10', 'z', 'tcc', 'tp'], ['t2m', 'z', 't', 'tcc', 'tp'],['t2m', 'z', 't', 'tcc', 'tp'],
#               ['t2m', 'u10', 't', 'tcc', 'tp'], ['u10', 'v10', 'z', 'tcc', 'tp'], ['t2m', 'u10', 'z', 't', 'tp'], ['t2m', 'u10', 'z', 't', 'tcc']]
#
# bar_chart_std = False
# params_zer = [4, 2, 1, 2, 0, 2, 2]
# if(bar_chart_std):
#     for i in range(7):
#         grads = [[70, 65, 105, 41, 25], [26, 391, 232, 83, 282], [100, 341, 248, 181, 461],  [98, 227, 216, 50, 38],
#                  [118, 115, 169, 64, 20], [37, 234, 116, 117, 241], [0.25, 31, 5, 0.15, 27]]
#
#         stds = [[0.1091, 0.1478, 0.2207, 0.0671, 0.0525], [0.0333, 0.5026, 0.3446, 0.1797, 0.4037], [0.1337,  0.4858, 0.3542, 0.2951, 0.6118]
#                 , [1.8661e-01, 5.7324e-01, 4.5843e-01, 7.0172e-02, 1.7575e-01], [0.1302, 0.1501, 0.2760, 0.0931, 0.0278], [0.0214, 0.1180,0.0981, 0.0634, 0.1965],
#                 [1.2607e-03, 1.7842e-02, 6.5283e-03, 1.8910e-03, 1.5993e-02]]
#
#         clm_losses = [0.1061, 0.5411, 0.6394, 0.2035, 0.1897, 0.6936, 0.354]
#
#         x = np.arange(5)
#         width = 0.3
#         plt.rcParams.update({'font.size': 25})
#         fig, ax = plt.subplots(figsize=(11,8))
#         # low = min(unet_losses)
#         # high = max(unet_losses)
#         # plt.ylim(0.24, 0.28)
#         rects1 = ax.bar(x, grads[i], width, yerr=stds[i], capsize=6, label="Gradient of {param} loss for p_{best} ~ 0 and p_others ~ 1".format(param=params_C[i], best=params_C[params_zer[i]]))
#         # rects2 = ax.bar(x + width/2, clm_losses, width, label="Climatology")
#         ax.set_ylabel('Average Gradient')
#         ax.set_xticks(x, params_g[i])
#         ax.set_xlabel('input parameters')
#         ax.set_title('UNet(5-to-1) prediction when one additional input is ignored')
#         ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
#                   ncol=3, fancybox=True, shadow=True)
#
#         # ax.bar_label(rects1, padding=3)
#         # ax.bar_label(rects2, padding=3)
#
#         fig.tight_layout()
#         plt.show()
#         fig.savefig('/home/ge75tis/Desktop/higher_order_grad_{param}'.format(param=params_C[i]))
#
#     # set_to_zeros = [4, 2, 1, 2, 0, 2, 2]
#     # for i in range(7):
#     #     loss_p = rec_losses[i][0]
#     #     std_p = rec_losses[i][1]
#     #     loss1_p = rec_losses1[i][0]
#     #     std1_p = rec_losses1[i][1]
#     #     x = np.arange(len(all_labels[i]))
#     #     width = 0.3
#     #     fig3, ax = plt.subplots()
#     #     rects1 = ax.bar(x - width/2, loss_p, width, yerr=std_p, capsize=4, label='p_{aa} ~ 0 and p_y ~ 0, p_other ~ 1'.format(param=params_C[i]), aa=params[set_to_zeros[i]])
#     #     # rects2 = ax.bar(x + width/2, loss1_p, width, yerr=std1_p, capsize=4, label='p_x ~ 1, p_other ~ 0'.format(param=params_C[i]))
#     #     ax.set_ylabel('Average validation losses')
#     #     ax.set_xlabel('parameter y')
#     #     ax.set_title('{param} Dropout Analysis noise comparison when p_{aa} ~ 0'.format(param=params_C[i]), aa=params[set_to_zeros[i]])
#     #     ax.set_xticks(x, all_labels[i])
#     #     ax.legend()
#     #     # ax.bar_label(rects1, padding=3)
#     #     fig3.tight_layout()
#     #     plt.show()
#     #     fig3.savefig('/home/ge75tis/Desktop/double_dropout_bar_chart_with_stds_{param}'.format(param=params_C[i]))
#
#
#
#
# distance_cluster = False
# if(distance_cluster):
#
#
#
#

#
#
#
