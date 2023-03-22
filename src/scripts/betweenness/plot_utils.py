import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os

from src.utils.plot_utils import *

def plot_hypergeom_pval(all_criteria_overlap_pvals_topks, interesting_in_pos, interesting_in_top,
                        interesting_in_net, title, filename):
    '''
        frac_overlap_hypergeom_pvals_topks is a dict. Where, k=each k in topks
        and the value is a tuple(x,y)=> x is fraction of overlapping prots in topk and interesting prot,
        y is the pvalue of that overlap.
    '''

    linestyle_dict = {'betweenness':'solid', 'contr_pathlen_2':'dotted',
                        'contr_pathlen_3':(0,(1,10)),'contr_pathlen_4':'dashdot', 'score':'solid'}
    a_sig = 0.01

    fig, ax = plt.subplots(1)
    for rank_criteron in all_criteria_overlap_pvals_topks:
        frac_overlap_hypergeom_pvals_topks = all_criteria_overlap_pvals_topks[rank_criteron]
        x = list(frac_overlap_hypergeom_pvals_topks.keys())
        y = [frac_overlap for (frac_overlap, pval) in list(frac_overlap_hypergeom_pvals_topks.values())]
        pvals = [pval for (frac_overlap, pval) in list(frac_overlap_hypergeom_pvals_topks.values())]
        c = ['g' if i<a_sig else 'r' for i in pvals]

        lines = [((x0, y0), (x1, y1)) for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:])]
        colored_lines = LineCollection(lines, colors=c, linewidths=(2,), linestyle=linestyle_dict[rank_criteron],
                                       label=rank_criteron)

        # plot data
        ax.add_collection(colored_lines)
        ax.autoscale_view()

    plt.axhline(y=interesting_in_pos, color='c', linestyle='--', label = 'frac_in_positive')
    plt.axhline(y=interesting_in_top, color='m', linestyle='--', label='frac_in_top_preds')
    plt.axhline(y=interesting_in_net, color='b', linestyle='--', label='frac_in_net')

    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large',
                       fancybox=True, framealpha=0.5)
    plt.ylim([0, 1])
    plt.xlabel('rank')
    plt.ylabel('fraction of overlap')
    plt.title(title)
    plt.tight_layout()

    # filename1 = filename.replace('.pdf','_'+rank_criteron+'.pdf')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.savefig(filename.replace('.pdf', '.png'))  # save plot in .png format as well
    plt.close()
    print('Save overlap fig to ', filename)


def plot_KS(frac_prots_ge_btns_marker,marker, title, filename):
    markers_dict = {'ppi_prots':'o', 'ess_cell': 'v', 'ess_org': 's', 'viral_sars2--':'P'}

    btns_markers = list(frac_prots_ge_btns_marker['ppi_prots'].keys())
    fig,ax = plt.subplots()
    for prot_type in frac_prots_ge_btns_marker:
        plt.plot(btns_markers, list(frac_prots_ge_btns_marker[prot_type].values()),
                   marker=markers_dict[prot_type], label=prot_type)
        # plt.loglog(btns_markers, list(frac_prots_ge_btns_marker[prot_type].values()),
        #         marker =markers_dict[prot_type] ,label = prot_type)
    #commenting out the log scale
    # ax.set_xscale("log", base=10)
    # ax.set_yscale("log", base=10)

    legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large',
                       fancybox=True, framealpha=0.5)
    # Put a nicer background color on the legend.

    plt.xlabel(marker +' along betweenness score')
    plt.ylabel('fraction of prots')
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.savefig(filename.replace('.pdf', '.png'))  # save plot in .png format as well
    plt.close()
    print('Save fig to ', filename)




def plot_prots_appearing_at_each_pathlens(new_prots_each_pathlens, filename):
    '''Input: a dict with keys: ['network', 'term', 'alg', 'alpha','new_appearing_prots'], each value is a list.
    Especially: value for the key 'new_appearing_prots' is a list of dicts.
    Where inner dict keys:['pathlen_2', 'pathlen_3','pathlen_4']

    Output: a plot where along x-axis we will have networks and along y axis we will have stacked bar chart.
    In the bar chart, we will have the #of_prots appearing as we go along each path_len.
    '''

    #converting prots_appearing_at_each_pathlens into dataframe
    new_prots_each_pathlens_df = pd.DataFrame(new_prots_each_pathlens)
    #for a certain term-alg-alpha combo plot all the networks and  #new_appearing_prots in one plot.
    #the following will return a list of unique (term, alg, alpha) tuples present in prots_appearing_at_each_pathlens
    term_alg_alpha_list = list(new_prots_each_pathlens_df.groupby(by=['term', 'alg', 'alpha']).groups.keys())

    for (term, alg, alpha) in term_alg_alpha_list:
        df = new_prots_each_pathlens_df[(new_prots_each_pathlens_df['term']==term)&
                                        (new_prots_each_pathlens_df['alg']==alg) &
                                        (new_prots_each_pathlens_df['alpha']==alpha)]

        df.set_index('network', inplace=True)
        df=df[['pathlen_2', 'pathlen_3','pathlen_4']]
        ax = df.plot.bar(stacked=True)

        legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large',
                           fancybox=True, framealpha=0.5)
        # Put a nicer background color on the legend.

        plt.xlabel('Networks')
        plt.ylabel('New prots appearing at each path lengths')
        plt.title(term + '_'+ alg + '_' + str(alpha))
        plt.tight_layout()

        filename1 =filename.replace('.pdf', term + '_'+ alg + '_' + str(alpha)+'.pdf')
        os.makedirs(os.path.dirname(filename1), exist_ok=True)
        plt.savefig(filename1)
        plt.savefig(filename1.replace('.pdf', '.png'))  # save plot in .png format as well
        plt.show()
        plt.close()
        print('Save fig to ', filename1)



def barplot_from_dict(n_essential_prots_per_topk, x_label, y_label,ymax, filename,title=''):
    '''
    n_essential_prots_per_topk: dict where key = k, value: number of essential prot in top k
    '''
    X = list(n_essential_prots_per_topk.keys())
    Y =  list(n_essential_prots_per_topk.values())
    sns.barplot(x=X, y=Y)

    plt.ylim([0,ymax])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.savefig(filename.replace('.pdf', '.png'))  # save plot in .png format as well
    # plt.show()
    plt.close()
    print('Save fig to ', filename)

def barplot_from_df(n_interesting_prots_per_topk_df, filename, x, y,ymax, title=''):
    '''
    n_essential_prots_per_topk: dict where key = k, value: number of essential prot in top k
    '''
    sns.barplot(n_interesting_prots_per_topk_df, x = x, y = y, hue='alpha')

    plt.ylim([0, ymax])
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.savefig(filename.replace('.pdf', '.png'))  # save plot in .png format as well
    # plt.show()
    plt.close()
    print('Save fig to ', filename)

def scatter_plot(X, Y,  x_label, y_label, ymin=0, ymax=100, title='', filename=''):


    #in this case the last to points are for positive/src_bin and to_predicted_proteins_bin. And we want to show them in
    #different color
    color_non_top_pos = ['blue']*(len(X)-2)
    marker_non_top_pos = '.'
    plt.scatter(X[0:-2],Y[0:-2],color = color_non_top_pos, marker=marker_non_top_pos)

    color_pos = ['red']
    marker_pos='^'
    plt.scatter(X[-2],Y[-2],color = color_pos, marker=marker_pos)


    color_top = ['yellow']
    marker_top='s'
    plt.scatter(X[-1],Y[-1],color = color_top, marker=marker_top)



    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim([ymin, ymax])
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.savefig(filename.replace('.pdf', '.png'))  # save plot in .png format as well
    # plt.show()
    plt.close()
    print('Save fig to ', filename)

def box_plot(data, x, y, ymin, ymax, title, filename):
    sns.boxplot(data, x=x, y=y )

    plt.ylim([ymin, ymax])
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.savefig(filename.replace('.pdf', '.png'))  # save plot in .png format as well
    # plt.show()
    plt.close()
    print('Save fig to ', filename)