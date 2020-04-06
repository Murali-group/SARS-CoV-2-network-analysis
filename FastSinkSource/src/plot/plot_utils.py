import argparse
import yaml
import itertools
import os
import sys
#from collections import defaultdict
#from tqdm import tqdm
#import time
import numpy as np
#from scipy import sparse
from scipy.stats import kruskal, mannwhitneyu, wilcoxon
import networkx as nx
# plotting imports
import matplotlib
matplotlib.use('Agg')  # To save files remotely. 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
# make this the default for now
sns.set_style('darkgrid')
# add two levels up to the path
#from os.path import dirname
#base_path = dirname(dirname(dirname(__file__)))
#sys.path.append(base_path)
#print(sys.path)
#os.chdir(base_path)
#print(base_path)
#import run_eval_algs
#from src.algorithms import runner as runner
#sys.path.append(base_path + "/src/algorithms")
#print(sys.path)
#import runner
# my local imports
fss_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,fss_dir)
import run_eval_algs
import src.algorithms.runner as runner
import src.utils.file_utils as utils
import src.evaluate.eval_utils as eval_utils
import src.evaluate.cross_validation as cv


ALG_NAMES = {
    'localplus': 'Local+', 'local': 'Local',
    'sinksource': 'SinkSource', 'sinksourceplus': 'SinkSource+',
    'sinksource_bounds': 'SinkSource_Bounds',
    'fastsinksource': 'FSS', 'fastsinksourceplus': 'FSS+',
    'genemania': 'GeneMANIA',  
    'logistic_regression': 'LogReg',
    'svm': 'SVM',
    }

measure_map = {'fmax': r'F$_{\mathrm{max}}$'}
param_map = {'alpha': r'$\rm \alpha$'}
#param_map = {'alpha': r'$\mathbf{\mathrm{\alpha}}$'}

# tried to be fancy :P
# colors: https://coolors.co/ef6e4a-0ec9aa-7c9299-5d88d3-96bd33
#my_palette = ["#EF6E4A", "#0EC9AA", "#7C9299", "#5D88D3", "#96BD33", "#937860", "#EFD2B8"]
my_palette = ["#EF6E4A", "#0EC9AA", "#8e71d0", "#5D88D3", "#96BD33", "#937860", "#EFD2B8"]
alg_colors = {
    'fastsinksource': my_palette[0],
    'genemania': my_palette[1],
    'localplus': my_palette[3],
    'sinksource': my_palette[4],
    }
# default list of markers
my_shapes = ['o', 's', 'P', '^', 'x', '*', '+', 'v', 'x',]
alg_shapes = {
    'fastsinksource': 'o',
    'genemania': 's',
    'localplus': '^',
    'sinksource': 'x',
    }
for alg, color in alg_colors.copy().items():
    alg_colors[ALG_NAMES[alg]] = color
    alg_shapes[ALG_NAMES[alg]] = alg_shapes[alg]
# for comparing sinksource with local
#my_palette = ["#EF6E4A", sns.xkcd_rgb["deep sky blue"], "#96BD33", "#937860", "#efd2b8"]
#my_palette = [sns.xkcd_rgb["orange"], sns.xkcd_rgb["azure"], "#96BD33", "#937860", "#efd2b8"]


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser()

    # general parameters
    group = parser.add_argument_group('Main Options')
    # TODO take multiple config files
    group.add_argument('--config', type=str, required=True,
                     help="Configuration file")
    group.add_argument('--alg', '-A', dest='algs', type=str, action="append",
                     help="Algorithms to plot. Must be in the config file. If specified, will ignore 'should_run' in the config file")
    group.add_argument('--out-pref', '-o', type=str, default="",
                     help="Output prefix for writing plot to file. Default: outputs/viz/<net_version>/<exp_name>/")
    group.add_argument('--goterm', '-G', type=str, action="append",
                     help="Specify the GO terms to use (should be in GO:00XX format)")
    #group.add_argument('--exp-type', type=str, default='cv-5folds',
    #                 help='Type of experiment (e.g., cv-5fold, loso, temporal-holdout). Default: cv-5folds')
    group.add_argument('--only-terms-file', type=str, 
                     help="File containing a list of terms (in the first col, tab-delimited) for which to limit the results")
    group.add_argument('--only-terms-name', type=str, default='',
                     help="If --only-terms is specified, use this option to append a name to the file. Default is to use the # of terms")
    group.add_argument('--postfix', type=str, 
                     help="Postfix to add to the end of the files")

    group = parser.add_argument_group("Evaluation options")
    group.add_argument('--cross-validation-folds', '-C', type=int,
                     help="Get results from cross validation using the specified # folds")
    group.add_argument('--num-reps', type=int, default=1,
                     help="If --exp-type is <cv-Xfold>, this number of times CV was repeated. Default=1")
    group.add_argument('--cv-seed', type=int,
                     help="Seed used when running CV")
    group.add_argument('--sample-neg-examples-factor', type=float, 
                     help="Factor of # positives used to sample a negative examples")
    group.add_argument('--loso', action='store_true',
                     help="Get results from leave-one-species-out validation")

    # plotting parameters
    group = parser.add_argument_group('Plotting Options')
    group.add_argument('--measure', action="append",
                     help="Evaluation measure to use. May specify multiple. Options: 'fmax', 'avgp', 'auprc', 'auroc'. Default: 'fmax'")
    group.add_argument('--boxplot', action='store_true', default=False,
                     help="Compare all terms for all runners in the config file using a boxplot")
    group.add_argument('--line', action='store_true', default=False,
                     help="Compare all runners on all datasets in the config file using a lineplot")
    group.add_argument('--ci', type=float,
                     help="Show the specified confidence interval (between 0 and 100) for the line plot")
    group.add_argument('--scatter', action='store_true', default=False,
                     help="Make a scatterplot, or pair plot if more than two runners are given." +
                     "If the # ann are given with --term-stats, then plot the fmax by the # ann")
    group.add_argument('--prec-rec', action='store_true', default=False,
                     help="Make a precision recall curve for each specified term")

    group = parser.add_argument_group('Parameter / Statisical Significance Options')
    group.add_argument('--compare-param', type=str,
                       help="name of parameter to compare (e.g., alpha)")
    group.add_argument('--max-val', type=str,
                       help="Maximum value of the parameter against which to compare statistical significance (e.g., 1.0 for alpha")
    # shouldn't be needed. Just don't specify any plotting options (e.g., --boxplot)
    #group.add_argument('--stats-only', type=float,
    #                   help="Maximum value of the parameter against which to compare statistical significance (e.g., 1.0 for alpha")

    # figure parameters
    group = parser.add_argument_group('Figure Options')
    # Moved to the config file
    group.add_argument('--title','-T', 
                     help="Title to give the figure. Default is the exp_name ")
    group.add_argument('--for-paper', action='store_true', default=False,
                     help="Exclude extra information from the title and make the labels big and bold")
    group.add_argument('--horiz','-H', dest="horizontal", action='store_true', default=False,
                     help="Flip the plot so the measure is on the y-axis (horizontal). Default is x-axis (vertical)")
    group.add_argument('--png', action='store_true', default=False,
                     help="Write a png in addition to a pdf")
    group.add_argument('--term-stats', type=str, action='append',
                     help="File which contains the term name, # ann and other statistics such as depth. " +
                     "Useful to add info to title of prec-rec plot. Can specify multiple")
    group.add_argument('--plot-postfix', type=str, default='',
                     help="Postfix to add to the end of the figure files")
    group.add_argument('--forceplot', action='store_true', default=False,
                     help="Force overwitting plot files if they exist. TODO not yet implemented.")

    return parser


def load_config_file(config_file):
    with open(config_file, 'r') as conf:
        #config_map = yaml.load(conf, Loader=yaml.FullLoader)
        config_map = yaml.load(conf)
    return config_map


def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    config_map = load_config_file(kwargs['config'])
    # TODO check to make sure the inputs are correct in config_map

    #if opts.exp_name is None or opts.pos_neg_file is None:
    #    print("--exp-name, --pos-neg-file, required")
    #    sys.exit(1)
    if kwargs['measure'] is None:
        kwargs['measure'] = ['fmax']
    kwargs['measures'] = kwargs['measure']
    del kwargs['measure']
    kwargs['alg_names'] = ALG_NAMES

    return config_map, kwargs


def main(config_map, ax=None, out_pref='', **kwargs):

    input_settings, alg_settings, output_settings, out_pref, kwargs = setup_variables(
        config_map, out_pref, **kwargs)
    kwargs['out_pref'] = out_pref
    if kwargs['out_pref'] is not None:
        kwargs['out_pref'] += kwargs.get('plot_postfix', '') 
    print(kwargs)

    # plot prec-rec separately from everything else
    if kwargs['prec_rec']:
        # loop through all specified terms, or use an empty string if no terms were specified
        terms = kwargs['goterm'] if kwargs['goterm'] is not None else ['']
        for term in terms:
            term = '-'+term if term != '' else ''
            prec_rec = 'prec-rec' + term
            #kwargs['prec_rec'] = prec_rec
            df_all = load_all_results(input_settings, alg_settings, output_settings, prec_rec_str=prec_rec, **kwargs)
            if len(df_all) == 0:
                print("no results found. Quitting")
                sys.exit()
            # limit to the specified terms
            if kwargs['only_terms'] is not None:
                df_all = df_all[df_all['#term'].isin(kwargs['only_terms'])]

            title = '-'.join(df_all['plot_exp_name'].unique())
            plot_curves(df_all, title=title, **kwargs)
    else:
        # get the path to the specified files for each alg
        df_all = load_all_results(input_settings, alg_settings, output_settings, **kwargs)
        if len(df_all) == 0:
            print("no terms found. Quitting")
            sys.exit()
        # limit to the specified terms
        if kwargs.get('only_terms') is not None:
            df_all = df_all[df_all['#term'].isin(kwargs['only_terms'])]
        num_terms = df_all['#term'].nunique()
        if kwargs.get('loso'):
            sp_taxon_pairs = df_all['#taxon'].astype(str) + df_all['#term']
            num_terms = sp_taxon_pairs.nunique()
            #num_terms = df_all.groupby(['#taxon', '#term']).size()
        algs = df_all['Algorithm'].unique()

        print("\t%d algorithms, %d plot_exp_name values\n" % (len(algs), len(df_all['plot_exp_name'].unique())))
        #print(df_all.head())
        results_overview(df_all, measures=kwargs['measures'])

        if kwargs.get('title') is not None:
            title = kwargs['title']
        else:
            title = '-'.join(df_all['plot_exp_name'].unique())
        # add the cross-validation settings to the plot
        if kwargs.get('cross_validation_folds'):
            title += " \n%s%s%s" % (
                " neg-factor=%s;"%kwargs['sample_neg_examples_factor'] if kwargs.get('sample_neg_examples_factor') else '',
                " seed=%s;"%kwargs['cv_seed'] if kwargs.get('cv_seed') else "",
                " # reps=%s;"%kwargs['num_reps'] if kwargs.get('num_reps',1) > 1 else "",
            )
        if not kwargs.get('for_paper') and num_terms > 1:
            title += " \n %d%s %s" % (
                    num_terms, ' %s'%kwargs.get('only_terms_name', ''),
                    "sp-term pairs" if kwargs.get('loso') else 'terms')
        kwargs['title'] = title
        kwargs['alg_params'] = alg_settings
        kwargs['algs'] = get_algs_to_run(alg_settings, **kwargs)
        # if no algs were specified and the yaml file default algs were used, 
        # then update the list of algs here
        #if kwargs['algs'] is None:


        # now attempt to figure out what labels/titles to put in the plot based on the net version, exp_name, and plot_exp_name
        for measure in kwargs['measures']:
            # also check the statistical significance options
            if kwargs['compare_param'] and kwargs['max_val']:
                compute_param_stat_sig(df_all, measure=measure, **kwargs)
            if kwargs['scatter']:
                ax = plot_scatter(df_all, measure=measure, ax=ax, **kwargs) 
            if kwargs['line']:
                ax = plot_line(df_all, measure=measure, ax=ax, **kwargs)
            if kwargs['boxplot']:
                if df_all['plot_exp_name'].nunique() > 1:
                    ax = plot_multi_boxplot(df_all, measure=measure, ax=ax, **kwargs)
                else:
                    ax = plot_boxplot(df_all, measure=measure, ax=ax, **kwargs)
    return ax


def setup_variables(config_map, out_pref='', **kwargs):
    """
    Function to setup the various args specified in kwargs
    """
    input_settings = config_map['input_settings']
    #input_dir = input_settings['input_dir']
    alg_settings = config_map['algs']
    output_settings = config_map['output_settings']
    # update the settings specified in this script with those set in the yaml file
    if config_map.get('eval_settings'):
        kwargs.update(config_map['eval_settings'])
    if config_map.get('plot_settings'):
        #config_map['plot_settings'].update(kwargs)
        kwargs.update(config_map['plot_settings'])
        # overwrite whatever is in the plot settings with the specified args
        if kwargs.get('out_pref') and out_pref != '':
            del kwargs['out_pref']
            #kwargs['out_pref'] = out_pref
        elif kwargs.get('out_pref'):
            out_pref = kwargs['out_pref']
    if kwargs.get('term_stats') is not None:
        df_stats_all = pd.DataFrame()
        for f in kwargs['term_stats']:
            df_stats = pd.read_csv(f, sep='\t')
            df_stats_all = pd.concat([df_stats_all, df_stats])
        kwargs['term_stats'] = df_stats_all

    # if no postfix was set in the yaml file or in this script, then set it to empty
    if kwargs.get('postfix') is None:
        kwargs['postfix'] = ''

    if out_pref == "":
        out_pref = "%s/viz/%s/%s/" % (
                output_settings['output_dir'], 
                input_settings['datasets'][0]['net_version'], 
                input_settings['datasets'][0]['exp_name'])
    if kwargs.get('only_terms_file') is not None:
        only_terms = pd.read_csv(kwargs['only_terms_file'], sep='\t', index_col=None)
        only_terms = only_terms.iloc[:,0].values
        print("limitting to %d terms from %s" % (len(only_terms), kwargs['only_terms_file']))
        kwargs['only_terms'] = only_terms
        # setup the name to add to the output file
        only_terms_postfix = kwargs['only_terms_name'].lower() + str(len(kwargs['only_terms'])) + '-'
        out_pref += only_terms_postfix

    # TODO only create the output dir if plots are will be created
    if out_pref is not None:
        out_pref += kwargs.get('postfix','')
        utils.checkDir(os.path.dirname(out_pref))

    return input_settings, alg_settings, output_settings, out_pref, kwargs


def savefig(out_file, **kwargs):
    print("Writing %s" % (out_file))
    plt.savefig(out_file, bbox_inches='tight')
    if kwargs.get('png'):
        plt.savefig(out_file.replace('.pdf','.png'), bbox_inches='tight')
    plt.close()


def set_labels(ax, title, xlabel, ylabel, axis_fontsize=11, **kwargs):
    xlabel, ylabel = (ylabel, xlabel) if kwargs.get('horizontal') else (xlabel, ylabel)
    if kwargs.get('for_paper'):
        ax.set_xlabel(xlabel, fontsize=axis_fontsize, weight="bold")
        ax.set_ylabel(ylabel, fontsize=axis_fontsize, weight="bold")
        ax.set_title(title, fontsize=18, weight="bold")
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)


def plot_line(df, measure='fmax', out_pref="test", title="", ax=None, **kwargs):
    #print(df[[measure, 'plot_exp_name', 'Algorithm']].head())
    x,y = measure, 'plot_exp_name'
    # flip the x and y axis if specified
    x,y = (y,x) if kwargs.get('horizontal') else (x,y)
    #print(df.columns)
    df = df[[x,y, 'Algorithm']]
    algs = df['Algorithm'].unique()
    try:
        # If there are algorithms which don't have a set shape or color, this will fail
        curr_palette = [alg_colors[alg] for alg in algs]
        curr_markers = [alg_shapes[alg] for alg in algs]
    except KeyError:
        curr_palette = my_palette
        curr_markers = my_shapes 
    #print(df.head())
    #df = df.pivot(columns='Algorithm', values=[x,y])
    #print(df.head())
    if ax is None:
        if kwargs['horizontal']: 
            f, ax = plt.subplots(figsize=(6,5))
        else:
            f, ax = plt.subplots(figsize=(4,6))
    if kwargs.get('for_paper'):
        # make the fonts bigger
        matplotlib.rc('xtick', labelsize=18) 
        matplotlib.rc('ytick', labelsize=18) 

    # doesn't work for categorical data
    #sns.lineplot(x=measure, y='pen-alg', data=df, ax=ax,
    if kwargs.get('ci'):
        ax = sns.pointplot(
            x=x, y=y, data=df, hue='Algorithm', hue_order=algs, estimator=np.median,  # ci=None,
            dodge=0.6, join=False, ci=kwargs['ci'], 
            ax=ax, markers=curr_markers, palette=curr_palette)
        # also make a table for the row names
        # and to alternate the grid colors
    else:
        ax = sns.pointplot(
            x=x, y=y, data=df, hue='Algorithm', estimator=np.median, ci=None,
            ax=ax, markers=curr_markers,  # order=[kwargs['alg_names'][a] for a in algorithms],
            palette=curr_palette)

    plt.setp(ax.lines,linewidth=1)  # set lw for all lines of g axes
    if kwargs['horizontal'] and len(df['plot_exp_name'].unique()[-1]) > 10:
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', labelsize='large')

    # TODO add an option for this
    if kwargs.get('for_paper') and kwargs.get('horizontal'):
        # add alternating colors for the columns
        num_cols = df[x].nunique()
        add_alternating_columns(ax, num_cols) 

        # add the table below the figure
        ax_below = add_net_combination_table(ax, df[x].unique())
        # and also add alternating columns for this table
        add_alternating_columns(ax_below, num_cols) 
        # remove the extra space on the ends of the x axis
        x1, x2 = ax.get_xlim()
        ax.set_xlim(int(x1)-0.5, int(x2)+0.5)

    ## change the fmax axis to be between 0 and 1
#    if kwargs.get('for_paper'):
#        # TODO add an option for this:
#        #x1, x2 = kwargs.get('ax_limit',(0.6, 0.8))
#        x1, x2 = kwargs.get('ax_limit',(0.34, 0.72))
#        if kwargs['horizontal']: 
#            #ax.set_ylim(-0.02, 1.02)
#            #ax.set_ylim(0.28, 0.72)
#            ax.set_ylim(x1,x2)
#            # set the yticks
#            #ax.set_yticks(np.arange(x1,x2,0.02))
#            #ax.set_yticks(np.arange(x1,x2,0.05))
#        else:
#            ax.set_xlim(x1, x2)

    xlabel = "%s" % (measure_map.get(measure, measure.upper()))
    ylabel = kwargs.get('exp_label', '')
    axis_fontsize = 18 if kwargs.get('for_paper') else 11
    set_labels(ax, title, xlabel, ylabel, axis_fontsize=axis_fontsize, **kwargs)

    if out_pref is not None:
        out_file = "%s%s-line.pdf" % (out_pref, measure)
        savefig(out_file, **kwargs)
    return ax


def add_alternating_columns(ax, num_cols, gray="#d1d1d9"):
    for i in range(num_cols):
        if i % 2:
            #ax.axvline(i, linewidth=1, color=gray, zorder=0)
            #ptch = patches.Rectangle(
            #    (i-0.5,0),
            #    width=1,
            #    height=1,
            #    edgecolor=gray,
            #    facecolor=gray,
            #    zorder=0)
            #ax.add_artist(ptch)
            ax.axvspan(
                i-0.5, i+0.5,
                facecolor=gray, edgecolor=gray,
                zorder=0)


def add_net_combination_table(ax, plot_exp_names, checkmark="✓"):
    # make the subplot 1/4 the size of the original
    ax_below = ax.get_figure().add_subplot(4,1,4, sharex=ax)
    box = ax_below.get_position()
    # move the axes below the current figure
    ax_below.set_position([box.x0, box.y0-0.18, box.width, box.height])
    # now add the checkmarks
    net_types = [
        'ssnT',
        'ssnLocal',
        'ssnC',
        'stringT',
        'stringC',
    ][::-1]
    # change the labels here
    net_type_labels = {
        'ssnT': 'SSN-T',
        'ssnLocal': 'SSN-Neighbors',
        'ssnC': 'SSN-C',
        'stringT': 'STRING-T',
        'stringC': 'STRING-C',
        }
    for x, plot_exp_name in enumerate(plot_exp_names):
        #print(plot_exp_name)
        for y, net_type in enumerate(net_types):
            if net_type in plot_exp_name:
                #print("\t%s (%s)" % (net_type, y))
                ax_below.text(x-0.1, y+0.05, checkmark, fontsize="14")
    # set the tick labels as the network combination types
    ax_below.set_yticklabels(
        [net_type_labels[net_type] for net_type in net_types]+[""],
        fontsize="14")
    for label in ax_below.yaxis.get_majorticklabels():
        # this is a trick to get the tick label to be between the grid lines
        label.set_verticalalignment('bottom')

    ax_below.set_yticks(list(range(0,5)))
    ax_below.set_ylim(-0.05, 5)
    ax_below.xaxis.grid(False)
    ax_below.set_xticklabels([""]*len(plot_exp_names))
    return ax_below


def plot_multi_boxplot(df, measure='fmax', out_pref="test", title="", ax=None, **kwargs):
    algs = df['Algorithm'].unique()
    try:
        # If there are algorithms which don't have a set shape or color, this will fail
        curr_palette = [alg_colors[alg] for alg in algs]
        curr_markers = [alg_shapes[alg] for alg in algs]
    except KeyError:
        curr_palette = my_palette
        curr_markers = my_shapes 

    if kwargs['for_paper'] is True:
        sns.set_style("darkgrid", {
            'xtick.top': True, 'xtick.bottom': True, #'xtick.color': '.3',
            #'grid.color': '.3', 'axes.labelcolor': '.15',
            #'axes.edgecolor': '.3',
            'axes.spines.bottom': True, 'axes.spines.left': True,
            'axes.spines.right': True, 'axes.spines.top': True,})
    df['Algorithm'] = df['Algorithm'].astype(str)
    df = df[['Algorithm', measure, 'plot_exp_name']]
    g = sns.catplot(x=measure, y='Algorithm', row='plot_exp_name', data=df,  # hue='Algorithm',
                    height=1., aspect=4, palette=curr_palette, 
                    #orient='v' if not kwargs.get('horizontal') else 'h',
                    kind='box',)
                    #kind="violin", cut=0, inner='quartile',)
    # put less space between the plots
    g.fig.subplots_adjust(hspace=.05)
    g.set(xticks=np.arange(0,11)*0.1)
    if kwargs['for_paper'] is True:
        g.set_ylabels("")
        g.set_titles("")
        g.fig.tight_layout()
        xlabel = measure_map.get(measure, measure.upper())
        g.set_xlabels(xlabel)

    if out_pref is not None:
        out_file = "%s%s-multi-boxplot.pdf" % (out_pref, measure)
        savefig(out_file, **kwargs)


def plot_boxplot(df, measure='fmax', out_pref="test", title="", ax=None, **kwargs):
    df['Algorithm'] = df['Algorithm'].astype(str)
    df = df[['Algorithm', measure]]
    #print(df.head())
    df.reset_index(inplace=True)
    df = df.pivot(columns='Algorithm', values=measure)
    #print(df.head())
    #ax = sns.boxplot(x=measure, y='Algorithm', data=df, ax=ax,
    if kwargs.get('compare_param'):
        # we're not ordering the algorithms, but the parameters
        # and the param is in the 'algs' column
        order = []
        for alg in kwargs['algs']:
            order += [str(p) for p in kwargs['alg_params'][alg][kwargs['compare_param']]]
    else:
        order = [kwargs['alg_names'][a] for a in kwargs['algs']]
    print("horizontal: %s" % (kwargs.get('horizontal')))
    print("orient: %s" % ('v' if not kwargs.get('horizontal') else 'h'))

    ax = sns.boxplot(data=df, ax=ax,
                     fliersize=1.5, order=order,
                     orient='v' if not kwargs.get('horizontal') else 'h',
                     palette=my_palette if 'palette' not in kwargs else kwargs['palette'],
                )

    xlabel = ""
    if 'exp_label' in kwargs:
        xlabel = kwargs['exp_label']
    elif 'compare_param' in kwargs:
        xlabel = param_map.get(kwargs['compare_param'],kwargs['compare_param'])
    ylabel = measure_map.get(measure, measure.upper())
    # for some reason sharex/sharey wasn't working, so I just removed the label
    if kwargs.get('share_measure') is True:
        ylabel = ""
    set_labels(ax, title, xlabel, ylabel, **kwargs)

    if out_pref is not None:
        out_file = "%s%s-boxplot.pdf" % (out_pref, measure)
        savefig(out_file, **kwargs)
    return ax


def plot_curves(df, out_pref="test", title="", ax=None, **kwargs):
    """
    Plot precision recall curves, or (TODO) ROC curves 
    """
    # make a prec-rec plot per term
    for term in sorted(df["#term"].unique()):
        curr_df = df[df['#term'] == term]
        # get only the positive examples to plot prec_rec
        curr_df = curr_df[curr_df['pos/neg'] == 1]
        # also put the fmax on the plot, and add it to the label
        new_alg_names = []
        fmax_points = {}
        for alg in curr_df['Algorithm'].unique():
            df_alg = curr_df[curr_df['Algorithm'] == alg]
            #print(df_alg['prec'], df_alg['rec'])
            fmax, idx = eval_utils.compute_fmax(df_alg['prec'].values, df_alg['rec'].values, fmax_idx=True)
            new_alg_name = "%s (%0.3f)" % (alg, fmax)
            new_alg_names.append(new_alg_name) 
            fmax_points[alg] = (df_alg['prec'].values[idx], df_alg['rec'].values[idx])

        fig, ax = plt.subplots()
        # TODO show the standard deviation from the repititions
        sns.lineplot(x='rec', y='prec', hue='Algorithm', data=curr_df,
                ci=None, ax=ax, legend=False,
                )
                #xlim=(0,1), ylim=(0,1), ci=None)

        ax.set_xlim(-0.02,1.02)
        ax.set_ylim(-0.02,1.02)

        ax.legend(title="Alg (Fmax)", labels=new_alg_names)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

        # also add the fmax point to the plot
        for i, alg in enumerate(fmax_points):
            prec, rec = fmax_points[alg]
            ax.plot([rec], [prec], marker="*", color=sns.color_palette()[i])

        if kwargs.get('term_stats') is not None:
            df_stats = kwargs['term_stats'] 
            curr_df_stats = df_stats[df_stats['#GO term'] == term]
            # TODO what if there are multiple stats lines?
            term_name = curr_df_stats['GO term name'].values[0]
            term_cat = curr_df_stats['GO category'].values[0]
            # For HPO, map O to Phenotypic abnormality
            cat_map = {"O": "PA", 'P': 'BP', 'F': 'MF', 'c': 'CC'}
            term_cat = cat_map[term_cat] if term_cat in cat_map else term_cat
            term_ann = curr_df_stats['# positive examples'].values[0]
            print(term_name, term_cat, term_ann)
            ax.set_title(title + "\n %s (%s) - %s, %s ann" % (term_name, term, term_cat, term_ann))
        else:
            ax.set_title(title + " %s" % (term))

        if out_pref is not None:
            out_file = "%s%s-prec-rec.pdf" % (out_pref, term)
            savefig(out_file)


def plot_scatter(df, measure='fmax', out_pref="test", title="", ax=None, **kwargs):
    if kwargs.get('loso'):
        df['taxon-term'] = df['#taxon'].map(str) + '-' + df['#term']
        index_col = 'taxon-term'
    else:
        # change the index to the terms
        index_col = df.columns[0]
    df.set_index(index_col, inplace=True)
    # check if algorithms are being compared, or experiments for a single algorithm
    algs = df['Algorithm'].unique()
    plot_exp_names = df['plot_exp_name'].unique()
    if len(algs) == 1 and len(plot_exp_names) > 1:
        print("Comparing %d experiments: %s" % (len(plot_exp_names), ', '.join(plot_exp_names)))
        compare_col = "plot_exp_name"
        algs = plot_exp_names
    elif len(algs) > 1:
        print("Comparing %d algorithms: %s" % (len(algs), ', '.join(algs)))
        compare_col = "Algorithm"
    else:
        print("ERROR: nothing to compare for scatterplot. %d algs (%s), %d experiments (%s)" % (len(algs), ', '.join(algs), len(plot_exp_names), ', '.join(plot_exp_names)))
    df2 = df[[measure, compare_col]]
    print(df2.head())
    if kwargs['term_stats'] is not None:
        df_stats = kwargs['term_stats'] 
        # change the index to the terms
        df_stats.set_index(df_stats.columns[0], inplace=True)
        # plot the fmax by the # annotations
        df2['num_ann'] = df_stats['# positive examples']

        # now plot the # annotations on the x axis, and the fmax on the y axis
        ax = sns.scatterplot('num_ann', measure, hue=compare_col, data=df2, ax=ax,
                             linewidth=0,)
        ax.set_title(title)
    else:
        df2 = df2.pivot(columns=compare_col)
        orig_num = str(len(df2.index))
        df2.dropna(inplace=True)
        new_num = str(len(df2.index))
        df2.columns = [' '.join(col).strip() for col in df2.columns.values]
        # if there are only two algorithms, make a joint plot
        if len(algs) == 2:
            g = sns.jointplot(df2.columns[0], df2.columns[1], data=df2,
                              xlim=(-0.02,1.02), ylim=(-0.02,1.02),
                              marginal_kws=dict(bins=10))
            # also plot x=y
            g.ax_joint.plot((0,1),(0,1))
            alg1, alg2 = df2.columns[0], df2.columns[1]
            diff_col = "diff"
            df2[diff_col] = df2[alg1] - df2[alg2]
            # and print out some stats
            print("%s > %s: %d (%0.3f)"% (alg1, alg2, len(df2[df2[diff_col] > 0]), len(df2[df2[diff_col] > 0]) / float(len(df2))))
            print("%s < %s: %d (%0.3f)"% (alg1, alg2, len(df2[df2[diff_col] < 0]), len(df2[df2[diff_col] < 0]) / float(len(df2))))
            print("%s = %s: %d (%0.3f)"% (alg1, alg2, len(df2[df2[diff_col] == 0]), len(df2[df2[diff_col] == 0]) / float(len(df2))))
        # if there are more, a pairplot
        else:
            g = sns.pairplot(data=df2)
            g.set(xlim=(-0.02,1.02), ylim=(-0.02,1.02))
            # draw the x=y lines
            for i in range(len(algs)):
                for j in range(len(algs)):
                    if i != j:
                        g.axes[i][j].plot((0,1),(0,1))

        # move the plots down a bit so the title isn't overalpping the subplots
        plt.subplots_adjust(top=0.9)
        title = title.replace(orig_num, new_num)
        g.fig.suptitle(title)

    if out_pref is not None:
        out_file = "%s%s-scatter.pdf" % (out_pref, measure)
        savefig(out_file, **kwargs)


def compute_param_stat_sig(df, measure='fmax', **kwargs):
    compare_param, max_val = kwargs['compare_param'], str(kwargs['max_val'])
    # 'Algorithm' is actually holding the parameter value
    param_col = 'Algorithm'
    alg_params = kwargs['alg_params']
    out_file = "%s%s-pvals.tsv" % (kwargs['out_pref'], measure)
    out_str = "#alg\t%s\t%s-2\t2-median fmax\tpval\tmedian of differences\n" % (compare_param, compare_param)
    if kwargs['forceplot'] or not os.path.isfile(out_file):
        params_list = []
        params_alg = {}
        for alg in kwargs['algs']:
            params_list += [str(p) for p in alg_params[alg][compare_param]]
            # also get which alg the parameter value came from
            params_alg.update({str(p):alg for p in alg_params[alg][compare_param]})

        #for param, df_param in df.groupby('Algorithm'):
        max_param_vals = df[df[param_col] == max_val][measure].dropna().values
        for p in params_list:
            param_vals = df[df[param_col] == p][measure].dropna().values
            if len(max_param_vals) != len(param_vals):
                print("ERROR: # terms doesn't match exactly for %s vs %s:" % (max_val, p))
                print(len(max_param_vals), len(param_vals))
                sys.exit("quitting")
            if p == max_val:
                out_str += "%s\t%s\t%s\t%0.3f\t-\t-\n" % (
                    params_alg[p], max_val, p, np.median(param_vals))
            else:
                # to be consistent, I'm going to use the wilcoxon rank-sum test here as well
                #test_statistic, pval = mannwhitneyu(max_param_vals, param_vals, alternative='greater') 
                test_statistic, pval = wilcoxon(max_param_vals, param_vals, alternative='greater') 
                out_str += "%s\t%s\t%s\t%0.3f\t%0.3e\t%0.3e\n" % (
                    params_alg[p], max_val, p, np.median(param_vals),
                    pval, np.median(max_param_vals - param_vals))
                    #np.median(max_param_vals - param_vals) if len(max_param_vals) == len(param_vals) else -1)

        print("writing to %s" % (out_file))
        with open(out_file, 'w') as out:
            out.write(out_str)
        print(out_str)
    else:
        print("%s already exists. Use --forceplot to overwrite. Skipping" % (out_file))
# backup from before
#def compute_stats(df_fmax, params_list, stat_file, alg='sinksource', exp_type='alpha', forced=False, compare_to=None):
#    if forced or not os.path.isfile(stat_file):
#        if compare_to is None:
#            alg1 = max(params_list)
#            if exp_type == "maxi-alpha":
#                alg1 = "a1.0-1000" 
#        else:
#            alg1 = compare_to
#        alpha_1 = df_fmax[alg1].dropna().values
#        out_str = "#alg\t%s\t%s-2\t2-median fmax\tpval\tmedian of differences\n" % (exp_type, exp_type)
#        # for a in params_list[:-1]:
#        for a in params_list:
#            fmax_a = df_fmax[a].dropna().values
#            test_statistic, pval = mannwhitneyu(alpha_1, fmax_a, alternative='greater') 
#            out_str += "%s\t%s\t%s\t%0.3f\t%0.3e\t%0.3e\n" % (alg, max(params_list), str(a), np.median(fmax_a), pval, 
#                    np.median(alpha_1 - fmax_a) if len(alpha_1) == len(fmax_a) else -1)
#        print("appending to %s" % (stat_file))
#        with open(stat_file, 'w') as out:
#            out.write(out_str)
#        print(out_str)
#    else:
#        print("%s already exists. Skipping" % (stat_file))


def results_overview(df, measures=['fmax']):
    """
    Print an overview of the number of values / terms, as well as the median fmax
    """
    print("plot_exp_name\tmeasure\talg\tmedian\t# terms")
    #print("net_version\texp_name\tmeasure\talg\tmedian\t# terms")
    #for plot_exp_name in sorted(df['plot_exp_name'].unique()):
    for plot_exp_name in df['plot_exp_name'].unique():
        df_curr = df[df['plot_exp_name'] == plot_exp_name]
        #net_version, exp_name = df_curr['net_version'].unique()[0], df_curr['exp_name'].unique()[0]
        #if len(plot_exp_name) > len(net_version + exp_name):
        # limit the goterms to those that are also present for SinkSource(?)
        for measure in measures:
            for alg in sorted(df_curr['Algorithm'].unique()):
                df_alg = df_curr[df_curr['Algorithm'] == alg][measure]
                print("%s\t%s\t%s\t%0.3f\t%d" % (plot_exp_name, measure, alg, df_alg.median(), len(df_alg)))
                #print("%s\t%s\t%s\t%s\t%0.3f\t%d" % (net_version, exp_name, measure, alg, df_alg.median(), len(df_alg)))


print_warning = True
def get_algs_to_run(alg_settings, **kwargs):
    global print_warning
    # if there aren't any algs specified by the command line (i.e., kwargs),
    # then use whatever is in the config file
    if kwargs['algs'] is None:
        algs_to_run = run_eval_algs.get_algs_to_run(alg_settings)
        kwargs['algs'] = [a.lower() for a in algs_to_run]
        if print_warning:
            print("\nNo algs were specified. Using the algorithms in the yaml file:")
            print(str(kwargs['algs']))
            print_warning = False 
        if len(algs_to_run) == 0:
            print("ERROR: Must specify algs with --alg or by setting 'should_run' to [True] in the config file")
            sys.exit("Quitting")
    else:
        # make the alg names lower so capitalization won't make a difference
        kwargs['algs'] = [a.lower() for a in kwargs['algs']]
    return kwargs['algs']


def load_all_results(input_settings, alg_settings, output_settings, prec_rec_str="", **kwargs):
    """
    Load all of the results for the datasets and algs specified in the config file
    """
    df_all = pd.DataFrame()
    algs = get_algs_to_run(alg_settings, **kwargs)
    for dataset in input_settings['datasets']:
        for alg in algs:
            if alg not in alg_settings:
                print("%s not found in config file. Skipping" % (alg))
                continue
            alg_params = alg_settings[alg]
            if kwargs.get('cross_validation_folds'):
                folds = kwargs.get('cross_validation_folds')
                cv_seed = kwargs.get('cv_seed')
                neg_factor = kwargs.get('sample_neg_examples_factor')
                for rep in range(1,kwargs.get('num_reps',1)+1):
                    if cv_seed is not None:
                        curr_seed = cv_seed + rep-1
                    eval_type = cv.get_output_prefix(folds, rep, neg_factor, curr_seed)
                    df = load_alg_results(
                        dataset, alg, alg_params, prec_rec_str=prec_rec_str,
                        results_dir=output_settings['output_dir'],
                        eval_type=eval_type, **kwargs,
                        #only_terms=kwargs.get('only_terms'), postfix=kwargs.get('postfix',''),
                    )
                    add_dataset_settings(dataset, df) 
                    df['rep'] = rep
                    df_all = pd.concat([df_all, df])
            else:
                eval_type = "" 
                if kwargs.get('loso'):
                    eval_type = 'loso'
                df = load_alg_results(
                    dataset, alg, alg_params, prec_rec_str=prec_rec_str, 
                    results_dir=output_settings['output_dir'],
                    eval_type=eval_type, **kwargs,
                    #only_terms=kwargs.get('only_terms'), postfix=kwargs.get('postfix',''),
                )
                add_dataset_settings(dataset, df) 
                df_all = pd.concat([df_all, df])
    return df_all


def add_dataset_settings(dataset, df):
    # also add the net version and exp_name
    df['net_version'] = dataset['net_version']
    df['exp_name'] = dataset['exp_name']
    if 'net_settings' in dataset and 'weight_method' in dataset['net_settings']:
        df['weight_method'] = dataset['net_settings']['weight_method'] 
    # if they specified a name to use in the plot for this experiment, then use that
    plot_exp_name = "%s %s" % (dataset['net_version'], dataset['exp_name'])
    if 'plot_exp_name' in dataset:
        plot_exp_name = dataset['plot_exp_name']
    df['plot_exp_name'] = plot_exp_name
    return df


def load_alg_results(
        dataset, alg, alg_params, prec_rec_str="", 
        results_dir='outputs', eval_type="cv-5folds",
        only_terms=None, postfix='', **kwargs):
    """
    For a given dataset and algorithm, build the file path and load the results
    *prec_rec_str*: postfix to change file name. Usually 'prec-rec' if loading precision recal values
    *results_dir*: the base output directory
    *eval_type*: The string specifying the evaluation type. For example: 'cv-5folds' or 'th' for temporal holdout
    *terms*: a set of terms for which to limit the output
    """
    alg_name = alg
    # if a name is specified to use when plotting, then get that
    if 'plot_name' in alg_params:
        alg_name = alg_params['plot_name'][0]
    elif alg in ALG_NAMES:
        alg_name = ALG_NAMES[alg]

    out_dir = "%s/%s/%s" % (
        results_dir, dataset['net_version'], dataset['exp_name'])
    combos = [dict(zip(alg_params.keys(), val))
        for val in itertools.product(
            *(alg_params[param] for param in alg_params))]
    print("%d combinations for %s" % (len(combos), alg))
    # load the CV file for each parameter combination for this algorithm
    df_all = pd.DataFrame()
    for param_combo in combos:
        # first get the parameter string for this runner
        params_str = runner.get_runner_params_str(alg, dataset, param_combo)
        cv_file = "%s/%s/%s%s%s%s.txt" % (out_dir, alg, eval_type, params_str, postfix, prec_rec_str)
        if not os.path.isfile(cv_file):
            print("\tnot found %s - skipping" % (cv_file))
            continue
        print("\treading %s" % (cv_file))
        df = pd.read_csv(cv_file, sep='\t')
        # remove any duplicate rows
        df.drop_duplicates(inplace=True)
        # hack to get the script to plot just the parameter value
        if kwargs.get('compare_param') is not None:
            df['Algorithm'] = str(param_combo[kwargs['compare_param']])
            df['alg_name'] = alg_name
        elif len(combos) == 1: 
            df['Algorithm'] = alg_name
        else:
            df['Algorithm'] = alg_name + params_str
        df_all = pd.concat([df_all, df])
    return df_all


def mad(vals):
    med = np.median(vals)
    mad = np.median([abs(val - med) for val in vals])
    return mad


if __name__ == "__main__":
    config_map, kwargs = parse_args()

    main(config_map, **kwargs)
