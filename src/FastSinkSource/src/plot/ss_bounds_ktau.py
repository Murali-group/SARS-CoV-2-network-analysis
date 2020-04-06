# Plot results from the SinkSource Bounds compare ranks - LOSO

from collections import defaultdict
import argparse
import os, sys
from tqdm import tqdm
#import utils.file_utils as utils
# also compute the significance of sinksource vs local
#from scipy.stats import kruskal, mannwhitneyu
# plotting imports
import matplotlib
matplotlib.use('Agg')  # To save files remotely.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# make this the default for now
sns.set_style('darkgrid')
# my local imports
fss_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,fss_dir)
from src.plot import plot_utils


# TODO use the ax that was passed in
#def main(config_map, ax=None, out_pref='', **kwargs):
def main(config_map, out_pref='', **kwargs):
    input_settings, alg_settings, output_settings, out_pref, kwargs = plot_utils.setup_variables(
        config_map, out_pref, **kwargs)
    alg = "sinksource_bounds"

    # load the results using the plt_utils
    ranks_str = "-ranks"
    df_all = plot_utils.load_all_results(
        input_settings, alg_settings, output_settings,
        prec_rec_str=ranks_str, **kwargs)
    df = df_all

    if out_pref is not None:
        # type of sinksource-squeeze rank comparison
        if alg_settings[alg]['rank_all'][0] is True:
            exp_type = "all-"
        elif alg_settings[alg]['rank_pos_neg'][0] is True:
            exp_type = "pos-neg-"
        else:
            print("ERROR: must speficy either 'rank_all' or 'rank_pos_neg'") 
            sys.exit("Quitting")
        # TODO how should I handle multiple parameters?
        alpha = alg_settings[alg]['alpha'][0]
        out_pref = "%s%sa%s" % (out_pref, exp_type, alpha)
        print("out_pref: %s" % (out_pref))

    # keep only the GO terms with at least 10 annotations
    #df = df[df['num_pos'] >= 10]
    # get all goterm-taxon pairs
    df['goterm-taxon'] = df['#goterm'] + '-' + df['taxon'].map(str)
    print("%d GO terms, %d taxon, %d GO term-taxon pairs" % (df['#goterm'].nunique(), df['taxon'].nunique(), df['goterm-taxon'].nunique()))
    print(df.columns)
    print(df.head(2))
    print(df[df.columns[:9]].head(2))
    #print(df[['#goterm', 'iter', 'kendalltau']].head(200))

    # for each goterm, get the iteration at which kendalltau hits 95%, 99% and 100%
    iter_70 = get_iteration_at_cutoff(df, 0.70, col_to_get='iter', cutoff_col='kendalltau', less_than=False)
    iter_80 = get_iteration_at_cutoff(df, 0.80, col_to_get='iter', cutoff_col='kendalltau', less_than=False)
    iter_90 = get_iteration_at_cutoff(df, 0.90, col_to_get='iter', cutoff_col='kendalltau', less_than=False)
    iter_95 = get_iteration_at_cutoff(df, 0.95, col_to_get='iter', cutoff_col='kendalltau', less_than=False)
    iter_99 = get_iteration_at_cutoff(df, 0.99, col_to_get='iter', cutoff_col='kendalltau', less_than=False)
    iter_100 = get_iteration_at_cutoff(df, 1.0, col_to_get='iter', cutoff_col='kendalltau', less_than=False)
    df_cutoffs = pd.DataFrame({
        '0.70': iter_70, '0.80': iter_80, '0.90': iter_90, 
        '0.95': iter_95, '0.99': iter_99, '1.0': iter_100})
    df_cutoffs = df_cutoffs[['0.70', '0.80', '0.90', '0.95', '0.99', '1.0']]
    #     - Also plot the total # of iterations it takes to fix all node ranks
    total_iters = df.groupby('goterm-taxon')['iter'].max()
    df_cutoffs['Fixed ordering'] = total_iters
    # for each goterm, get the iteration at which kendalltau hits 95%, 99% and 100%
    for col in df_cutoffs.columns:
        print("%s median: %d (%d values)" % (col, df_cutoffs[col].median(), df_cutoffs[col].dropna().count()))
    #print(df_cutoffs.head())

    plot(df_cutoffs, out_pref, alpha=alpha)


def get_axes_to_plot():
    # TODO add subplots to a grid if passed in
    # insert a break into the plot to better show the small and large ranges
    f, (ax2, ax1) = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(5,4.5))
    plt.subplots_adjust(hspace=0.05)
    return f, ax1, ax2


def plot(df_cutoffs, out_pref, alpha=0.95):
    f, ax1, ax2 = get_axes_to_plot()
    sns.boxplot(data=df_cutoffs, ax=ax1, order=['0.70', "0.80", '0.90', '0.95', '0.99', '1.0', 'Fixed ordering'], fliersize=1.5)
    sns.boxplot(data=df_cutoffs, ax=ax2, order=['0.70', "0.80", '0.90', '0.95', '0.99', '1.0', 'Fixed ordering'], fliersize=1.5)
    # f, ax1 = plt.subplots()
    # sns.boxplot(df_cutoffs, ax=ax1)

    ymin, ymax = ax1.get_ylim()
    # ax1.set_ylim(-2, df_cutoffs['1.0'].max())
    ax1.set_ylim(-2, df_cutoffs['0.99'].max()+5)
    #ax1.set_ylim(0, 75)
    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(df_cutoffs['0.99'].max()+10, ymax)
    #ax2.set_ylim(df_cutoffs['Fixed ordering'].min()-5, ymax)
    #ax2.set_ylim(df_cutoffs['1.0'].min()-10, ymax)

    # now get the fancy diagonal lines
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    plt_kwargs = dict(transform=ax2.transAxes, color='tab:gray',
                      clip_on=False, zorder=10)
    # ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax2.plot((+d, -d), (-d, +d), **plt_kwargs)        # top-left diagonal
    ax2.plot((1 + d, 1 - d), (-d, +d), **plt_kwargs)  # top-right diagonal

    plt_kwargs.update(transform=ax1.transAxes)  # switch to the bottom axes
    ax1.plot((+d, -d), (1 - d, 1 + d), **plt_kwargs)  # bottom-left diagonal
    ax1.plot((1 + d, 1 - d), (1 - d, 1 + d), **plt_kwargs)  # bottom-right diagonal

    xlabel = r"Kendall's $\tau$ cutoff compared to fixed ordering"
    ylabel = r"# Iterations, $\alpha$ = %s" % (alpha)
    # ax1.set_xlabel(xlabel)
    # ax1.set_ylabel(ylabel)
    # ax1.set_title("# iterations to \n rank nodes correctly")
    # plt.suptitle("%d goterms, %s, %s k\n%s" % (df['#goterm'].nunique(), alg, k_str, exp_name))
    f.text(0.02, 0.5, ylabel, va='center', rotation='vertical', fontsize=12, fontweight='bold')
    f.text(0.5, 0.02, xlabel, ha='center', fontsize=12, fontweight='bold')
    # plt.tight_layout()

    # out_file = all_ranks_file.replace('.txt', '-%s.png' % (h))
    # out_file = "%s/%s-%s-loso-pos-neg-ktau-cutoffs-boxplots.png" % (out_dir, version, h)
    out_file = "%s-ktau-cutoffs-boxplots.pdf" % (out_pref)
    print("Writing figure to %s" % (out_file))
    plt.savefig(out_file)
    #plt.show()
    plt.close()


def plot_series(s, title='', xlabel='', ylabel='', out_file=None):
    fig, ax = plt.subplots()
    s.index += 1
    s.plot()
    # also add an inlet 
    s.index -= 1
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if s.min() < 1e-10:
        ax.set_yscale('log')
    plt.tight_layout()
    #plt.show()
    if out_file is not None:
        print("writing figure to %s" % (out_file))
        plt.savefig(out_file)
    plt.close()


def get_iteration_at_cutoff(df, cutoff, col_to_get='kendalltau', cutoff_col='max_d', less_than=True):
    """
    *less_than*: If True, find the first occurance <= cutoff. Otherwise find >= cutoff
    """
    val_at_cutoff = {}
    # UPDATE: should include both the goterm and taxon
    #for goterm_taxon in tqdm(df['goterm-taxon'].unique()):
    #    goterm_df = df[df['goterm-taxon'] == goterm_taxon]
    #    for v1, v2 in goterm_df[[col_to_get, cutoff_col]].values:
    goterm_taxons = df['goterm-taxon'].unique()
    df = df[['goterm-taxon', col_to_get, cutoff_col]]
    g = df.groupby(['goterm-taxon'])
    for goterm_taxon in tqdm(goterm_taxons):
        goterm_df = g.get_group(goterm_taxon)
        for v1, v2 in goterm_df[[col_to_get, cutoff_col]].values:
            if (less_than is True and v2 <= cutoff) \
               or (less_than is False and v2 >= cutoff):
                val_at_cutoff[goterm_taxon] = v1
                #print("%s: %s: %s; %s: %s" % (goterm, col_to_get, v1, cutoff_col, v2))
                #pdb.set_trace()
                #return
                break
        if goterm_taxon not in val_at_cutoff:
            val_at_cutoff[goterm_taxon] = v1
    #print(len(val_at_cutoff))
    #sys.exit()
    return val_at_cutoff


def setup_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description='Script for setting up various string experiments')

        parser.add_argument('--config', required=True,
            help='Configuration file')
    #group = parser.add_argument_group('masterscript options')
    #group.add_argument('--script-to-run', default="FastSinkSource/run_eval_algs.py",
    #        help="script to run when submitting to screen / qsub")
    return parser


def parse_args():
    parser = plot_utils.setup_opts()
    parser = setup_parser(parser)
    args = parser.parse_args()
    kwargs = vars(args)
    config_map = plot_utils.load_config_file(kwargs['config'])
    # TODO check to make sure the inputs are correct in config_map

    #if opts.exp_name is None or opts.pos_neg_file is None:
    #    print("--exp-name, --pos-neg-file, required")
    #    sys.exit(1)
    if kwargs['measure'] is None:
        kwargs['measure'] = ['fmax']
    kwargs['measures'] = kwargs['measure']
    del kwargs['measure']

    return config_map, kwargs


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)
