# For each target species, plot the fmax boxplot distributions and # annotations being evaluated.

from collections import defaultdict
import itertools
import argparse
import os, sys
from tqdm import tqdm
import numpy as np
#import utils.file_utils as utils
# also compute the significance of sinksource vs local
from scipy.stats import kruskal, mannwhitneyu, wilcoxon
# plotting imports
import matplotlib
matplotlib.use('Agg')  # To save files remotely.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# make this the default for now
sns.set_style('darkgrid')

# my local imports
from . import plot_utils
from ..evaluate import eval_leave_one_species_out as eval_loso
from .. import setup_sparse_networks as setup
from .. import main as run_eval_algs


measure_map = plot_utils.measure_map


def main(config_map, ax=None, out_pref='', **kwargs):
    # copy the alg names over from plot utils
    kwargs['alg_names'] = plot_utils.ALG_NAMES
    input_settings, alg_settings, output_settings, out_pref, kwargs = plot_utils.setup_variables(
        config_map, out_pref, **kwargs)

    # should only have one dataset.
    # TODO run once per dataset
    if len(input_settings['datasets']) > 1:
        print("WARNING: currently only works with one dataset, and %d were found in the config file. Using the first one" % (
            len(input_settings['datasets'])))
        input_settings['datasets'] = [input_settings['datasets'][0]]

    # load the results using plot_utils
    df_all = plot_utils.load_all_results(
        input_settings, alg_settings, output_settings, **kwargs)
    df = df_all
    # also get the annotation matrix
    dataset, in_dir = input_settings['datasets'][0], input_settings['input_dir']
    # store the ann obj files so they don't have to be recreated each time
    out_dir = "outputs/viz/loso/%s-%s/" % (
            dataset['net_version'].split('/')[-1], dataset['exp_name'].split('/')[-1])
    os.makedirs(out_dir, exist_ok=True)
    ann_obj, eval_ann_obj = load_ann_datasets(
            out_dir, dataset, in_dir, 
            alg_settings, **kwargs)

    #ann_obj, eval_ann_obj, goid_names_file = load_annotations(dataset, in_dir, alg_settings, **kwargs) 
    kwargs['goid_names_file'] = get_goid_names_file(in_dir, dataset)
    # if eval_ann_obj is specified, then use it to get stats instead of the ann_obj
    curr_ann_obj = eval_ann_obj if eval_ann_obj is not None else ann_obj
    # get the # ann per species
    taxon_file = "%s/%s" % (in_dir, dataset['taxon_file'])
    only_taxon_file = "%s/%s" % (in_dir, dataset['only_taxon_file'])
    species_to_uniprot_idx = eval_loso.get_uniprot_species(taxon_file, ann_obj)
    selected_species, taxons = eval_loso.get_selected_species(
        species_to_uniprot_idx, only_taxon_file, kwargs.get('taxons'))
    kwargs['sp_names'] = selected_species
    kwargs['sp_abbrv'] = {t: ''.join(
        subs[0] for subs in sp_name.split(' ')[:2]) for t, sp_name in selected_species.items()}
    taxon_num_ann = get_taxon_ann_counts(curr_ann_obj, species_to_uniprot_idx, taxons, **kwargs)
    pos_mat = (ann_obj.ann_matrix > 0).astype(int)
    goid_num_ann = {ann_obj.goids[i]: num for i, num in enumerate(np.ravel(pos_mat.sum(axis=1)))}

    print("\t%d algorithms, %d plot_exp_name values\n" % (df_all['Algorithm'].nunique(), len(df_all['plot_exp_name'].unique())))
    #print(df_all.head())
    #print(df_all.columns)
    plot_utils.results_overview(df_all, measures=kwargs['measures'])

    if kwargs.get('title'):
        title = kwargs['title']
    else:
        #title = '-'.join(df_all['plot_exp_name'].unique())
        title = df_all['plot_exp_name'].unique()[0] \
                .replace('net_versions/', '') \
                .replace('neg_experiments/','') \
                .replace(' ','\n')
    #if not kwargs.get('for_paper'):
    #    title += " \n %d%s %s" % (
    #            num_terms, ' %s'%kwargs.get('only_terms_name', ''),
    #            "sp-term pairs" if kwargs['exp_type'] == 'loso' else 'terms')
    if not kwargs.get('for_paper'):
        kwargs['title'] = title

    if out_pref is not None:
        print("out_pref: %s" % (out_pref))
    kwargs['out_pref'] = out_pref

    #results_overview(ev_code_results, measures=measures, **kwargs) 
    generate_plots(df, taxon_num_ann, goid_num_ann, **kwargs) 


def load_annotations(dataset, input_dir, alg_settings, **kwargs):
    # don't need to load the net_obj, just get the prots from the node ids file
    #_, ann_obj, eval_ann_obj = run_eval_algs.setup_dataset(dataset, input_dir, alg_settings, **kwargs) 
    net_files = ["%s/%s/%s" % (input_dir, dataset['net_version'], net_file) for net_file in \
                dataset.get('net_files',[]) + dataset.get('string_net_files',[])]
    # get the nodes from the network node-ids file
    if dataset.get('multi_net'):
        if 'node_ids_file' in dataset['net_settings']:
            node_ids_file = dataset['net_settings']['node_ids_file']
        else:
            # get the node_ids_file using the file path that would've been created
            string_cutoff = dataset['net_settings'].get('string_cutoff', 150) 
            out_pref = "%s/sparse-nets/c%d-" % (os.path.dirname(net_files[0]), string_cutoff)
            node_ids_file = "%snode-ids.txt" % (out_pref)
    else:
        node_ids_file = net_files[0].replace('.gz','.npz') + "-node-ids.txt"
    print("Reading node ids file from %s" % (node_ids_file))
    nodes = pd.read_csv(node_ids_file, header=None, index_col=None, sep='\t')[0].values
    _, ann_obj, eval_ann_obj = run_eval_algs.load_annotations(nodes, dataset, input_dir, **kwargs)

    #summary_file = "inputs/pos-neg/%s/pos-neg-10-summary-stats.tsv" % (curr_ev_codes)
    #kwargs['goid_names_file'] =  goid_names_file
    # if there are no annotations, then skip this dataset
    if len(ann_obj.goids) == 0:
        print("No annotations found. Quitting")
        sys.exit()
    # don't need to worry about negative examples here because we're only using positive examples
#    # if specified, remove negative examples that are neighbors of positive examples
#    if kwargs.get('rem_neg_neighbors'):
#        ann_obj = experiments.rem_neg_neighbors(net_obj, ann_obj)
#        if eval_ann_obj is not None:
#            eval_ann_obj = experiments.rem_neg_neighbors(net_obj, eval_ann_obj)
#    if kwargs.get('youngs_neg'):
#        obo_file = kwargs['youngs_neg']
#        ann_obj = experiments.youngs_neg(ann_obj, obo_file, "%s/%s" % (input_dir,dataset['pos_neg_file']))
#        if eval_ann_obj is not None:
#            eval_ann_obj = experiments.youngs_neg(eval_ann_obj, obo_file, "%s/%s" % (input_dir,dataset['pos_neg_file']))
    return ann_obj, eval_ann_obj, goid_names_file


def get_goid_names_file(input_dir, dataset):
    pos_neg_file = "%s/%s" % (input_dir, dataset['pos_neg_file'])
    pos_neg_file = pos_neg_file.replace("bp-",'').replace("mf-",'')
    # pos-neg-bp-10-list.tsv
    if '-list' in pos_neg_file:
        goid_names_file = pos_neg_file.replace("-list","-summary-stats")
    elif '.gz' in pos_neg_file:
        goid_names_file = pos_neg_file.replace(".tsv.gz","-summary-stats.tsv")
    else:
        goid_names_file = pos_neg_file.replace(".tsv","-summary-stats.tsv")
    return goid_names_file


def get_taxon_ann_counts(ann_obj, species_to_uniprot_idx, taxons, **kwargs):
    """
    Get the total # protein-term annotation pairs per species
    """
    # make sure to keep only the pos examples i.e., annotations
    pos_mat = (ann_obj.ann_matrix > 0).astype(int)
    print("%s ann for all taxon" % (pos_mat.sum()))
    taxon_num_ann = {}
    for t in taxons:
        taxon_prot_idx = np.asarray(list(species_to_uniprot_idx[t]))
        # sum over the columns to get the # ann per gene
        #ann_prots = np.ravel(ann_obj.ann_matrix.sum(axis=0))
        # get the total number of annotations for this taxon
        taxon_ann_mat = pos_mat[:,taxon_prot_idx]
        # each annotation is a 1 and the rest are 0s, so just need to take the sum
        num_ann = taxon_ann_mat.sum()
        taxon_num_ann[t] = num_ann 
        if num_ann > 0:
            print("\t%s: %s ann (%d taxon prots)" % (t, num_ann, len(taxon_prot_idx)))
    return taxon_num_ann


def generate_plots(df, taxon_num_ann, goid_num_ann, 
        out_pref='', measures=['fmax'], **kwargs):
    #cutoffs_data, out_dir = ev_code_results[(version, ev_codes, eval_ev_codes, h)]
    sort_taxon_by_fmax = None    

#         title = "Evaluation of recovery of %s ev. codes from %s ev. codes %s\n for 19 pathogenic bacteria. - %s  %s\n %d GO terms with %d <= # annotations < %d %s" % (
#             eval_ev_codes, ev_codes, use_neg_str, version, keep_ann, df_curr['goid'].nunique(), cutoff1, cutoff2, "overall" if split_overall else "")
    #recov_str = " recovery of %s ev. codes from" % (eval_ev_codes) if eval_ev_codes != ""  else ""

    #for i, (cutoff1, cutoff2) in enumerate(cutoffs):
    #    df_curr, species_exp_counts, species_comp_counts, species_iea_counts  = cutoffs_data[i]
    #ann_stats = {'EXPC': species_exp_counts}
    #title = kwargs.get('title', '')
    #if kwargs['for_pub'] is False:
    #    title = "Evaluation of%s %s ev. codes\n for 19 pathogenic bacteria. - %s\n %d %s GO terms with %d+ annotations" % (
    #        recov_str, ev_codes, version, df_curr['goid'].nunique(), h.upper(), cutoff1)
    #out_file = ""
    for measure in measures:
        print("Creating plots for '%s'" % (measure))

        out_file = "%sloso-%s-%dalgs%s.pdf" % (
                out_pref, measure, df['Algorithm'].nunique(), kwargs.get('plot_postfix',''))
        stat_file = out_file.replace('.pdf', '-sig.txt')
        # get comparative p-values
        # UPDATE: limit to taxons that have at least 5 terms
        # TODO make an option for this
        sp_term_cutoff = 3
        df_taxon_cutoff = pd.concat(dfT for taxon, dfT in df.groupby('#taxon') \
                                    if dfT['#goid'].nunique() < sp_term_cutoff)
                                    #if dfT['#goid'].nunique() >= sp_term_cutoff)
        sig_results, sig_species = eval_stat_sig(
            df, stat_file, measure=measure,
            sort_taxon_by_fmax=sort_taxon_by_fmax, **kwargs)

        # only make this plot if there are < 20 species
        if df_taxon_cutoff['#taxon'].nunique() < 20:
            plot_fmax_eval(df_taxon_cutoff, out_file, measure=measure,  # title=title,
                    sort_taxon_by_fmax=sort_taxon_by_fmax, ann_stats=taxon_num_ann, 
                    sig_species=sig_species, **kwargs)
        else:
            print("skipping species-specific distributions since %d species > 20" % (df_taxon_cutoff['#taxon'].nunique()))

        out_file = "%s/stats/loso-%s-%s-%s%s.pdf" % (
                out_pref, measure, kwargs['alg1'], kwargs['alg2'], kwargs.get('plot_postfix',''))
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        if kwargs.get('compare_taxon_median'):
            out_file = out_file.replace('.pdf','-taxon-median.pdf')
            # summarize the fmax values by the median per species
            scatterplot_taxon_fmax(df, measure=measure, 
                    out_file=out_file, **kwargs)
        else:
            scatterplot_fmax(df, goid_num_ann, measure=measure, 
                    out_file=out_file, **kwargs)


def eval_stat_sig(
        df_curr, out_file, measure='fmax', sp_term_cutoff=5, 
        sort_taxon_by_fmax=None, alg1='fastsinksource', alg2='localplus', **kwargs):
    sp_pval = {}
    out_str = ""

    combinations = list(itertools.combinations(kwargs['algs'], 2))
    out_str += "#alg1\talg2\tpval\tCorrected p-value (x%d)\n" % (len(combinations))
    # Don't think sorting is needed
    #df_curr.sort_values('goid', inplace=True) 
    for a1, a2 in combinations:
        df1 = df_curr[df_curr['Algorithm'] == kwargs['alg_names'].get(a1, a1)]
        df2 = df_curr[df_curr['Algorithm'] == kwargs['alg_names'].get(a2, a2)]
        # make sure they line up
        df1 = df1.loc[df1.index.isin(df2.index)]
        df2 = df2.loc[df2.index.isin(df1.index)]
        a1_fmax = df1[measure]
        a2_fmax = df2[measure]
        #test_statistic, pval = mannwhitneyu(a1_fmax, a2_fmax, alternative='greater') 
        test_statistic, pval = wilcoxon(a1_fmax, a2_fmax, alternative='greater') 
        out_str += "%s\t%s\t%0.3e\t%0.3e\n" % (a1, a2, pval, pval*len(combinations))

    print(out_str)
    # also compare individual species
    curr_species = df_curr['#taxon'].unique()
    # limit the species for which we run the test to those that have at least 5 terms
    species_with_terms = set([s for s, df_s in df_curr.groupby('#taxon') \
                              if df_s['#goid'].nunique() >= sp_term_cutoff])
    print("%d species with at least %d terms" % (len(species_with_terms), sp_term_cutoff))
    if sort_taxon_by_fmax is not None:
        curr_species = sort_taxon_by_fmax
    out_str += "Species\tAlg1\tAlg2\tAlg1-med\tAlg2-med\tRaw p-value\tCorrected p-value (x%d)\n" % (len(species_with_terms))
    for s in curr_species:
        #name = f_settings.NAME_TO_SHORTNAME2.get(selected_species[str(s)],'-')
        name = kwargs['sp_names'].get(str(s),'-') if 'sp_names' in kwargs else '-'
        df_s = df_curr[df_curr['#taxon'] == s]
        df1 = df_s[df_s['Algorithm'] == kwargs['alg_names'].get(a1, a1)]
        df2 = df_s[df_s['Algorithm'] == kwargs['alg_names'].get(a2, a2)]
        # make sure they line up
        df1 = df1.loc[df1.index.isin(df2.index)]
        df2 = df2.loc[df2.index.isin(df1.index)]
        a1_fmax = df1[measure]
        a2_fmax = df2[measure]
        if s in species_with_terms:
        #try:
            #test_statistic, pval = mannwhitneyu(a1_fmax, a2_fmax, alternative='greater') 
            # TODO requires an updated version of scipy
            test_statistic, pval = wilcoxon(a1_fmax, a2_fmax, alternative='greater') 
            line = "%s\t%s\t%s\t%0.3f\t%0.3f\t%0.2e\t%0.2e" % (name, alg1, alg2, a1_fmax.median(), a2_fmax.median(), pval, pval*len(species_with_terms))
            sp_pval[s] = pval
        # this only really happens when there is only 1 or two terms with the same fmax
        #except ValueError:
        else:
            line = "%s\t%s\t%s\t%0.3f\t%0.3f\t-\t-" % (name, alg1, alg2, a1_fmax.median(), a2_fmax.median())
            pval = 1
        out_str += line+'\n'

    if kwargs.get('forceplot') or not os.path.isfile(out_file):
        print("writing to %s" % (out_file))
        with open(out_file, 'w') as f:
            f.write(out_str)
    else:
        print("Would've written to %s. Use --forceplot to overwrite" % (out_file))
    print(out_str)

    # now check how many are significant
    sig_species = set()
    for s, pval in sp_pval.items():
        # TODO implement the BH correction
        if pval*len(species_with_terms) < 0.05:
            sig_species.add(s)
    print("\t%d species with pval*%d < 0.05" % (len(sig_species), len(species_with_terms)))
    return sp_pval, sig_species


def plot_fmax_eval(
        df_curr, out_file, measure='fmax', sort_taxon_by_fmax=None,
        ann_stats=None, sig_species=None, **kwargs):
    """
    *ann_stats*: Set of annotation types (either 'EXP', 'COMP' or 'IEA') for which a boxplot will be added to the right # of annotations
    *for_pub*: If true, the title at the top will not be included
    """
    if kwargs.get('forceplot') or not os.path.isfile(out_file):
        print("Writing figure to %s" % (out_file))
    else:
        print("Already exists: %s Use --forceplot to overwrite" % (out_file))
        return
    # flip the x and y of the axis
    horiz = kwargs.get('horizontal')
    algs = kwargs.get('algs', df_curr['Algorithm'].unique())
    # add the species name to the boxplot
    species_labels = {}
    for s in df_curr['#taxon']:
        species = kwargs['sp_abbrv'].get(str(s),s) if 'sp_abbrv' in kwargs else s
        species_labels[s] = "%s.%s(%d)" % (
            species, ' ' if not horiz else '\n',
            df_curr[df_curr['#taxon'] == s]['#goid'].nunique())
    species_labels = pd.Series(species_labels)
    df_curr['species'] = df_curr['#taxon'].apply(lambda x: species_labels[x])
    df_curr = df_curr.sort_values(by=['species', 'Algorithm'], ascending=[True, False])
    # sort the species by fmax median of the first algorithm (sinksource)
    if sort_taxon_by_fmax is None:
        sort_taxon_by_fmax = df_curr[df_curr['Algorithm'] == plot_utils.ALG_NAMES.get(algs[0],algs[0])].groupby('#taxon')[measure].median().sort_values(ascending=False).index
        # now get the species order from the taxon order
    df_curr['#taxon'] = df_curr['#taxon'].astype("category")
    df_curr['#taxon'].cat.set_categories(sort_taxon_by_fmax, inplace=True)
    sort_by_med_fmax = df_curr.sort_values(by='#taxon')['species'].unique()

    # and get the colors
    curr_palette = [plot_utils.alg_colors[alg] for alg in algs]
#         fig, ax = plt.subplots(figsize=(6,10))
#         fig, ax = plt.subplots(figsize=(5,6))
    # make the figure taller to fit all the species
    height = 8 if df_curr['#taxon'].nunique() > 21 else 6
    # temporary fix to fit more algorithms
    if len(algs) > 4:
        height += 4
    if horiz:
        fig, ax = plt.subplots(figsize=(height,4))
    else:
        fig, ax = plt.subplots(figsize=(4,height))
    x,y = measure, 'species'
    if horiz:
        sns.boxplot(x=y,y=x, order=sort_by_med_fmax[::-1], 
                hue='Algorithm', hue_order=[plot_utils.ALG_NAMES.get(a,a) for a in algs], 
                data=df_curr, orient='v', fliersize=1.5,
                palette=curr_palette, saturation=0.9)
    else:
        sns.swarmplot(x=x,y=y, order=sort_by_med_fmax, dodge=True,
                    hue='Algorithm', hue_order=[plot_utils.ALG_NAMES.get(a,a) for a in algs], 
                    data=df_curr, orient='h', 
                    palette=curr_palette)
        #sns.boxplot(x=x,y=y, order=sort_by_med_fmax, 
                    #data=df_curr, orient='h', fliersize=1.5,
    ax.set_title(kwargs['title'])
    xlabel = measure_map.get(measure, measure.upper())
    ylabel = 'Species (# GO Terms)'
    if horiz:
        xlabel,ylabel = ylabel,xlabel 
    size = 16 if kwargs.get('for_paper') else 12
    ax.set_xlabel(xlabel, fontsize=size, weight="bold")
    ax.set_ylabel(ylabel, fontsize=size, weight="bold")
    #plt.legend(bbox_to_anchor=(.3, 0.9))
    # the ticklabels are the species names. 
    # now try and bold the ones that are significant
    ticklabels = ax.get_yticklabels() if not horiz else ax.get_xticklabels()
    ticklabels = [item.get_text() for item in ticklabels]
    print(ticklabels)
    taxons = df_curr.sort_values(by='#taxon')['#taxon'].unique()
    taxons = taxons[::-1] if horiz else taxons
    for i, s in enumerate(taxons):
        # if this is not a significant species, then add latex to make it not bold
        if s not in sig_species:
            #ticklabels[i] = r"$\\rm %s$" % (ticklabels[i]) 
            ticklabels[i] = "$\\rm %s$" % (
                    ticklabels[i].replace('\n','$\n$'))
    print(ticklabels)
    ticks = np.arange(0,1.01,.2 if kwargs.get('for_paper') else 0.1)
    if horiz:
        ax.set_xticklabels(ticklabels, fontweight="bold")
        plt.setp(ax, yticks=ticks, yticklabels=["%0.1f"%x for x in ticks])
    else:
        ax.set_yticklabels(ticklabels, fontweight="bold")
        plt.setp(ax, xticks=ticks, xticklabels=["%0.1f"%x for x in ticks])
    if kwargs.get('for_paper'):
        ax.tick_params(axis='y', labelsize='large')
        ax.tick_params(axis='x', labelsize='large')
        # remove the legend title
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[:], labels=labels[:])

    if ann_stats is not None:
        # add another plot to the right that is a bar plot of the # of non-iea (or exp) annotations
        df_counts = pd.DataFrame(pd.Series(ann_stats))
        #df_counts = pd.DataFrame([ann_stats])
            #df_counts = pd.DataFrame([ann_stats[ev_code]]).T
            #df_counts.columns = [ev_code]
        df_counts['#taxon'] = df_counts.index
        df_counts = df_counts[df_counts['#taxon'].astype(int).isin(species_labels.index)]
        df_counts.index = range(len(df_counts.index))
        df_counts = pd.melt(df_counts, id_vars="#taxon", var_name="Ev. codes", value_name="# ann")
        df_counts['species'] = df_counts['#taxon'].apply(lambda x: species_labels[int(x)])
        y,x = ('species','# ann')
        if horiz:
            right_ax = plt.axes([.125, .9, .775, .125])
            sns.barplot(x=y, y=x, data=df_counts, ax=right_ax, 
                    hue="Ev. codes", order=list(sort_by_med_fmax)[::-1], 
                    orient='v',
                    palette=["#9c6d57"], saturation=0.95, edgecolor='k')
        else:
            right_ax = plt.axes([.95, .13, .2, .75])
            sns.barplot(x=x, y=y, data=df_counts, ax=right_ax, 
                    hue="Ev. codes", order=list(sort_by_med_fmax), 
                    orient='h',
                    palette=["#9c6d57"], saturation=0.95, edgecolor='k')
        change_width(right_ax, 0.45, horiz=horiz)
        xlabel,ylabel = ("","# Ann") if horiz else ("# Ann","")
        right_ax.set_ylabel(ylabel)
        right_ax.set_xlabel(xlabel)
        if horiz:
            right_ax.set_xticks([])
        else:
            right_ax.set_yticks([])
        if df_counts['Ev. codes'].nunique() < 2:
            right_ax.get_legend().remove()
        # TODO add the ev_code type 
        #else:
        #    right_ax.set_xlabel("# Ann")

    plt.savefig(out_file, bbox_inches='tight')
    #plt.show()
    plt.close()
    return


def scatterplot_fmax(df_curr, goid_num_ann, measure='fmax', 
        out_file=None, alg1="sinksource", alg2="localplus", **kwargs):
    # figure out which GO terms have the biggest difference for all species
    # plot a scatter plot of the differences across all GO terms
    alg1 = kwargs['alg_names'].get(alg1, alg1)
    alg2 = kwargs['alg_names'].get(alg2, alg2)
    print("\nComparing %s values of %s and %s" % (measure, alg2, alg1))
    df_alg2 = df_curr[df_curr['Algorithm'] == alg2]
    # key: taxon, goid tuple. Value: fmax
    alg2_scores = dict(zip(zip(df_alg2['#taxon'], df_alg2['#goid']), df_alg2[measure]))
    df_alg1 = df_curr[df_curr['Algorithm'] == alg1]
    alg1_scores = dict(zip(zip(df_alg1['#taxon'], df_alg1['#goid']), df_alg1[measure]))

    goid_diffs = {}
    for taxon, goid in df_curr[['#taxon', '#goid']].values:
        if (taxon, goid) not in alg1_scores and goid not in alg2_scores:
            if kwargs['verbose']:
                print("WARNING: %s not in both %s and %s." % (goid, alg1, alg2))
            continue
        if (taxon, goid) not in alg1_scores:
            if kwargs['verbose']:
                print("WARNING: %s not in %s" % (goid, alg1))
            continue
        if (taxon, goid) not in alg2_scores:
            if kwargs['verbose']:
                print("WARNING: %s not in %s." % (goid, alg2))
            continue
        goid_diff = alg1_scores[(taxon, goid)] - alg2_scores[(taxon, goid)]
    #     goid_diff = (alg1_scores[(taxon, goid)] - alg2_scores[(taxon, goid)]) / float(alg2_scores[(taxon, goid)])
        goid_diffs[(taxon, goid)] = goid_diff
    print("\t%d %s, %d %s, %d diffs" % (len(alg2_scores), alg2, len(alg1_scores), alg1, len(goid_diffs)))

    #goid_num_ann = dict(zip(df_summary['GO term'], df_summary['# positive examples']))
    goid_taxon_num_ann = dict(zip(zip(df_alg2['#taxon'], df_alg2['#goid']), df_alg2['# test ann']))
    # also load the summary of the GO terms
    if kwargs.get('goid_names_file'):
    #summary_file = "inputs/pos-neg/%s/pos-neg-10-summary-stats.tsv" % (curr_ev_codes)
        df_summary = pd.read_csv(kwargs['goid_names_file'], sep='\t')
        goid_names = dict(zip(df_summary['GO term'], df_summary['GO term name']))
    else:
        goid_names = {g: '-' for t, g in goid_taxon_num_ann}

    alg1_col = '%s %s'%(alg1,measure)
    alg2_col = '%s %s'%(alg2,measure)
    diff_col = '%s - %s %s'%(alg1, alg2, measure)
    df = pd.DataFrame({alg2_col: alg2_scores, alg1_col: alg1_scores, diff_col: goid_diffs})
    # print the # and % where SS is >, <, and = local+
    print("%s > %s: %d (%0.3f)"% (alg1, alg2, len(df[df[diff_col] > 0]), len(df[df[diff_col] > 0]) / float(len(df))))
    print("%s < %s: %d (%0.3f)"% (alg1, alg2, len(df[df[diff_col] < 0]), len(df[df[diff_col] < 0]) / float(len(df))))
    print("%s = %s: %d (%0.3f)"% (alg1, alg2, len(df[df[diff_col] == 0]), len(df[df[diff_col] == 0]) / float(len(df))))

    print("\nTop 5 difference in f-max:")
    print(''.join(["%s, %s\t%s\t%s\t%s\n" % (
        t, g, goid_names[g], goid_num_ann[g], goid_diffs[t, g]) for t, g in sorted(
        goid_diffs, key=goid_diffs.get, reverse=True)[:5]]))

    stats_file = out_file.replace('.pdf','.tsv')
    if kwargs.get('forceplot') or not os.path.isfile(stats_file):
        print("Writing to %s" % (stats_file))
        with open(stats_file, 'w') as out:
            out.write("#taxon\tgoid\tname\t# ann\ttaxon # ann\t%s\t%s\tdiff\n" % (alg2, alg1))
            out.write(''.join(["%s\n" % (
                '\t'.join(str(x) for x in [t, g, goid_names[g], goid_num_ann[g], goid_taxon_num_ann[(t,g)],
                alg2_scores[(t,g)], alg1_scores[(t,g)], goid_diffs[(t,g)]])
            ) for t, g in sorted(goid_diffs, key=goid_diffs.get, reverse=True)]))
    else:
        print("Already exists: %s Use --forceplot to overwrite" % (stats_file))

    # I can't get only the top histogram to change and not the right, so I have to make two copies of the file.
    # One with bins of 10 for the top, and the other with the regular # of bins
    #for bins in [None, 10]:
    #    grid = sns.jointplot(x=alg1_col, y=diff_col, data=df,
    #                stat_func=None, joint_kws={"s": 20}, marginal_kws=dict(bins=bins) if bins is not None else None,
    #                )
    #    # plt.suptitle('%s, %s, %s-%s \n %d species' % (version, ev_codes, cutoff1, cutoff2, df_h_infl['#taxon'].nunique()))

    # I can explicitly make the top and right histograms this way
    g = sns.JointGrid(x=alg1_col, y=diff_col, data=df, height=4.5)
    g.ax_marg_x.hist(df[alg1_col], bins=10, alpha=0.7)
    g.ax_marg_y.hist(df[diff_col], bins=30, 
            orientation="horizontal", alpha=0.7)
    g = g.plot_joint(plt.scatter, s=15)
    plt.tight_layout()
    #g.fig.set_figwidth(5)
    #g.fig.set_figheight(5)
#    out_file2 = out_file
#    if bins is not None:
#        out_file2 = out_file.replace('.pdf', '-10bins.pdf')
    measure = measure_map.get(measure, measure.upper())
    xlabel = r'%s %s'%(alg1,measure)
    ylabel = r'%s - %s %s'%(alg1, alg2, measure)
    size = 16 if kwargs.get('for_paper') else 12
    plt.xlabel(xlabel, fontsize=size, weight="bold")
    plt.ylabel(ylabel, fontsize=size, weight="bold")
    if kwargs.get('for_paper'):
        plt.tick_params(axis='y', labelsize='large')
        plt.tick_params(axis='x', labelsize='large')
        ticks = np.arange(0,1.01,.2)
        plt.xticks(ticks, labels=["%0.1f"%x for x in ticks])

    if kwargs.get('forceplot') or not os.path.isfile(out_file):
        print("Writing to %s" % (out_file))
        plt.savefig(out_file)
    else:
        print("Already exists: %s Use --forceplot to overwrite" % (out_file))
    #plt.show()
    plt.close()


# change the width of the extra bar plot to the side
def change_width(ax, new_value, horiz=False):
    for patch in ax.patches:
        if horiz:
            current_width = patch.get_width()
            diff = current_width - new_value
            # we change the bar width
            patch.set_width(new_value)
            # we recenter the bar
            patch.set_x(patch.get_x() + diff * .5 + 0.08)
        else:
            current_width = patch.get_height()
            diff = current_width - new_value
            # we change the bar width
            patch.set_height(new_value)
            # we recenter the bar
            patch.set_y(patch.get_y() + diff * .5 + 0.08)


def load_ann_datasets(
        out_dir, dataset, input_dir, 
        alg_settings, **kwargs):
    sparse_ann_file = "%s/pos-ann.npz" % (out_dir)
    sparse_ann_file_eval = "%s/eval-pos-ann.npz" % (out_dir)
    if not kwargs.get('forcenet') and \
            os.path.isfile(sparse_ann_file):
        ann_obj = load_ann_obj(sparse_ann_file)
        eval_ann_obj = None 
        if dataset.get('pos_neg_file_eval'):
            eval_ann_obj = load_ann_obj(sparse_ann_file_eval)
    else:
        _, ann_obj, eval_ann_obj = run_eval_algs.setup_dataset(
                dataset, input_dir, alg_settings, **kwargs)
        store_ann_obj(ann_obj, sparse_ann_file, pos_only=True)
        if eval_ann_obj is not None:
            store_ann_obj(eval_ann_obj, sparse_ann_file_eval, pos_only=True)
        # now store them to a file
    return ann_obj, eval_ann_obj


def load_ann_obj(sparse_ann_file):
    print("\nReading annotation matrix from %s" % (sparse_ann_file))
    loaded_data = np.load(sparse_ann_file, allow_pickle=True)
    dag_matrix = setup.make_csr_from_components(loaded_data['arr_0'])
    ann_matrix = setup.make_csr_from_components(loaded_data['arr_1'])
    goids, prots = loaded_data['arr_2'], loaded_data['arr_3']
    ann_obj = setup.Sparse_Annotations(dag_matrix, ann_matrix, goids, prots)
    return ann_obj


def store_ann_obj(ann_obj, sparse_ann_file, pos_only=False):
    print("Writing sparse annotations to %s" % (sparse_ann_file))
    if pos_only:
        print("\tpositive examples only")
        ann_obj.ann_matrix = (ann_obj.ann_matrix > 0).astype(int)
        ann_obj.ann_matrix.eliminate_zeros() 
    # store all the data in the same file
    dag_matrix_data = setup.get_csr_components(ann_obj.dag_matrix)
    ann_matrix_data = setup.get_csr_components(ann_obj.ann_matrix)
    np.savez_compressed(
        sparse_ann_file, dag_matrix_data, 
        ann_matrix_data, ann_obj.goids, ann_obj.prots)


def setup_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description='Script for setting up various string experiments')

        parser.add_argument('--config', required=True,
            help='Configuration file')
    group = parser.add_argument_group('Additional options')
    group.add_argument('--alg1', default="fastsinksource",
            help="First algorithm for the scatterplot")
    group.add_argument('--alg2', default="localplus",
            help="Second alg for the scatterplot")
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
