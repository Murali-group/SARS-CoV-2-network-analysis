
import argparse
import yaml
from collections import defaultdict
import os
import sys
#from tqdm import tqdm
import copy
import time
#import numpy as np
#from scipy import sparse
import pandas as pd
#import subprocess


# packages in this repo
sys.path.insert(0,"/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/")
from setup_datasets import setup_dataset_files
from utils.parse_utils import run_command
from FastSinkSource.src.utils.config_utils import get_algs_to_run

def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)

    with open(args.config, 'r') as conf:
        config_map = yaml.load(conf, Loader=yaml.FullLoader)
        # config_map = yaml.load(conf)
    # TODO check to make sure the inputs are correct in config_map

    return config_map, kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="Script to download and parse input files, and (TODO) run the FastSinkSource pipeline using them.")

    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/config-files/master-config-yeast.yaml",
                       help="Configuration file for this script.")
    group.add_argument('--download-only', action='store_true', default=False,
                       help="Stop once files are downloaded and mapped to UniProt IDs.")
    group.add_argument('--force-download', action='store_true', default=False,
                       help="Force re-downloading and parsing of the input files")

    group = parser.add_argument_group('FastSinkSource Pipeline Options')
    group.add_argument('--alg', '-A', dest='algs', type=str, action="append",
                       help="Algorithms for which to get results. Must be in the config file. " +
                       "If not specified, will get the list of algs with 'should_run' set to True in the config file")
    group.add_argument('--stats-only', action='store_true', default=False,
                       help="Rather than run the entire FSS pipeline, just print out statistics about the network size and such. " +
                       "Useful to parse the networks and setup the sparse matrices")
    group.add_argument('--force-run', action='store_true', default=False,
                       help="Force re-running the FSS pipeline, and re-writing the associated config files")

#    # additional parameters
#    group = parser.add_argument_group('Additional options')
#    group.add_argument('--forcealg', action="store_true", default=False,
#            help="Force re-running algorithms if the output files already exist")
#    group.add_argument('--forcenet', action="store_true", default=False,
#            help="Force re-building network matrix from scratch")
#    group.add_argument('--verbose', action="store_true", default=False,
#            help="Print additional info about running times and such")

    return parser


def main(config_map, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """
    dataset_settings = config_map['dataset_settings']
    datasets_dir = dataset_settings['datasets_dir']

    # Download and parse the ID mapping files 
    # Download, parse, and map (to uniprot) the network files 
    setup_dataset_files(datasets_dir, dataset_settings['datasets_to_download'], dataset_settings.get('mappings'), **kwargs)

    if kwargs.get('download_only'):
        return
    # Now setup the config file to run the FastSinkSource pipeline using the specified networks
    # For now I will assume some of the essential structure is already setup (i.e., committed to the repo) 
    # TODO setup the pos-neg file(s) automatically (the file containing the positive and negative examples for which to use when running FSS)
    fss_settings = config_map['fastsinksource_pipeline_settings']
    # Before writing the config file, setup the geneset files and their settings for the config file 
    geneset_settings = fss_settings.get('genesets_to_test')
    if geneset_settings is not None:
        fss_settings['genesets_to_test'] = setup_geneset_config(
            datasets_dir, dataset_settings['datasets_to_download'], geneset_settings,
            fss_settings, **kwargs)

    config_files = setup_fss_config(datasets_dir, dataset_settings['datasets_to_download'], fss_settings, **kwargs) 

    # Now run FSS on each of the config files
    # TODO allow for parallelization of multiple networks / algorithms
    print("\nRunning the FastSinkSource pipeline on %d config files" % (len(config_files)))
    if len(config_files) > 0:
        print("\tsee the respective log files for more details")
    for config_file in config_files:
        log_file = config_file.replace('.yaml', '.log')
        command = "python -u src/FastSinkSource/run_eval_algs.py "  + \
                  " --config %s " % (config_file) + \
                  " %s " % ("--forcealg" if kwargs.get('force_run') else "") + \
                  " %s " % ("--stats-only" if kwargs.get('stats_only') else "") + \
                  " >> %s 2>&1 " % (log_file)
        run_command(command) 

    # Now test for overlap of the top predictions with related gene sets
    if geneset_settings is not None:
        if len(config_files) > 0:
            print("\nTesting the overlap of the top predictions of each method with %d gene sets" % (
                len(geneset_settings)))
        for config_file in config_files:
            command = "python -u src/Enrichment/enrichment.py "  + \
                    " --config %s " % (config_file) + \
                    " %s " % ("--force-run" if kwargs.get('force_run') else "") + \
                    " >> %s 2>&1 " % (log_file)
            run_command(command) 
    print("Finished")


def setup_geneset_config(
        datasets_dir, dataset_settings, genesets_to_run,
        fss_settings, **kwargs):
    """
    Setup the config options to test for enrichment of FastSinkSource algorithm predictions 
        with the specified genesets
    *returns*: an updated geneset_settings list with dictionaries of names, 
        file paths to the gene sets, and other settings
    """
    new_genesets_settings = []
    # Instead of a list of genesets to run, change to a dictionary with the name as the key
    download_genesets_settings = {}
    for geneset in dataset_settings.get('genesets', []) + dataset_settings.get('drug-targets', []):
        download_genesets_settings[geneset['name']] = geneset
    for geneset in genesets_to_run:
        name = geneset['name']
        # Possibly types are "genesets", "drug-targets"
        data_type = geneset.get('data_type', 'genesets')
        if name not in download_genesets_settings:
            print("WARNING: %s with the name '%s' not found " % (data_type, name) +
                  "in 'datasets_to_download' section of config file. Skipping") 
            continue

        if 'gmt_file' in download_genesets_settings[name]:
            file_name = download_genesets_settings[name]['gmt_file'] 
        else:
            file_name = download_genesets_settings[name]['file_name'] 
        gmt_file = "%s/%s/%s/%s" % (datasets_dir, data_type, name, file_name) 
        print(name, file_name)
        new_gmt_file = "%s/genesets/%s/%s" % (fss_settings['input_dir'], name, file_name) 
        # TODO allow to filter out some gene sets
        # don't need to do anything if it already exists since this is just a symlink
        if not os.path.isfile(new_gmt_file):
            print("\tadding symlink from %s to %s" % (gmt_file, new_gmt_file))
            if not os.path.isfile(gmt_file):
                print("ERROR: %s does not exist. Quitting" % (gmt_file))
                sys.exit()
            os.makedirs(os.path.dirname(new_gmt_file), exist_ok=True)
            os.symlink(os.path.abspath(gmt_file), new_gmt_file)
        # now setup the settings
        ## remove the fss_inputs dir since that is specified as a variable in the config file
        #new_gmt_file = '/'.join(new_gmt_file.split('/')[1:])
        geneset['gmt_file'] = os.path.basename(new_gmt_file)
        new_genesets_settings.append(geneset)

    return new_genesets_settings


def setup_fss_config(datasets_dir, dataset_settings, fss_settings, **kwargs):
    """
    Setup the config file(s) to run the FastSinkSource pipeline

    *returns*: a list of config files that are ready to be used to run FSS
    """

    fss_dir = fss_settings['input_dir']
    config_dir = "%s/config_files" % (fss_dir)
    # start setting up the config map settings that will remain the same for all datasets
    algs = get_algs_to_run(fss_settings['algs'], **kwargs)
    # make sure should_run is set to True
    for alg in algs:
        fss_settings['algs']['should_run'] = [True]
    config_map = {
        'input_settings': {'input_dir': fss_dir,
            'datasets': [],},
        'output_settings': {'output_dir': fss_settings['output_dir']},
        # only keep the specified algorithms in the config file
        'algs': {alg: fss_settings['algs'][alg] for alg in algs},
    }
    if 'genesets_to_test' in fss_settings:
        config_map['genesets_to_test'] = fss_settings['genesets_to_test']
    eval_str = ""
    if fss_settings.get('eval_settings'):
        # append a string of the evaluation settings specified to the yaml file so you can run multiple in parallel.
        # TODO also add a postfix parameter
        eval_s = fss_settings.get('eval_settings')
        config_map['eval_settings'] = eval_s
        eval_str = "%s%s%s%s" % (
            "-cv%s" % eval_s['cross_validation_folds'] if 'cross_validation_folds' in eval_s else "",
            "-nf%s" % eval_s['sample_neg_examples_factor'] if 'sample_neg_examples_factor' in eval_s else "",
            "-nr%s" % eval_s['num_reps'] if 'num_reps' in eval_s else "",
            "-seed%s" % eval_s['cv_seed'] if 'cv_seed' in eval_s else "",
            )
    # for now, use the same pos_neg_file and base experiment name for every network 
    base_dataset_settings = {
        'exp_name': fss_settings['exp_name'],
        'pos_neg_file': fss_settings['pos_neg_file'],
        'net_files': [],
        'string_net_files': [],
    }

    pos_neg_file = "%s/%s" % (fss_dir, fss_settings['pos_neg_file'])
    print("Reading %s" % (pos_neg_file))
    krogan_nodes = pd.read_csv(pos_neg_file, sep='\t', index_col=None, header=0)['prots']
    print("\t%d 'krogan' nodes" % (len(krogan_nodes)))

    # Instead of a list of networks to run, change to a dictionary with the name as the key
    download_net_settings = {net['name']: net for net in dataset_settings['networks']}
    download_drug_target_settings = {dt['name']: dt for dt in dataset_settings.get('drug-targets',{})}
    # now make the config files, and then call run_eval_algs.py
    config_files = []
    for net in fss_settings['networks_to_run']:
        net_config_map = copy.deepcopy(config_map)
        names = net['names']
        # extract the dataset settings to get the file paths to given networks specified
        dataset_net_files, dataset_string_files = get_net_filepaths(
            names, download_net_settings, datasets_dir+'/networks')
        drug_targets = net.get('drug_target_names')
        if drug_targets is not None:
            drug_target_files, _ = get_net_filepaths(
                drug_targets, download_drug_target_settings, datasets_dir+'/drug-targets', drug_targets=True)
        # if specified, use the given net_version
        net_version = net.get('net_version')
        if net_version is not None:
            name = net_version
        else:
            name = '-'.join(names)
        net_settings = net.get('net_settings')
        if net.get('network_collection'):
            if len(names) > 1:
                print("WARNING: combining multiple networks with a collection is not yet implemented.\nQuitting")
                sys.exit()
            #name = names[0]
            base_dir = "%s/networks/%s" % (datasets_dir, names[0])
            base_net_files = "%s/net-files.txt" % (base_dir)
            net_files = pd.read_csv(base_net_files, header=None, squeeze=True)
            net_files = ["%s/%s" % (base_dir, f) for f in net_files]

            for net_file in net_files:
                curr_net_files = [net_file]
                file_name = os.path.basename(net_file).split('.')[0]
                net_version = name+'/'+file_name
                if drug_targets is not None:
                    setup_drug_target_nets(
                        fss_dir, net_version, drug_target_files,
                        curr_net_files+dataset_string_files, nodes_to_remove=krogan_nodes)
                    curr_net_files += drug_target_files 
                curr_settings = setup_net_settings(
                    fss_dir, copy.deepcopy(base_dataset_settings), net_version,
                    net_files=curr_net_files, dataset_net_settings=net_settings)
                curr_settings['exp_name'] += "-%s" % (os.path.basename(net_file).split('.')[0])
                net_config_map['input_settings']['datasets'].append(curr_settings) 
        else:
            if drug_targets is not None:
                # for drug target files, only include the edges where the target is in the network
                setup_drug_target_nets(
                    fss_dir, name, drug_target_files, dataset_net_files+dataset_string_files,
                    nodes_to_remove=krogan_nodes)
                dataset_net_files += drug_target_files
            curr_settings = setup_net_settings(
                fss_dir, copy.deepcopy(base_dataset_settings), name,
                net_files=dataset_net_files, string_net_files=dataset_string_files,
                dataset_net_settings=net_settings, plot_exp_name=net_settings.get('plot_exp_name'))
            net_config_map['input_settings']['datasets'].append(curr_settings) 
        # now write the config file
        config_file = "%s/%s%s.yaml" % (config_dir, name, eval_str)
        if kwargs.get('force_run') is not True and os.path.isfile(config_file):
            print("'%s' already exists. Use --force-run to overwite and run 'run_eval_algs.py'." % (config_file))
            # if --stats-only is specified, then run_eval_algs.py
            # will be run with this config file using the --stats-only option
            if not kwargs.get('stats_only'):
                continue
        write_yaml_file(config_file, net_config_map)
        config_files.append(config_file)
    return config_files


# extract the dataset settings to get the file paths to given networks specified
def get_net_filepaths(names, download_net_settings, datasets_dir, drug_targets=False):
    net_files = []
    string_net_files = []
    for name in names:
        if name not in download_net_settings:
            print("ERROR: network name '%s' not found in the " % (name) +
                    "'datasets_to_download' section of the config file. " +
                    "\n\tAvailable names: '%s'. \nQuitting" % (
                        "', '".join(download_net_settings.keys())))
            sys.exit()
        net_file = "%s/%s/%s" % (datasets_dir, name, download_net_settings[name]['file_name']) 
        # run_eval_algs.py can now handle a file without an edge weight column
        #if drug_targets is True:
        #    # make a new file with all 1s added as the weight column
        #    drug_targets_net_file = net_file.split('.')[0] + '-net.tsv'
        #    print("Copying %s to %s and adding a column of 1s to make it a network file" % (
        #        net_file, drug_targets_net_file)) 
        #    with open(drug_targets_file, 'r') as f:
        #        with open(drug_targets_net_file, 'w') as out:
        #            for line in f:
        #                line = line.rstrip().split('\t')
        #                out.write("%s\t%s\t%s\n" % (line[0], line[1], "1"))
        if download_net_settings[name].get('string_networks') is True:
            string_net_files.append(net_file)
        else:
            net_files.append(net_file)
    return net_files, string_net_files


def write_yaml_file(yaml_file, config_map):
    # make sure the directory exists first
    os.makedirs(os.path.dirname(yaml_file), exist_ok=True)
    print("Writing %s" % (yaml_file))
    with open(yaml_file, 'w') as out:
        yaml.dump(config_map, out, default_flow_style=False)


def setup_drug_target_nets(
        input_dir, net_version, drug_target_files, net_files, nodes_to_remove=None, forced=False):
    print("Setting up drug-target networks")
    not_setup_files = []
    for drug_target_file in drug_target_files:
        file_name = os.path.basename(drug_target_file)
        new_drug_target_file = "%s/networks/%s/%s" % (input_dir, net_version, file_name)
        if not forced and os.path.isfile(new_drug_target_file):
            print("%s already exists. Not overwriting it." % (new_drug_target_file))
            continue
        else:
            not_setup_files.append(new_drug_target_file)
    if len(not_setup_files) == 0:
        return

    # first get the nodes from the networks
    all_prots = set()
    # leave this network as unweighted, but set the weight as the largest weight in the network
    # so the edges are treated similarly to the weighted network
    largest_weight = 1
    for net_file in net_files:
        print("Reading prots/nodes from %s" % (net_file))
        df = pd.read_csv(net_file, sep='\t', header=None)
        prots = set(df[df.columns[0]].values) | set(df[df.columns[1]].values)
        all_prots.update(prots)
        # TODO set this up to automatically extract the edge weight
        if 'string' in net_file:
            largest_weight = 1000
    print("\t%d total prots" % (len(all_prots)))

    for new_drug_target_file in not_setup_files:
        print("Reading %s and limiting edges to those with a target in the given %d prots" % (
            drug_target_file, len(all_prots)))
        # first column should be the UniProt ID, second is the Drug ID
        df = pd.read_csv(drug_target_file, sep='\t', header=None, skiprows=1)
        num_drug_targets = len(df)
        num_drugs, num_targets = df[1].nunique(), df[0].nunique()
        df = df[df[df.columns[0]].isin(all_prots)]
        df['weights'] = largest_weight
        os.makedirs(os.path.dirname(new_drug_target_file), exist_ok=True)
        if nodes_to_remove is not None:
            # UPDATE: Also remove the targets that are Krogan nodes (i.e., SARS-CoV-2 - Human PPIs)
            # so that we get the drug targets of our predictions.
            # Keep the "all-targets" file to be able to visualize both later (i.e., posting to graphspace)
            new_drug_target_file2 = new_drug_target_file.replace('.tsv','-all-targets.tsv')
            print("\t%d drug-targets (%d drugs, %d targets) limited to %d (%d drugs, %d targets)" % (
                num_drug_targets, num_drugs, num_targets, len(df), df[1].nunique(), df[0].nunique()))
            print("\twriting to %s" % (new_drug_target_file2))
            df.to_csv(new_drug_target_file2, sep='\t', index=None, header=False)
            df = df[~df[df.columns[0]].isin(nodes_to_remove)]

        print("\t%d drug-targets (%d drugs, %d targets) limited to %d (%d drugs, %d targets)" % (
            num_drug_targets, num_drugs, num_targets, len(df), df[1].nunique(), df[0].nunique()))
        print("\twriting to %s" % (new_drug_target_file))
        df.to_csv(new_drug_target_file, sep='\t', index=None, header=False)
    return 


def setup_net_settings(
        input_dir, dataset_settings, net_version,
        net_files=None, string_net_files=None, dataset_net_settings=None,
        plot_exp_name=None):
    """
    Add the network file and all other network settings to this config file
    Will create a symbolic link from the original dataset 
    """
    num_net_files = 0
    for net_file_group, net_type in [(net_files, ''), (string_net_files, 'string')]:
        if net_file_group is None:
            continue
        for net_file in net_file_group:
            file_name = os.path.basename(net_file)
            new_net_file = "%s/networks/%s/%s" % (input_dir, net_version, file_name)
            if not os.path.isfile(new_net_file):
                print("\tadding symlink from %s to %s" % (net_file, new_net_file))
                if not os.path.isfile(net_file):
                    print("ERROR: %s does not exist. Quitting" % (net_file))
                    sys.exit()
                os.makedirs(os.path.dirname(new_net_file), exist_ok=True)
                os.symlink(os.path.abspath(net_file), new_net_file)
            dataset_settings['net_version'] = "networks/"+net_version

            # The FSS pipeline uses this to distinguish the file with all individual string channels
            if net_type == 'string':
                dataset_settings['string_net_files'].append(file_name)
            else:
                dataset_settings['net_files'].append(file_name)
            num_net_files += 1 
    if dataset_net_settings is not None:
        dataset_settings['net_settings'] = dataset_net_settings
    if plot_exp_name is not None:
        dataset_settings['plot_exp_name'] = plot_exp_name
    # finally clean up the unused options
    if (string_net_files is None or len(string_net_files) == 0) and 'string_net_files' in dataset_settings:
        del dataset_settings['string_net_files']
    if (net_files is None or len(net_files) == 0) and 'net_files' in dataset_settings:
        del dataset_settings['net_files']
    return dataset_settings


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)
