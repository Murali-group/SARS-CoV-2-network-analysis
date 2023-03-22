
from collections import defaultdict
import os
import sys
import numpy as np
import gzip
import pandas as pd
import subprocess
import glob
import wget
import requests
import shutil
# add this file's directory to the path so these imports work from anywhere
#sys.path.insert(0,os.path.dirname(__file__))
from src.utils import parse_utils as utils

def setup_mappings(datasets_dir, mapping_settings, **kwargs):
    """
    Download and build mappings to/from UniProt protein ids.
    """

    namespace_mappings = defaultdict(set)
    for mapping in mapping_settings:
        mapping_file = "%s/mappings/%s/%s" % (
            datasets_dir, mapping['species'], mapping['mapping_file'])

        # download the file
        if kwargs['force_download'] or not os.path.isfile(mapping_file):
            try:
                utils.download_file(mapping['url'], mapping_file)
                print("downloaded file from:", mapping['url'] )
            except:
                print("Failed to download '%s' using the url '%s'." % (mapping_file, mapping['url']))
                print("Please either try again, manually download the file, or remove this section of the config file. Quitting")
                sys.exit(1)

        # now parse the file
        #namespace_mappings[mapping['namespaces']] =  
        from_uniprot, to_uniprot = parse_mapping_file(
            mapping_file, file_type=mapping['type'], namespaces=mapping['namespaces'],
            sep=mapping.get('sep', '\t'))

        # for now, just store everything together
        for mapping_dict in [from_uniprot, to_uniprot]:
            for id1, alt_ids in mapping_dict.items():
                namespace_mappings[id1].update(alt_ids)
    return namespace_mappings


def parse_mapping_file(mapping_file, file_type='list', namespaces=['all'], sep='\t'):
    """
    *mapping_file*: path to file. Can be gzipped
    *file_type*: type of mapping file. Current can be: 
        'list': a list of mappings (i.e., many-to-many) where the first column is a uniprot ID, 
                second is the namespace, and third is the ID of the other namespace
        'table': a table of 1-to-1 mappings.
    *namespaces*: namespaces to keep from the file. If 'all' is given, then keep all of the mappings. If *file_type* is table, should be the column names. 

    *returns*: two dictionaries of sets. First is uniprot to alternate IDs, second is alternate IDs to uniprot
    """
    print("Reading mapping file %s with these settings: file_type: '%s'; namespaces: '%s', sep: '%s'" % (
        mapping_file, file_type, "', '".join(namespaces), sep))
    to_uniprot = defaultdict(set)
    from_uniprot = defaultdict(set)
    if file_type == "list":
        open_command = gzip.open if '.gz' in mapping_file else open
        with open_command(mapping_file, 'r') as f:
            for line in f:
                line = line.decode() if '.gz' in mapping_file else line
                if line[0] == "#":
                    continue
                uniprot, namespace, alt_id = line.rstrip().split(sep)[:3]
                if 'all' in namespaces or namespace.lower() in namespaces:
                    from_uniprot[uniprot].add(alt_id)
                    to_uniprot[alt_id].add(uniprot)
    elif file_type == "table":
        df = pd.read_csv(mapping_file, sep=sep, header=0) 
        for namespace in namespaces:
            # TODO the "Entry" column is the name of the UniProt ID column when downloaded from UniProt.
            # Need to automatically get the right columns
            for uniprot, alt_id in zip(df['Entry'], df[namespace]):
                to_uniprot[alt_id].add(uniprot)
                from_uniprot[uniprot].add(alt_id)
    else:
        print("ERROR: unrecognized file_type '%s'. Must be either 'list' or 'table'. Quitting" % (file_type))
        sys.exit(1)

    print("\t%d uniprot IDs map to/from %d alternate ids" % (len(from_uniprot), len(to_uniprot)))
    return from_uniprot, to_uniprot


def setup_dataset_files(datasets_dir, dataset_settings, mapping_settings, **kwargs):
    #global namespace_mappings
    # TODO only setup the mappings if they are needed
    namespace_mappings = None 
    if mapping_settings is not None:
        namespace_mappings = setup_mappings(datasets_dir, mapping_settings, **kwargs)

    # TODO combine genesets and drug-targets, since I essentially treat them the same anyway
    if 'genesets' in dataset_settings:
        genesets_dir = "%s/genesets" % (datasets_dir)
        setup_genesets(genesets_dir, dataset_settings['genesets'], namespace_mappings, **kwargs)
    if 'drug-targets' in dataset_settings:
        out_dir = "%s/drug-targets" % (datasets_dir)
        # use the same function to process both genesets and drug targets
        setup_genesets(out_dir, dataset_settings['drug-targets'], namespace_mappings, **kwargs)
    if 'networks' in dataset_settings:
        networks_dir = "%s/networks" % (datasets_dir)
        setup_networks(networks_dir, dataset_settings['networks'], namespace_mappings, **kwargs)

    print("Finished downloading/setting up datasets")
    return


def setup_genesets(drug_targets_dir, drug_target_settings, namespace_mappings, **kwargs):
    """
    Download the genesets / drug target files, and map the genes in them to UniProt IDs, if specified.
    Will write the parsed files in gmt format.
    Currently supports the following file types: gmt, table, wikipathways
    """
    for settings in drug_target_settings:
        drug_targets_file = "%s/%s/%s" % (
            drug_targets_dir, settings['name'], settings['file_name'])
        dataset_dir = os.path.dirname(drug_targets_file)
        # also write a gmt file if the original file is of a different type
        if settings.get('gmt_file') is None:
            # default is to just use a different suffix
            file_end = '.'.join(drug_targets_file.split('.')[1:])
            gmt_file = drug_targets_file.replace(file_end, 'gmt')
            #settings['gmt_file'] = os.path.basename(gmt_file) 
        else:
            gmt_file = "%s/%s" % (dataset_dir, settings['gmt_file'])
            del settings['gmt_file']
        mapping_settings = settings.get("mapping_settings", {}) 
        # add the mapping settings to all settings
        settings.update(mapping_settings)

        # TODO separate downloading from parsing?
        if not kwargs.get('force_download') and os.path.isfile(drug_targets_file):
            print("%s already exists. Use --force-download to overwrite and re-map to uniprot" % (drug_targets_file))
            continue
        if 'url' in settings:
            downloaded_file = "%s/%s" % (dataset_dir, settings['url'].split('/')[-1])
            if 'downloaded_file' in settings:
                downloaded_file = "%s/%s" % (dataset_dir, settings['downloaded_file'])
            if kwargs.get('force_download') and os.path.isfile(downloaded_file):
                print("Deleting %s and its contents" % (dataset_dir))
                shutil.rmtree(dataset_dir)
            # download the file 
            #try:
            utils.download_file(settings['url'], downloaded_file)
            #except:
            #    print("Failed to download '%s' using the url '%s'. Skipping" % (drug_target_files, drug_target['url']))
            #    continue
        unpack_command = settings.get('unpack_command')
        # if its a zip file, unzip first
        if unpack_command is not None and unpack_command != "":
            command = "%s %s" % (unpack_command, os.path.basename(downloaded_file))
            utils.run_command(command, chdir=dataset_dir)

        unpacked_file = settings.get('unpacked_file')
        unpacked_file = downloaded_file if unpacked_file is None else \
                        "%s/%s" % (dataset_dir, unpacked_file)
        all_mapping_stats = setup_geneset(
            unpacked_file, drug_targets_file, namespace_mappings,
            gmt_file=gmt_file, **settings)
        if all_mapping_stats is not None:
            write_mapping_stats(all_mapping_stats, os.path.dirname(drug_targets_file)) 


def setup_geneset(
        geneset_file, new_file, namespace_mappings,
        gmt_file=None, namespace=None, prefer_reviewed=True,
        file_type='gmt', sep='\t', **kwargs):
    """
    *geneset_file*: path/to/original geneset file in edge-list format. First two columns should be the tail and head of the edge
    *new_file*: path/to/new file to write. Should be a .gmt file since that's the format that will be used
        i.e., 1st column: geneset name, 2nd column: description, then a tab-separated list of genes 
    *namespace_mappings*: a dictionary of IDs each mapped to a set of uniprot IDs
    *namespace*: the namespace of the nodes in the genesets.
    *prefer_reviewed*: when mapping, if any of the alternate IDs map to a reviewed UniProt ID, keeping only that one. Otherwise keep all UniProt IDs
    *sep*: the delimiter of columns in the files

    *returns*: a dictionary of mapping statistics
    """
    mapping_stats = None
    if namespace is not None and namespace.lower() not in ['uniprot', 'uniprotkb']:
        if namespace_mappings is None:
            print("WARNING: mappings not supplied. Unable to map from '%s' to UniProt IDs." % (namespace))
            namespace = None

    if file_type == "drugbank_csv":
        print("Reading %s" % (geneset_file))
        # after parsing the file, treat it like a regular table
        df = utils.parse_drugbank_csv(geneset_file, **kwargs)
    if file_type == 'table':
        print("Reading %s" % (geneset_file))
        df = pd.read_csv(geneset_file, sep=sep, index_col=None)
    if file_type == "wikipathways":
        genesets, descriptions = utils.download_wikipathways(geneset_file, new_file, **kwargs) 

    if file_type in ['table', 'drugbank_csv']:
        df2 = filter_and_map_table(
            df, namespace_mappings, namespace=namespace,
            prefer_reviewed=prefer_reviewed, **kwargs)
        # now write to the table to file
        if kwargs.get('columns_to_keep') is not None:
            df2 = df2[kwargs['columns_to_keep']]
        print("\twriting %s" % (new_file))
        df2.to_csv(new_file, sep='\t', index=None)
        # extract the genesets from the table
        genesets, descriptions = extract_genesets_from_table(df2, **kwargs)
        # don't need to map from a different namespace anymore
        namespace = None
        new_file = gmt_file

    if file_type == 'gmt':
        genesets, descriptions = utils.parse_gmt_file(geneset_file)

    all_mapping_stats = {}
    mapped_genesets = genesets
    if namespace is not None:
        for name, genes in genesets.items():
            ids_to_prot, mapping_stats = map_nodes(genes, namespace_mappings, prefer_reviewed=prefer_reviewed)
            # get just the set of uniprot IDs to which the genes map
            prots = set(p for ps in ids_to_prot.values() for p in ps)
            mapped_genesets[name] = prots 
            all_mapping_stats[name] = mapping_stats
    # now write the genesets and descriptions as a gmt file
    utils.write_gmt_file(new_file, mapped_genesets, descriptions)
    # keep track of the mapping statistics
    return all_mapping_stats if len(all_mapping_stats) > 0 else None


def extract_genesets_from_table(df, **kwargs):
    # extract the "genesets" from the table
    geneset_name_col, description_col, genes_col = kwargs['gmt_cols']
    #print(geneset_name_col, description_col, genes_col)
    genesets = {}
    descriptions = {}
    for geneset_name, geneset_df in df.groupby(geneset_name_col):
        genesets[geneset_name] = set(geneset_df[genes_col])
        if description_col != "":
            descriptions[geneset_name] = set(geneset_df[description_col]).pop()
    print("\t%d genesets, %d total genes" % (len(genesets), len(set(g for gs in genesets.values() for g in gs))))
    return genesets, descriptions


def filter_and_map_table(
        df, namespace_mappings, namespace=None,
        prefer_reviewed=None, col_to_map=0,
        filters=None, **kwargs):
    # first apply the filters
    if filters is not None:
        for filter_to_apply in filters:
            col, vals = filter_to_apply['col'], filter_to_apply['vals']
            print("\tkeeping only %s values in the '%s' column" % (str(vals), col))
            df = df[df[col].isin(vals)]
    df2 = df
    #print(df.head())
    num_rows = len(df.index)
    print("\t%d rows in table" % (num_rows))
    # now map the ids to UniProt
    if namespace is not None:
        # the map the ids column to uniprot
        col_to_map = df.columns[col_to_map] if isinstance(col_to_map, int) else col_to_map
        ids_to_map = set(df[col_to_map])
        ids_to_prot, mapping_stats = map_nodes(ids_to_map, namespace_mappings, prefer_reviewed=prefer_reviewed)
        print(mapping_stats)
        # now update the dataframe with UniProt IDs
        df[col_to_map+"_orig"] = df[col_to_map]
        df.set_index(col_to_map, inplace=True)
        orig_cols = df.columns
        df['uniprot'] = pd.Series(ids_to_prot)
        # since there could be cases where a single id maps to multiple ids, we need to expand the rows
        # TODO what if I keep only one of the mapped uniprot IDs instead of allowing for multiple?
        df2 = pd.concat([df['uniprot'].apply(pd.Series), df], axis=1) \
            .drop('uniprot', axis=1) \
            .melt(id_vars=orig_cols, value_name="uniprot") \
            .drop("variable", axis=1) \
            .dropna(subset=['uniprot'])
        # keep the original column name
        df2.rename(columns={'uniprot':col_to_map}, inplace=True)
        # keep the original order
        df2 = df2[[col_to_map]+list(orig_cols)]
        print("\t%d rows mapped to %d rows" % (num_rows, len(df2)))
        print(df2.head())
    return df2


def map_nodes(nodes, namespace_mappings, prefer_reviewed=True):
    """
    *returns*: a dictionary of node ID to a set of uniprot IDs
    """
    num_unmapped_nodes = 0
    if prefer_reviewed: 
        # get all of the reviewed prots 
        reviewed_prots = namespace_mappings.get('reviewed')
        if reviewed_prots is None:
            print("\tWarning: 'prefer_reviewed' was specified, but the reviewed status of the UniProt IDs was not given. Skipping this option.")
    ids_to_prot = {}
    for n in nodes:
        new_ns = namespace_mappings.get(n)
        if new_ns is None or len(new_ns) == 0:
            num_unmapped_nodes += 1
            continue
        if prefer_reviewed and reviewed_prots is not None:
            # if any of the mapped UniProt IDs are reviewed, keep only those
            ns_to_keep = new_ns & reviewed_prots
            new_ns = ns_to_keep if len(ns_to_keep) > 0 else new_ns
        ids_to_prot[n] = list(new_ns)
    new_nodes = set(p for ps in ids_to_prot.values() for p in ps)
    #print("\t%d nodes map to %d new nodes" % (len(nodes), len(new_nodes)))
    #print("\t%d unmapped edges" % (num_unmapped_edges))
    mapping_stats = {'num_nodes': len(nodes), 'num_mapped_nodes': len(new_nodes),
                     'num_unmapped_nodes': num_unmapped_nodes}
    return ids_to_prot, mapping_stats


def write_mapping_stats(all_mapping_stats, out_dir):
    """
    *all_mapping_stats*: a dictionary from the network name or gene set to the mapping statistics 
        e.g., # nodes, # edges before and after mapping
    *out_dir*: folder in which to place the 'mapping-stats.tsv' file
    """
    df = pd.DataFrame(all_mapping_stats).T
    print(df)
    stats_file = "%s/mapping-stats.tsv" % (out_dir)
    print("Writing mapping statistics to %s" % (stats_file))
    df.to_csv(stats_file, sep='\t')
    return df


def setup_networks(networks_dir, network_settings, namespace_mappings, **kwargs):
    """
    Download all specified network files, and map the identifiers of the proteins in them to UniProt IDs, if necessary
    """
    #global namespace_mappings
    for network in network_settings:
        # this is the path to the file that will contain the pre-processed and map network
        network_file = "%s/%s/%s" % (
            networks_dir, network['name'], network['file_name'])
        final_file = network_file
        # if this is a network collecton, then network file should be the name to give to the zip file with the collection inside
        if network.get('network_collection') is True:
            downloaded_file = network_file
            # multiple networks will be extracted, and a 'net-files.txt' file
            # will be placed in the dir with the names of all the networks.
            # If that exists, then the download and mapping process must have finished
            final_file = "%s/net-files.txt" % (os.path.dirname(network_file))
        # If this isn't a network collection, then download the file and keep the original filename
        else:
            downloaded_file = "%s/%s" % (os.path.dirname(network_file), network['url'].split('/')[-1])

        net_dir = os.path.dirname(final_file)
        if not kwargs.get('force_download') and os.path.isfile(final_file):
            print("%s already exists. Use --force-download to overwrite and re-map to uniprot" % (final_file))
            continue
        # if setting up the networks didn't finish (i.e., it was killed early), then start over
        elif not os.path.isfile(final_file) and os.path.isdir(net_dir):
            print("%s not found. Deleting %s and its contents" % (final_file, net_dir))
            shutil.rmtree(net_dir)
        # otherwise, if force_download was used, then start over.
        elif kwargs.get('force_download') and os.path.isdir(net_dir):
            print("Deleting %s and its contents" % (net_dir))
            shutil.rmtree(net_dir)

        # download the file 
        #try:
        utils.download_file(network['url'], downloaded_file)
        #except:
        #    print("Failed to download '%s' using the url '%s'. Skipping" % (network_file, network['url']))
        #    continue

        unpack_command = network.get('unpack_command')
        # if its a zip file, unzip first
        if unpack_command is not None and unpack_command != "":
            command = "%s %s" % (unpack_command, os.path.basename(downloaded_file))
            utils.run_command(command, chdir=os.path.dirname(downloaded_file))

        mapping_settings = network.get('mapping_settings', {})
        opts = network.get('collection_settings', {})
        if network.get('network_collection') is True:
            setup_network_collection(
                network_file, namespace_mappings,
                namespace=network.get('namespace'), gzip_files=opts.get('gzip_files'),
                weighted=network.get('weighted'),
                prefer_reviewed=mapping_settings.get('prefer_reviewed'),
                remove_filename_spaces=opts.get('remove_filename_spaces'),
                columns_to_keep=network.get('columns_to_keep'),
                sep=network.get('sep')
            ) 
        else:
            mapping_stats = setup_network(
                downloaded_file, network_file, namespace_mappings,
                namespace=network.get('namespace'), 
                prefer_reviewed=mapping_settings.get('prefer_reviewed'),
                weighted=network.get('weighted'),
                columns_to_keep=network.get('columns_to_keep'), sep=network.get('sep')
            )

            if mapping_stats is not None:
                all_mapping_stats = {os.path.basename(network_file).split('.')[0]: mapping_stats}
                write_mapping_stats(all_mapping_stats, os.path.dirname(network_file)) 
    return 


def setup_network_collection(
        collection_file, namespace_mappings, namespace=None,
        gzip_files=False, weighted=False,
        prefer_reviewed=True, remove_filename_spaces=False,
        columns_to_keep=None, sep='\t', **kwargs):
    """
    *collection_file*: the original archive file
    *namespace_mappings*: a dictionary of IDs each mapped to a set of uniprot IDs
    *namespace*: the namespace of the nodes in the networks
    *prefer_reviewed*: when mapping, if any of the alternate IDs map to a reviewed UniProt ID, keeping only that one. Otherwise keep all UniProt IDs
    *gzip_files*: gzip the individual files
    *remove_filename_spaces*: if there are spaces in the file names, remove them
    *columns_to_keep*: a list of indexes of columns to keep in the new file. Should be >= 2 (first two columns should be the tail and head of the edge) 
    *sep*: the delimiter of columns in the files
    """
    collection_dir = os.path.dirname(collection_file)
    if namespace is not None:
        print("Mapping each of the networks in %s to UniProt IDs" % (collection_dir))
        if prefer_reviewed is True:
            print("\t'prefer_reviewed': if any of the alternate IDs map to a reviewed UniProt ID, keeping only that one. Otherwise keep all UniProt IDs") 
    # keep the original files in a separate directory
    orig_dir = "%s/orig_files" % (collection_dir)
    os.makedirs(orig_dir, exist_ok=True)
    files = [f for f in glob.glob("%s/*.*" % (collection_dir)) if f != collection_file]
    # keep track of the network files and the mapping statistics
    net_files = [] 
    # this will be a dictionary from the network name to the mapping statistics (i.e., # nodes, # edges)
    all_mapping_stats = {}
    for f in files:
        new_f = f
        # move the file into the orig_files dir
        orig_f = "%s/%s" % (orig_dir, os.path.basename(f))
        shutil.move(f, orig_f)
        if remove_filename_spaces:
            new_f = new_f.replace(' - ', '-').replace(' ','-') \
                         .replace('(','').replace(')','') \
                         .lower()
        mapping_stats = setup_network(
            orig_f, new_f, namespace_mappings, namespace=namespace, prefer_reviewed=prefer_reviewed,
            weighted=weighted, columns_to_keep=columns_to_keep, sep=sep)

        # gzip the original file to save on space
        utils.gzip_file(orig_f, orig_f+'.gz', remove_orig=True)
        # also gzip the new file if specified
        if gzip_files:
            utils.gzip_file(new_f, new_f+'.gz', remove_orig=True)
            new_f += ".gz"

        f_name = os.path.basename(new_f)
        net_files.append(f_name)
        if mapping_stats is not None:
            all_mapping_stats[f_name.split('.')[0]] = mapping_stats 

    with open('%s/net-files.txt' % (collection_dir), 'w') as out:
        out.write('\n'.join(net_files)+'\n')
    if len(all_mapping_stats) > 0:
        write_mapping_stats(all_mapping_stats, collection_dir) 
    print("")
    #return df


def setup_network(
        network_file, new_file, namespace_mappings, namespace=None,
        prefer_reviewed=True, weighted=False, columns_to_keep=None, sep='\t'):
    """
    *network_file*: path/to/original network file in edge-list format. First two columns should be the tail and head of the edge
    *new_file*: path/to/new file to write. 
    *namespace_mappings*: a dictionary of IDs each mapped to a set of uniprot IDs
    *namespace*: the namespace of the nodes in the networks
    *prefer_reviewed*: when mapping, if any of the alternate IDs map to a reviewed UniProt ID, keeping only that one. Otherwise keep all UniProt IDs
    *weighted*: T/F. If False, add a column of all 1s (i.e., unweighted) after the first two columns. 
        If True, the first column in *columns_to_keep* should be the weights
    *columns_to_keep*: a list of indexes of columns to keep in the new file. Should be >= 2 (first two columns should be the tail and head of the edge) 
    *sep*: the delimiter of columns in the files

    *returns*: a dictionary of mapping statistics
    """
    mapping_stats = None

    # read in the file
    # the edges should be the first two columns
    edges = []
    # store the specified extra columns
    extra_cols = []
    print("Reading %s" % (network_file))
    open_command = gzip.open if '.gz' in network_file else open
    with open_command(network_file, 'r') as f:
        for line in f:
            line = line.decode() if '.gz' in network_file else line
            if line[0] == '#':
                continue
            line = line.rstrip().split(sep)
            u,v = line[:2]
            edges.append((u,v))
            if columns_to_keep is not None:
                extra_cols.append(list(np.asarray(line)[columns_to_keep]))
            else:
                extra_cols.append(line[2:] if len(line) > 2 else [])
    # map to uniprot if its in a different namespace
    new_edges, new_extra_cols = edges, extra_cols

    if namespace is not None and namespace.lower() not in ['uniprot', 'uniprotkb']:
        if namespace_mappings is None:
            print("WARNING: mappings not supplied. Unable to map from '%s' to UniProt IDs." % (namespace))
        else:
            new_edges, new_extra_cols, mapping_stats = map_network(edges, extra_cols, namespace_mappings, prefer_reviewed=prefer_reviewed)

    print("\twriting %s" % (new_file))
    # and re-write the file with the specified columns to keep
    open_command = gzip.open if '.gz' in new_file else open
    with open_command(new_file, 'wb' if '.gz' in new_file else 'w') as out:
        for i, e in enumerate(new_edges):
            new_line = "%s\t%s%s\n" % (
                '\t'.join(e), "1\t" if weighted is False else "",
                # if weighted is True, then the first column in new_extra_cols should be the weight
                '\t'.join(new_extra_cols[i]))
            out.write(new_line.encode() if '.gz' in new_file else new_line)

    # keep track of the mapping statistics
    return mapping_stats


def map_network(edges, extra_cols, namespace_mappings, prefer_reviewed=True):
    # keep track of some mapping stats
    num_unmapped_edges = 0
    new_edges = []
    new_extra_cols = []
    if prefer_reviewed: 
        # get all of the reviewed prots 
        reviewed_prots = namespace_mappings.get('reviewed')
        if reviewed_prots is None:
            print("\tWarning: 'prefer_reviewed' was specified, but the reviewed status of the UniProt IDs was not given. Skipping this option.")
    for i, (u,v) in enumerate(edges):
        # the mappings have a set of ids to which the old id maps to
        new_us = namespace_mappings.get(u)
        new_vs = namespace_mappings.get(v)
        if new_us is None or new_vs is None:
            num_unmapped_edges += 1
            continue
        if prefer_reviewed and reviewed_prots is not None:
            # if any of the mapped UniProt IDs are reviewed, keep only those
            us_to_keep = set(new_u for new_u in new_us if new_u in reviewed_prots)
            vs_to_keep = set(new_v for new_v in new_vs if new_v in reviewed_prots)
            new_us = us_to_keep if len(us_to_keep) > 0 else new_us
            new_vs = vs_to_keep if len(vs_to_keep) > 0 else new_vs
        for new_u in new_us:
            for new_v in new_vs:
                new_edges.append([new_u, new_v])
                new_extra_cols.append(extra_cols[i])
    nodes = set(n for u,v in edges for n in (u,v))
    new_nodes = set(n for u,v in new_edges for n in (u,v))
    print("\t%d nodes map to %d new nodes" % (len(nodes), len(new_nodes)))
    print("\t%d edges map to %d new edges" % (len(edges), len(new_edges)))
    print("\t%d unmapped edges" % (num_unmapped_edges))
    mapping_stats = {'nodes': len(nodes), 'new_nodes': len(new_nodes),
                     'edges': len(edges), 'new_edges': len(new_edges),
                     'unmapped_edges': num_unmapped_edges}
    return new_edges, new_extra_cols, mapping_stats
