
from collections import defaultdict
import os
import sys
import numpy as np
import gzip
import wget
import pandas as pd
import subprocess
import glob
import shutil


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
                download_file(mapping['url'], mapping_file)
            except:
                print("Failed to download '%s' using the url '%s'. Skipping" % (mapping_file, mapping['url']))
                continue

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
                    to_uniprot[uniprot].add(alt_id)
                    from_uniprot[alt_id].add(uniprot)
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

    if 'networks' in dataset_settings:
        networks_dir = "%s/networks" % (datasets_dir)
        setup_networks(networks_dir, dataset_settings['networks'], namespace_mappings, **kwargs)
    if 'gene_sets' in dataset_settings:
        print("WARNING: downloading and parsing of 'gene_sets' is not yet implemented. Skipping.")
    return


def setup_networks(networks_dir, network_settings, namespace_mappings, **kwargs):
    """
    Download all specified network files, and map the identifiers of the proteins in them to UniProt, if necessary
    """
    #global namespace_mappings
    for network in network_settings:
        network_file = "%s/%s/%s" % (
            networks_dir, network['name'], network['file_name'])

        if not kwargs['force_download'] and os.path.isfile(network_file):
            print("%s already exists. Use --force-download to overwrite and re-map to uniprot" % (network_file))
            continue
        elif os.path.isfile(network_file) and kwargs['force_download']:
            print("--force-download not yet setup to overwrite the networks. Please manually remove the file(s) and re-run")
            continue
        # download the file 
        #try:
        download_file(network['url'], network_file)
        #except:
        #    print("Failed to download '%s' using the url '%s'. Skipping" % (network_file, network['url']))
        #    continue

        unpack_command = network.get('unpack_command')
        # if its a zip file, unzip first
        if unpack_command is not None and unpack_command != "":
            command = "%s %s" % (unpack_command, network_file)
            run_command(command, chdir=os.path.dirname(network_file))

        mapping_settings = network.get('mapping_settings', {})
        opts = network.get('collection_settings', {})
        if network['network_collection'] is True:
            setup_network_collection(
                network_file, namespace_mappings,
                namespace=network.get('namespace'), gzip_files=opts.get('gzip_files'),
                prefer_reviewed=mapping_settings.get('prefer_reviewed'),
                remove_filename_spaces=opts.get('remove_filename_spaces'),
                columns_to_keep=network.get('columns_to_keep'),
                sep=network.get('sep')
            ) 
        else:
            file_end = network_file[-4]
            new_f = network_file.replace(file_end, "-parsed"+file_end)
            mapping_stats = setup_network(
                network_file, new_f, namespace_mappings,
                namespace=network.get('namespace'), 
                prefer_reviewed=mapping_settings.get('prefer_reviewed'),
                columns_to_keep=network.get('columns_to_keep'), sep=network.get('sep')
            )

            if opts.get('gzip_files'):
                gzip_file(new_f, new_f+'.gz', remove_orig=True)
                new_f += ".gz"

            if mapping_stats is not None:
                all_mapping_stats = {os.path.basename(new_f).split('.')[0]: mapping_stats}
                df = pd.DataFrame(all_mapping_stats).T
                print(df)
                stats_file = "%s/mapping-stats.tsv" % (os.path.dirname(new_f))
                print("Writing network mapping statistics to %s" % (stats_file))
                df.to_csv(stats_file, sep='\t')
    return 


def setup_network_collection(
        collection_file, namespace_mappings, namespace=None,
        gzip_files=False, prefer_reviewed=True,
        remove_filename_spaces=False,
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
            orig_f, new_f, namespace_mappings, namespace=namespace,
            prefer_reviewed=prefer_reviewed, columns_to_keep=columns_to_keep, sep=sep)

        # gzip the original file to save on space
        gzip_file(orig_f, orig_f+'.gz', remove_orig=True)
        # also gzip the new file if specified
        if gzip_files:
            gzip_file(new_f, new_f+'.gz', remove_orig=True)
            new_f += ".gz"

        f_name = os.path.basename(new_f)
        net_files.append(f_name)
        if mapping_stats is not None:
            all_mapping_stats[f_name.split('.')[0]] = mapping_stats 

    with open('%s/net-files.txt' % (collection_dir), 'w') as out:
        out.write('\n'.join(net_files)+'\n')
    if len(all_mapping_stats) > 0:
        df = pd.DataFrame(all_mapping_stats).T
        print(df)
        stats_file = "%s/mapping-stats.tsv" % (collection_dir)
        print("Writing network mapping statistics to %s" % (stats_file))
        df.to_csv(stats_file, sep='\t')
    print("")
    return df


def setup_network(
        network_file, new_file, namespace_mappings, namespace=None,
        weighted=False, prefer_reviewed=True, columns_to_keep=None, sep='\t'):
    """
    *network_file*: path/to/original network file in edge-list format. First two columns should be the tail and head of the edge
    *new_file*: path/to/new file to write. 
    *namespace_mappings*: a dictionary of IDs each mapped to a set of uniprot IDs
    *namespace*: the namespace of the nodes in the networks
    *weighted*: T/F. If False, add a column of all 1s (i.e., unweighted) after the first two columns. 
        If True, the first column in *columns_to_keep* should be the weights
    *prefer_reviewed*: when mapping, if any of the alternate IDs map to a reviewed UniProt ID, keeping only that one. Otherwise keep all UniProt IDs
    *columns_to_keep*: a list of indexes of columns to keep in the new file. Should be >= 2 (first two columns should be the tail and head of the edge) 
    *sep*: the delimiter of columns in the files

    *returns*: a dictionary of mapping statistics
    """
    # read in the file
    # the edges should be the first two columns
    edges = []
    # store the specified extra columns
    extra_cols = []
    print("Reading %s" % (network_file))
    with open(network_file, 'r') as f:
        for line in f:
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
    if namespace is not None and namespace.lower() not in ['uniprot', 'uniprotkb']:
        new_edges, new_extra_cols, mapping_stats = map_network(edges, extra_cols, namespace_mappings, prefer_reviewed=prefer_reviewed)
    else:
        new_edges, new_extra_cols = edges, extra_cols
        mapping_stats = None

    print("\twriting %s" % (new_file))
    # and re-write the file with the specified columns to keep
    with open(new_file, 'w') as out:
        for i, e in enumerate(new_edges):
            new_line = "%s%s%s\n" % (
                '\t'.join(e), "\t1\t" if weighted is False else "",
                # if weighted is True, then the first column in new_extra_cols should be the weight
                '\t'.join(new_extra_cols[i]))
            out.write(new_line)
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


def setup_genesets():
    pass


def run_command(command, chdir=None):
    """
    Run the given command using subprocess. 
    *chdir*: Change to the specified directory before running the command, 
        then change back to the original directory
    """
    if chdir is not None:
        curr_dir = os.getcwd()
        os.chdir(chdir)
    print("Running: %s" % (command))
    subprocess.check_output(command, shell=True)

    if chdir is not None:
        os.chdir(curr_dir)


def gzip_file(f1, f2, remove_orig=True):
    print("\tgzipping %s" % (f1))
    with open(f1, 'rb') as f_in:
        with gzip.open(f2, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    # and remove the original
    if remove_orig:
        os.remove(f1)


def download_file(url, file_path):
    # make sure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print("Downloading to file '%s' from '%s'" % (file_path, url))
    wget.download(url, file_path)
    # TODO also add the date of download and url to a README file
    print("")
