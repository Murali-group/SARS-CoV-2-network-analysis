#!/usr/bin/python

print("Importing libraries")

from optparse import OptionParser
import os
import sys
import gzip
#import subprocess
#import networkx as nx
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import pdb


# the names of the columns in the full links file
full_column_names = OrderedDict()
full_column_names["protein1"]                = 1
full_column_names["protein2"]                = 2
full_column_names["neighborhood"]            = 3
full_column_names["neighborhood_transferred"]= 4
full_column_names["fusion"]                  = 5
full_column_names["cooccurence"]             = 6
full_column_names["homology"]                = 7
full_column_names["coexpression"]            = 8
full_column_names["coexpression_transferred"]= 9
full_column_names["experiments"]             = 10
full_column_names["experiments_transferred"] = 11
full_column_names["database"]                = 12
full_column_names["database_transferred"]    = 13
full_column_names["textmining"]              = 14
full_column_names["textmining_transferred"]  = 15
full_column_names["combined_score"]          = 16

# should be out_dir/taxon/taxon.links.full.version-cutoff.txt
STRING_TAXON_FILE = "%%s/%%s/%%s-%%d.txt" 
# TODO get the version automatically
STRING_TAXON_UNIPROT_FULL = "%%s/%%s/%%s-uniprot-full-links-v11-%%d.txt"


def parse_args(args):
    ## Parse command line args.
    usage = '%s [options]\n' % (args[0])
    parser = OptionParser(usage=usage)
    parser.add_option('-i', '--string-file', type='string',
            help="File containing interactions of all species downloaded from STRING. Will be split into each individual species. Required.")
    parser.add_option('-o', '--out-dir', type='string', 
            help="Directory to write the interactions of each species to. Required.")
    parser.add_option('', '--uniprot-to-string', type='string',
            help="Tab-delimited file from UniProt containing the UniProt ID in the first column, and the STRING ID in column 5. Required." )
    #parser.add_option('', '--string-to-uniprot', type='string',
    #        help="Tab-delimited file from STRING containing the UniProt ID in the first column, and the STRING ID in column 5. Required." )
    parser.add_option('-S', '--score-cutoff', type='int', default=400,
            help="Cutoff for the STRING interaction score. Scores range from 150-1000. Default is 400 (medium). Useful to save on file space")
    parser.add_option('-s', '--selected-species', type='string', 
            help="Species for which to write the networks and print mapping statistics. Not required.")

    (opts, args) = parser.parse_args(args)

    if opts.string_file is None or opts.out_dir is None or opts.uniprot_to_string is None:
        sys.exit("--string-file, --out-dir, and --map-to-uniprot  required")

    return opts, args


def main(args):
    opts, args = parse_args(args)

    if opts.uniprot_to_string is not None:
        string_to_uniprot, uniprot_to_taxon, taxons = parse_uniprot_to_string_mapping(opts.uniprot_to_string)

    #if opts.selected_species is not None:
    #    selected_species = utils.readItemSet(opts.selected_species, 1)
    #else:
    # TODO there could be species without a mapping. Should I still write those?
    # for now, I'm only writing the ones with uniprot mappings
    selected_species = taxons

    base_file_name = os.path.basename(opts.string_file).replace('protein.','').replace('.txt.gz','')
    global STRING_TAXON_FILE, STRING_TAXON_UNIPROT_FULL
    # should be <out_dir>/<taxon>/<taxon>.links.full.version-cutoff.txt
    STRING_TAXON_FILE = "%s/%%s/%%s.%s-%%d.txt.gz" % (opts.out_dir, base_file_name)
    # TODO get the version automatically
    STRING_TAXON_UNIPROT_FULL = "%s/%%s/%%s-uniprot-full-links-v11-%%d.txt.gz" % (opts.out_dir)

    species_to_split = []
    for species in selected_species:
        if not os.path.isfile(STRING_TAXON_UNIPROT_FULL % (species, species, opts.score_cutoff)):
            species_to_split.append(species)

    # This can take a few hours, so only split the original file if needed
    if len(species_to_split) > 0:
        if len(species_to_split) > 200: 
            print("Splitting %d species from STRING file from %s to %s" % (len(species_to_split), opts.string_file, opts.out_dir))
        else:
            print("Splitting %d species from STRING file from %s to %s: %s" % (len(species_to_split), opts.string_file, opts.out_dir, ', '.join(species_to_split)))
        #split_string_to_species(opts.string_file, opts.out_dir, species_to_split=species_to_split, score_cutoff=opts.score_cutoff)
        # UPDATE 2019-05-30 use the UniProt mappings directly, and write only the STRING networks with UniProt IDs
        stats = split_string_to_species_uniprot(
                opts.string_file, opts.out_dir, string_to_uniprot, uniprot_to_taxon,
                score_cutoff=opts.score_cutoff)
        num_string_itx_per_taxon, num_uniprot_itx_per_taxon, num_string_per_taxon, num_uniprot_per_taxon = stats
        # write the statistics to a file
        df = pd.DataFrame(list(stats)).T
        df.columns = ['num_string_itx_per_taxon', 'num_uniprot_itx_per_taxon', 'num_string_per_taxon', 'num_uniprot_per_taxon']
        stats_file = "%s/mapping-stats.tsv" % (opts.out_dir)
        print("writing mapping stats to %s" % (stats_file))
        df.to_csv(stats_file, sep='\t')
    else:
        print("%d species have already been split from the main STRING file %s to %s." % (len(species_to_split), opts.string_file, opts.out_dir))

    print("Finished.")


def parse_uniprot_to_string_mapping(uniprot_mapping_file):
    print("parsing uniprot to string mapping file %s" % (uniprot_mapping_file))

    # example UniProt mapping line:
    #  Entry   Status  Gene names  Organism ID Cross-reference (STRING)
    #  Q11094  reviewed    hlh-8 C02B8.4   6239    6239.C02B8.4;
    df = pd.read_csv(uniprot_mapping_file, sep='\t')

    # take off the ';' at the end of the ID(??)
    df.loc[:, 'Cross-reference (STRING)'] = df['Cross-reference (STRING)'].apply(lambda x: x.strip(';'))
    # make sure the taxonomy ID is read as a string
    df['Organism ID'] = df['Organism ID'].apply(str) 
    print(df.head())
    
    # return the mapping as a dictionary and also return the taxonomy IDs
    taxons = df['Organism ID'].unique()
    string_to_uniprot = dict(zip(df['Cross-reference (STRING)'], df['Entry']))
    uniprot_to_taxon = dict(zip(df['Entry'], df['Organism ID']))
    return string_to_uniprot, uniprot_to_taxon, taxons


# TODO
def parse_string_to_uniprot_mapping(string_mapping_file):
    # example STRING mapping line:
    #  #species   uniprot_ac|uniprot_id   string_id   identity   bit_score
    #  742765    G1WQX1|G1WQX1_9FIRM    742765.HMPREF9457_01522    100.0    211.0
    pass


def split_string_to_species_uniprot(string_file, out_dir, string_to_uniprot, uniprot_to_taxon, species_to_split=None, score_cutoff=150):
    """
    Split all of the STRING networks to individual files using the UniProt mappings.
    Use the species listed by UniProt instead of the species listed by STRING.
    *score_cutoff*: Only include lines with a combined score >= *score_cutoff*. 150 is the lowest for STRING
    """
    # keep track of some statistics
    num_string_itx_per_taxon = defaultdict(int)
    num_uniprot_itx_per_taxon = defaultdict(int)
    string_per_taxon = defaultdict(set)
    uniprot_per_taxon = defaultdict(set)
    num_uniprot_itx_mismatch_taxon = 0

    #if score_cutoff > 150:
    print("\tonly including interactions with a combied score >= %d" % (score_cutoff))
    # estimate the number of lines
    # takes too long (~20 minutes)
    #num_lines = rawgencount(string_file)
    # this was from v10.5
    #num_lines = 2800000000
    num_lines = 28000000000
    last_taxon = "" 

    # files are very large, so use the gzip library 
    with gzip.open(string_file, 'rb') as f:
        header = f.readline().decode('UTF-8')
        last_species = ""
        # now split the file by species
        # TODO the file appears to be organized by species, so only have to open one output file for writing at a time
        # otherwise this would have to either sort the file, open multiple files for writing, or append to all files
        for line in tqdm(f, total=num_lines):
            # needed for Python3
            line = line.decode('UTF-8')
            line = line.rstrip().split(' ')
            combined_score = int(line[-1])
            # only write the interactions with a score >= the score cutoff
            if score_cutoff is not None and combined_score < score_cutoff:
                continue

            # get the species from uniprot now
            string_curr_species = line[0][:line[0].index('.')]
            #if species_to_split is not None and curr_species not in species_to_split:
            #    continue

            a = line[0]
            b = line[1]

            # keep stats of all STRING interactions
            num_string_itx_per_taxon[string_curr_species] += 1 
            string_per_taxon[string_curr_species].add(a)
            string_per_taxon[string_curr_species].add(b)

            u_a = string_to_uniprot.get(a,None)
            u_b = string_to_uniprot.get(b,None)
            if u_a is None or u_b is None:
                continue
            u_a_taxon = uniprot_to_taxon[u_a]
            u_b_taxon = uniprot_to_taxon[u_b]
            # sometimes the STRING taxon doesn't match the UniProt taxon, 
            # so these IDs could be from different species
            if u_a_taxon != u_b_taxon:
                num_uniprot_itx_mismatch_taxon += 1 
                continue
            curr_taxon = u_a_taxon
            if species_to_split is not None and curr_taxon not in species_to_split:
                continue

            num_uniprot_itx_per_taxon[curr_taxon] += 1 
            uniprot_per_taxon[curr_taxon].add(u_a)
            uniprot_per_taxon[curr_taxon].add(u_b)

            if curr_taxon != last_taxon:
                curr_out_dir = "%s/%s" % (out_dir, curr_taxon)
                # analagous to mkdir -p directory from the command line
                if not os.path.isdir(curr_out_dir):
                    os.makedirs(curr_out_dir)
                out_file = STRING_TAXON_UNIPROT_FULL % (curr_taxon, curr_taxon, score_cutoff)
                tqdm.write("Writing new taxon interactions to '%s'" % (out_file))
                if last_taxon != "":
                    # close the last file we had open
                    out.close()
                write_header = False 
                if not os.path.isfile(out_file):
                    write_header = True
                # open the new file for writing
                # write it as a binary gzip file
                out = gzip.open(out_file, 'ab')
                #out = open(out_file, 'w')
                last_taxon = curr_taxon
                if write_header:
                    out.write(str("#" + header.replace(' ','\t')).encode())

            out_line = "%s\t%s\t%s\n" % (u_a, u_b, '\t'.join(line[2:]))
            out.write(out_line.encode())
            #interactors = ' '.join(line.split(' ')
            #out2.write("%s %s %d" % (line.split,,score)

    out.close()
    print("Finished splitting species string interactions")
    print("\t%d interactions had mismatching UniProt taxon IDs" % (num_uniprot_itx_mismatch_taxon))
    num_string_per_taxon = {t: len(ids) for t, ids in string_per_taxon.items()}
    num_uniprot_per_taxon = {t: len(ids) for t, ids in uniprot_per_taxon.items()}
    return num_string_itx_per_taxon, num_uniprot_itx_per_taxon, num_string_per_taxon, num_uniprot_per_taxon


def split_string_to_species(string_file, out_dir, species_to_split=None, score_cutoff=150):
    """
    Split all of the STRING networks to individual files using only the STRING IDs
    *score_cutoff*: Only include lines with a combined score >= *score_cutoff*. 150 is the lowest for STRING
    """
    #if score_cutoff > 150:
    print("\tonly including interactions with a combied score >= %d" % (score_cutoff))
    # estimate the number of lines
    # takes too long (~20 minutes)
    #num_lines = rawgencount(string_file)
    # this was from v10.5
    #num_lines = 2800000000
    num_lines = 28000000000

    # files are very large, so use the gzip library 
    with gzip.open(string_file, 'rb') as f:
        header = f.readline()
        last_species = ""
        # now split the file by species
        # TODO the file appears to be organized by species, so only have to open one output file for writing at a time
        # otherwise this would have to either sort the file, open multiple files for writing, or append to all files
        for line in tqdm(f, total=num_lines):
            # needed for Python3
            line = line.decode('UTF-8')
            score = int(line.rstrip().split(' ')[-1])
            # only write the interactions with a score >= the score cutoff
            if score_cutoff is not None and score < score_cutoff:
                continue

            # get the species from uniprot now
            curr_species = line[:line.index('.')]
            if species_to_split is not None and curr_species not in species_to_split:
                continue

            if curr_species != last_species:
                out_dir = "%s/%s" % (out_dir, curr_species)
                # analagous to mkdir -p directory from the command line
                if not os.path.isdir(out_dir):
                    os.makedirs(out_dir)
                out_file = STRING_TAXON_FILE % (curr_species, curr_species, score_cutoff)
                tqdm.write("Writing new species interactions to '%s'" % (out_file))
                if last_species != "":
                    # close the last file we had open
                    out.close()
                # open the new file for writing
                out = open(out_file, 'w')
                #out2 = open(out_file2, 'w')
                last_species = curr_species

            out.write(line)
            #interactors = ' '.join(line.split(' ')
            #out2.write("%s %s %d" % (line.split,,score)

    out.close()
    print("Finished splitting species string interactions")
    return


if __name__ == '__main__':
    main(sys.argv)
