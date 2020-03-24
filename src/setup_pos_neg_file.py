
import argparse
from collections import defaultdict
import os
import sys
#from tqdm import tqdm
#import pandas as pd
import numpy as np


def parse_args():
    ## Parse command line args.
    parser = argparse.ArgumentParser(
        description='Script to take known human-virus PPIs, (TODO) generate negative examples, ' +
        'and create a "pos-neg-file" which matches the format expected by the FastSinkSource pipeline. ' +
        'Writes a tab-separated table with proteins on the rows, the name of the input file on the column, ' +
        'and 1/0/-1 for pos/unk/neg example as the values.')

    # general parameters
    parser.add_argument('--pos-examples-file', type=str, required=True,
            help="Single-column file containing the positive examples")
    parser.add_argument('--name', type=str, 
            help="Name to give this dataset. Default is the file name")
    parser.add_argument('--prot-universe-file', type=str, 
            help="Single-column file containing universe of proteins from which to sample negative examples")
    parser.add_argument('--sample-neg-examples-factor', type=float, 
            help="If specified, sample negative examples randomly without replacement from the protein universe equal to <sample_neg_examples_factor> * # positives")
    parser.add_argument('--seed', type=float, 
            help="TODO Seed of the random number generator to use when sampling.")
    parser.add_argument('--out-file', type=str, default="pos-neg-file.tsv",
            help="path/to/file.tsv for which to write output.")

    # evaluation parameters
    #group = parser.add_argument_group('Evaluation options')
    #group.add_argument('--only-eval', action="store_true", default=False,
    #        help="Perform evaluation only (i.e., skip prediction mode)")
    args = parser.parse_args()
    return args


def main(pos_examples_file, prot_universe_file=None, sample_neg_examples_factor=None,
         out_file=None, **kwargs):
    """
    """
    print("Reading %s" % (pos_examples_file))
    pos_examples = set(np.loadtxt(pos_examples_file, dtype=str))
    print("\t%s positive examples" % (len(pos_examples)))
    neg_examples = None

    if prot_universe_file is not None:
        print("Reading %s" % (prot_universe_file))
        prot_universe = set(np.loadtxt(prot_universe_file, dtype=str))
        print("\t%s proteins" % (len(prot_universe)))

        pos_examples = pos_examples & prot_universe
        print("\t%s positive examples after limitting to those in the specified universe" % (len(pos_examples)))

    if sample_neg_examples_factor is not None:
        if prot_universe_file is None:
            print("ERROR: Must specify the universe from which to sample negative examples if --sample-neg-examples specified.")
            sys.exit()
        neg_sample = sample_neg_examples_factor * len(pos_examples) 
        non_pos_universe = prot_universe - pos_examples
        print("Sampling %s (%s*%s) negative examples from the universe of %s non-pos prots" % (
            neg_sample, sample_neg_examples_factor, len(pos_examples), len(non_pos_universe)))
        if neg_sample > len(non_pos_universe):
            print("ERROR: cannot sample more negative examples than specified by non_pos_universe")

        # now perform the sampling
        non_pos_universe = np.asarray(list(non_pos_universe), dtype=str)
        # sample without replacement
        neg_examples = np.random.choice(non_pos_universe, size=int(neg_sample), replace=False)

    # now write the output file
    print("\nWriting %s" % (out_file))
    # make sure the output directory exists
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    name = out_file.split('/')[-1] if kwargs.get('name') is None else kwargs['name']
    out_str = "prots\t%s\n" % (name)
    for p in sorted(prot_universe):
        val = 0
        if p in pos_examples:
            val = 1
        elif neg_examples is not None and p in neg_examples:
            val = -1
        out_str += "%s\t%s\n" % (p, val)

    with open(out_file, 'w') as out:
        out.write(out_str)


if __name__ == "__main__":
    args = parse_args()

    kwargs = vars(args)
    main(**kwargs)
