import pandas as pd
import argparse
import sys
import os
import yaml
sys.path.insert(0, os.path.dirname(__file__))
def parse_args():
	parser = setup_opts()
	args = parser.parse_args()
	kwargs = vars(args)
	with open(args.config, 'r') as conf:
		config_map = yaml.load(conf, Loader=yaml.FullLoader)
	return config_map, kwargs


def setup_opts():
	# Parse command line args.
	parser = argparse.ArgumentParser(description="Script for filtering Krogan positive examples from enrichment result")
	# general parameters
	group = parser.add_argument_group('Main Options')
	group.add_argument('--config', type=str, default="config.yaml",
		       help="Configuration file for this script.")

	return parser



def get_list_of_positive_proteins(positive_protein_file):
	# returns: the list of Krogan proteins that have been used as positive examples in prediction
	positive_proteins = pd.read_csv(positive_protein_file, '\t')
	return list(positive_proteins['prots'])


def main(config_map, **kwargs):
	predicted_prot_dir = config_map['predicted_prot_dir']
	positive_protein_file = config_map['positive_protein_file']

	# this is the Krogan protein list
	positive_protein_list = get_list_of_positive_proteins(positive_protein_file)

	for dirpath, dirs, files in os.walk(predicted_prot_dir):

		for filename in files:
			
			fname = os.path.join(dirpath, filename)
			
			dirpath_split = dirpath.split('/')
			algorithm = dirpath_split[-1]

			if 'pred-scores' in fname and 'sinksourceplus' not in algorithm:
				#print(fname)
				
				predicted_prot_info = pd.read_csv(fname, sep='\t')
				# filtering out the positive (Krogan) proteins from predicted list of proteins
				predicted_protein_info = predicted_prot_info[~predicted_prot_info['prot'].isin(positive_protein_list)]
				predicted_protein_info.to_csv(dirpath +'/'+'filtered_pred_scores.csv','\t', index=False)


if __name__ == "__main__":
	config_map, kwargs = parse_args()
	main(config_map, **kwargs)
