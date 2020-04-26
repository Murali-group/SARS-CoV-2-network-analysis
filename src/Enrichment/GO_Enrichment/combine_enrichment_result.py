import pandas as pd
import numpy as np
import argparse
import yaml
import sys
import os
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
	parser = argparse.ArgumentParser(description="Script for combining go enrichment results")
	# general parameters
	group = parser.add_argument_group('Main Options')
	group.add_argument('--config', type=str, default="config.yaml",
		       help="Configuration file for this script.")

	return parser


def combine_enrichment_result(predicted_prot_dir):

	df_CC_dict = {}
	df_BP_dict = {}
	df_MF_dict = {}

	for dirpath, dirs, files in os.walk(predicted_prot_dir):
		
		
			
		for filename in files:
			fname = os.path.join(dirpath, filename)
			if 'enrichGO' in fname:

				dirpath_split = dirpath.split('/')
				network_name = dirpath_split[-4]+'_'+dirpath_split[-3] + '_'+dirpath_split[-2]
				algorithm = dirpath_split[-1]

				#k = filename.split('_')[-1].replace('.csv','')
				k=200

				protein_source = algorithm +'-'+str(k)+'-'+network_name

				if algorithm not in df_CC_dict:
					df_CC_dict[algorithm] = pd.DataFrame()
					df_BP_dict[algorithm] = pd.DataFrame()
					df_MF_dict[algorithm] = pd.DataFrame()
					print(algorithm)
									
					
				if 'enrichGO_CC' in fname:
					
					df_ = pd.read_csv(fname,names= [protein_source+"_GO_ID", 
						protein_source+"_Description", protein_source+"_GeneRatio",protein_source+"_BgRatio",
						protein_source+"_pvalue", protein_source+"_p.adjust",protein_source+"_qvalue", 
						protein_source+"_geneID",protein_source+"_count"  ], sep =',',  header=0)

					if not df_.empty:
						df_CC_dict[algorithm] = pd.concat([df_CC_dict[algorithm], df_],axis=1, sort=False)

				if 'enrichGO_MF' in fname:

					df_ = pd.read_csv(fname,names= [protein_source+"_GO_ID", 
						protein_source+"_Description", protein_source+"_GeneRatio",protein_source+"_BgRatio",
						protein_source+"_pvalue", protein_source+"_p.adjust",protein_source+"_qvalue", 
						protein_source+"_geneID",protein_source+"_count"  ], sep =',',  header=0)

					if not df_.empty:
						df_MF_dict[algorithm] = pd.concat([df_MF_dict[algorithm], df_],axis=1, sort=False)

				if 'enrichGO_BP' in fname:

					df_ = pd.read_csv(fname,names= [protein_source+"_GO_ID", 
						protein_source+"_Description", protein_source+"_GeneRatio",protein_source+"_BgRatio",
						protein_source+"_pvalue", protein_source+"_p.adjust",protein_source+"_qvalue", 
						protein_source+"_geneID",protein_source+"_count"  ], sep =',',  header=0)

					if not df_.empty:
						df_BP_dict[algorithm] = pd.concat([df_BP_dict[algorithm], df_],axis=1, sort=False)
	if not os.path.exists(predicted_prot_dir + '/Combined_Output/'):
		os.mkdir(predicted_prot_dir + '/Combined_Output/')
	
	for algorithms in df_CC_dict:
		df_CC_dict[algorithms].to_csv(predicted_prot_dir + '/Combined_Output/'+ algorithms + '_combined_GO_CC_Enrichment.csv','\t', index= False)
		df_MF_dict[algorithms].to_csv(predicted_prot_dir + '/Combined_Output/'+ algorithms +'_combined_GO_MF_Enrichment.csv','\t', index= False)
		df_BP_dict[algorithms].to_csv(predicted_prot_dir + '/Combined_Output/'+ algorithms +'_combined_GO_BP_Enrichment.csv','\t', index= False)


def main(config_map, **kwargs):

	#combine_enrichment_result(predicted_prot_dir)
	combine_enrichment_result( "/media/tassnina/Study/VT/Research_Group/SARSCOV2/SARS-CoV-2-network-analysis/outputs/networks")
if __name__ == "__main__":
	config_map, kwargs = parse_args()
	main(config_map, **kwargs)
