import pandas as pd
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


def get_list_of_positive_proteins(positive_protein_file):
    # returns: the list of Krogan proteins that have been used as positive examples in prediction
    positive_proteins = pd.read_csv(positive_protein_file, '\t')
    return list(positive_proteins['prots'])


def main():
    predicted_prot_dir = "/media/tassnina/Study/VT/Research_Group/SARSCOV2/SARS-CoV-2-network-analysis/outputs/networks"
    positive_protein_file = "/media/tassnina/Study/VT/Research_Group/SARSCOV2/SARS-CoV-2-network-analysis/fss_inputs/pos-neg/2020-03-sarscov2-human-ppi/pos.txt"

    # this is the Krogan protein list
    positive_protein_list = get_list_of_positive_proteins(positive_protein_file)

    for dirpath, dirs, files in os.walk(predicted_prot_dir):

        for filename in files:

            fname = os.path.join(dirpath, filename)

            if 'pred-scores' in fname:
                predicted_prot_info = pd.read_csv(fname, sep='\t')
          	# filtering out the positive (Krogan) proteins from predicted list of proteins
                predicted_protein_info = predicted_prot_info[~predicted_prot_info['prot'].isin(positive_protein_list)]
                predicted_protein_info.to_csv(dirpath +'/'+'filtered_pred_scores.csv','\t', index=False)



if __name__ == "__main__":
    main()
