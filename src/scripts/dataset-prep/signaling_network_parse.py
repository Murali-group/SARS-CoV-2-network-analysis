import os
import pandas as pd

def parse_signor_file(signor_causal_interaction_file, parsed_file):
    ci_df = pd.read_csv(signor_causal_interaction_file, sep = '\t')[['ENTITYA', 'TYPEA', 'IDA',
            'DATABASEA', 'ENTITYB', 'TYPEB', 'IDB', 'DATABASEB', 'EFFECT','TAX_ID']]

    # keep the human protein interactions only i.e. TAX_ID = '9606'
    #keep prorts only i.e. TYPEA==TYPEB=='protein'
    #keep the proteins for which we have uniprot ids i.e. DATABASEA=DATABASEB='UNIPROT'
    # print( 'Non TAX ID: ',len(ci_df[ci_df['TAX_ID']!= 9606]))
    # print( 'Non prot: ',len(ci_df[ci_df['TYPEA']!='protein']), len(ci_df[ci_df['TYPEB']!='protein']))
    # print( 'Non uniprot: ',len(ci_df[ci_df['DATABASEA']!='UNIPROT']),
    #        len(ci_df[ci_df['DATABASEB']!='UNIPROT']))
    #
    # print('initial: ', len(ci_df))
    ci_df = ci_df[(ci_df['TAX_ID']==9606)]
    print('after filtering non human interactions: ', len(ci_df))
    #remove chemical or protein complexes. Keep only proteins.
    ci_df = ci_df[(ci_df['TYPEA']=='protein') & (ci_df['TYPEB']=='protein')]
    print('after filtering non prots: ', len(ci_df))

    # saw that all proteins have UNIPROT ID associated with it. Still for safety filtering
    # for keeping only 'UNIPROT'
    ci_df = ci_df[(ci_df['DATABASEA']=='UNIPROT') & (ci_df['DATABASEB']=='UNIPROT')]
    print('after filtering non UNIPROT: ', len(ci_df))

    #removing self loops
    ci_df = ci_df[ci_df['IDA']!=ci_df['IDB']]
    print('after filtering self loops: ', len(ci_df))


    os.makedirs(os.path.dirname(parsed_file), exist_ok=True)

    #save the edges only in parsed file
    #drop duplicate edges
    only_edges_df = ci_df[['IDA', 'IDB']].drop_duplicates()
    print('after filtering duplicate edges: ', len(only_edges_df))

    only_edges_df.to_csv(parsed_file, sep='\t', index=False, header=None)

    #save all parsed info from the network in another file
    ci_df.to_csv(parsed_file.replace('signor_','signor_all_'), sep='\t', index=False)


def main():
    project_dir = "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
    datasets_signor_dir = project_dir + "datasets/networks/signor/"
    inputs_signor_dir = project_dir + "fss_inputs/networks/signor/"

    downloaded_file = datasets_signor_dir + "/all_data_22_12_22.tsv"
    parsed_file = inputs_signor_dir + "/signor_9606_22_12_22.tsv"

    parse_signor_file(downloaded_file, parsed_file)

main()