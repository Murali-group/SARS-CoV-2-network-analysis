from zipfile import ZipFile
from src.utils import parse_utils as utils
import os
import pandas as pd
def parse_biogrid_file(downloaded_file, parsed_file, interaction_type):
    edge_weights_dict = {}
    outfile = open(parsed_file,'w')
    open_command = ZipFile
    invalid_entry = 0
    valid_entry=0
    with open_command(downloaded_file, 'r') as z:
        z.extractall(os.path.dirname(downloaded_file))
        with open(downloaded_file.replace('.zip','.txt'), 'r') as f:
            for line in f:
                # line = line.decode() if '.zip' in downloaded_file else line
                if line[0] == '#':
                    continue
                line = line.rstrip().split('\t')

                # columns_to_keep: ["SWISS-PROT Accessions Interactor A", "SWISS-PROT Accessions Interactor B", \
                #                   "Experimental System", "Experimental System Type","Organism ID Interactor A",
                #                   "Organism ID Interactor B",  "Throughput"]
                u = line[23].split('|')[0] #if there are multiple uniportID then take the first one only
                v = line[26].split('|')[0]  #if there are multiple uniportID then take the first one only
                w = 1
                exp_sys = line[11]
                exp_sys_type = line[12]
                org1 = line[15]
                org2 = line[16]
                throughput = line[17]
                #check if the exp_sys or exp_sys_type is the desired interaction type. otherwise leave the line
                if interaction_type == 'physical':
                    if exp_sys_type !='physical':
                        # print(exp_sys_type)
                        continue
                elif interaction_type == 'y2h':
                    if exp_sys != 'Two-hybrid':
                        continue

                if (org1==org2=='9606'): #keep only human PPI
                    if (u=='-') or (v=='-'):
                        invalid_entry+=1
                        continue
                    else:
                        if (((u, v) not in edge_weights_dict) and ((v, u)
                        not in edge_weights_dict) and (u!=v)): #if the edge is here for the first time
                            edge_weights_dict[(u, v)] = w
                            new_line = u+'\t'+v+'\t'+str(w)+'\t'+exp_sys+'\t'+exp_sys_type+'\t'+org1+'\t'+org2+'\t'+throughput+'\n'
                            valid_entry += 1
                            outfile.write(new_line)
    outfile.close()

def main(interaction_type, test = False, force_run=False):
    # interaction_type can be y2h or physical
    #download biogrid file
    download_filename = "BIOGRID-ALL-4.4.214.tab3.zip"
    url = "https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.4.214/"+download_filename
    
    biogridv ="biogrid-"+interaction_type+"-sept22-single-uniprot/"
    download_dir = "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/datasets/networks/"+biogridv
    parsed_dir = "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/fss_inputs/networks/"+biogridv

    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(parsed_dir,exist_ok=True)

    downloaded_file = download_dir + download_filename
    if interaction_type=='y2h':
        parsed_file = parsed_dir + 'biogrid-9606-two-hybrid.tab'
    elif interaction_type=='physical':
        parsed_file = parsed_dir + 'biogrid-9606-physical.tab'

    if (not os.path.isfile(downloaded_file))|(force_run==True):
        utils.download_file(url, downloaded_file)

    parse_biogrid_file(downloaded_file, parsed_file, interaction_type)

    if test:
        parsed_file_test = parsed_file.replace('.tab','-test.tab')
        parse_biogrid_file(downloaded_file, parsed_file_test, interaction_type)

        df_orig = pd.read_csv(parsed_file, sep='\t', names = ['prot1','prot2', 'weight',\
                'exp_sys', 'exp_sys_type','org1','org2','Throughput'])

        df_orig['prot1'] = df_orig['prot1'].apply(lambda x: x.split('|')[0])
        df_orig['prot2'] = df_orig['prot2'].apply(lambda x: x.split('|')[0])

        df_signlemap = pd.read_csv(parsed_file_test, sep='\t', names = ['prot1','prot2', 'weight',\
                'exp_sys', 'exp_sys_type','org1','org2','Throughput'])

        if len(df_orig) == len(df_signlemap):
            print('same length')

        orig_edges = set(zip(df_orig['prot1'],df_orig['prot2']))
        single_map_edges = set(zip(df_signlemap['prot1'], \
                                   df_signlemap['prot2']))

        print(len(orig_edges.difference(single_map_edges)))
        print(len(single_map_edges.difference(orig_edges)))



main(interaction_type='y2h', test=False)