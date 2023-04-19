import numpy as np
import yaml
import gzip
import os
import pandas as pd
network_alias = {'stringv11-5':'s', 'biogrid-y2h-sept22':'y', 'biogrid-physical-sept22':'p','HI-union':'h'   }

def extract_prots_indv_net(net_file):
    prots = []
    open_func = gzip.open if '.gz' in net_file else open
    with open_func(net_file, 'r') as f:
        for line in f:
            line = line.decode() if '.gz' in net_file else line
            if line[0] == "#":
                continue
            line = line.rstrip().split('\t')
            u, v = line[:2]
            #some biogrid prots' uniprot ids are in form 'AAAT|POQJS', take the first uniprot id only.
            u=u.split('|')[0]
            #Nure: 04/07/2023.fixed a bug where the following line was u=v.split('|')[0]. So run code again.
            v = v.split('|')[0]
            prots+=[u,v]
    return prots

def find_overlap_prots(nets,input_dir):
    prots_per_network = {}
    for net in nets:
        net_name = net['name']
        print(net_name)
        for root, dirs, files in os.walk(input_dir + net['dir']):
            for f in files:
                if 'node-ids.txt' in f:
                    prot_file = root+ '/'+f

        #TODO check if prot_file gives the full path
        if not os.path.isfile(prot_file):
            net_file = input_dir + net['dir'] + net['file']
            prots_per_network[net_name] = set(extract_prots_indv_net(net_file))
        else:
            prots = list(pd.read_csv(prot_file,sep = '\t', header=None,index_col=False, names = ['prot','idx'])['prot'])
            #some biogrid prots' uniprot ids are in form 'AAAT|POQJS', take the first uniprot id only.
            prots = [x.split('|')[0] for x in prots]
            prots_per_network[net_name] = set(prots)

    overlap_prots = set()
    for net_name in prots_per_network:
        if len(overlap_prots)==0:
            overlap_prots = prots_per_network[net_name]
        else:
            overlap_prots = overlap_prots.intersection(prots_per_network[net_name])

    return list(overlap_prots)

def save_netfile_with_overlap_prots(overlap_prots, net_file, new_net_file, force_run=False):
    if (not os.path.isfile(new_net_file))| (force_run):
        open_func = gzip.open if '.gz' in net_file else open

        os.makedirs(os.path.dirname(new_net_file), exist_ok=True)
        f_out = open_func(new_net_file, 'wb' if '.gz' in net_file else 'w')

        with open_func(net_file, 'r') as f_in:
            for line in f_in:
                #TODO check if every line ends with '\n'
                line = line.decode() if '.gz' in net_file else line
                if line[0] == "#":
                    f_out.write(line.encode() if '.gz' in net_file else line)
                    # TODO  if every line does not end with '\n', then make sure we add '\n'
                    continue
                line_strip = line.rstrip().split('\t')
                u, v = line_strip[:2]
                #some biogrid prots' uniprot ids are in form 'AAAT|POQJS', take the first uniprot id only.
                u = u.split('|')[0]
                v = v.split('|')[0]

                if (u in overlap_prots)&(v in overlap_prots):
                    f_out.write(line.encode() if '.gz' in net_file else line)
        f_out.close()

def main(config_file):
    '''
    This code will create new network files for the networks mentioned in
    config file. The new files are for networks containing  only the overlapping proteins/nodes among
    the networks mentioned in config file.
    '''
    with open(config_file, 'r') as conf:
        config_map = yaml.load(conf)
    input_dir = config_map['input_settings']['input_dir']
    nets = config_map['input_settings']['nets']

    #find out the overlapping proteins
    overlap_prots = find_overlap_prots(nets, input_dir)

    #create tag for overlap network files such that tag contains among which networks we are finding overlap
    overlap_tag = ''
    for net in nets:
        overlap_tag += network_alias[net['name']]

    #now for each network, save/write a new network file only with lines containing overlapping prots
    for net in nets:
        net_file = input_dir + net['dir'] + net['file']
        new_net_file = input_dir + net['dir'].replace(net['name'], net['name']+overlap_tag) + net['file']
        save_netfile_with_overlap_prots(overlap_prots, net_file, new_net_file, force_run=True)
        print('Done:' , net['name'])


main('/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/fss_inputs/config_files/overlap_network_config.yaml')