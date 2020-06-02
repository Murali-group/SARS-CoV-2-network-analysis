# standalone script to parse EMMAA, DrugBank, and Krogan files and analyse them for logical paths from a source to a target.

import pandas as pd
import os
import xml.etree.ElementTree as xml
import argparse
import json

class Node:
    def __init__(self, name, id, type):
        self.name = name
        self.id = id
        self.type = type
        self.edges = set()

class Edge:
    def __init__(self, node, weight, description, evidence):
        self.node = node
        self.description = description
        if (True):
          self.weight = 1
#TODO config based condition
        else:
          self.weight = weight
        self.evidence = evidence

    def __hash__(self):
        return hash(self.node.name) + hash(self.weight)

    def __eq__(self, other):
        if isinstance(other, Edge):
            return self.node.name == other.node.name and self.weight == other.weight
        else:
            return False

human_ppi_nodes = {}
virus_human_ppi_nodes = {}
drug_target_nodes = {}

def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    print("Starting with configuration: " + str(vars(args)))
    if args.config:
        with open(args.config, 'r') as conf:
            #config_map = yaml.load(conf, Loader=yaml.FullLoader)
            config_map = yaml.load(conf)
    else:
        config_map = {}

    return config_map, kwargs

def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="Script to compute activating and inhibatory regulatory paths from drug candidates to each viral protein.")

    # general parameters
    group = parser.add_argument_group('Main Options')
    
    group.add_argument('--config', type=str, default="",
                       help="Configuration file used to run TODO")
    
    group.add_argument('--virus-human-ppis', type=str, default="datasets/protein-networks/2020-03-biorxiv-krogan-sars-cov-2-human-ppi.tsv",
                       help="Viral proteins and the human proteins they interact with. PPIs should be the first two columns, as in the Krogan data set obtained from https://www.biorxiv.org/content/biorxiv/early/2020/03/22/2020.03.22.002386/DC5/embed/media-5.xlsx?download=true." +
                       "Default=datasets/protein-networks/2020-03-biorxiv-krogan-sars-cov-2-human-ppi.tsv")
    
    group.add_argument('--drug-targets', type=str, default="datasets/drug-targets/drugbank/drugbank.xml",
                       help="Drugs and the human proteins they interact with, as in the full database XML obtained from DrugBank: https://www.drugbank.ca/releases/5-1-6/downloads/all-full-database" + 
                       "Default=datasets/drug-targets/drugbank/drugbank.xml") 

    group.add_argument('--human-regulatory-network', type=str, default="datasets/regulatory-networks/2020-04-02-bachman-emmaa-covid19.json",
                       help="Regulatory network of human protein-protein interactions, as in the network obtained from EMMAA: https://emmaa.s3.amazonaws.com/assembled/covid19/statements_2020-05-11-18-12-05.json" +
                       "Default=datasets/regulatory-networks/2020-05-21-bachman-emmaa-covid19.json")
    
    group.add_argument('--output', type=str, default="output",
                       help="Directory in which output should be written" +
                       "Default=output")

    group.add_argument('--pathlinker-output', type=str, default = None,
                       help="Directory containing PathLinker output that should be annotated" +
                       "Default=None")

    return parser

def main(config_map, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """

    # TODO: add command-line or config file option to ignore some edge types.
    
    # read file of sources (e.g., predictions with scores/ranks). TODO: Should link to the overall pipeline to get the filenames automatically.
    
    # read file of targets (e.g., human proteins that interact with SARS-CoV-2 proteins). TODO: this file name should be in the master config file.
    emmaa_activates = ['Activation', 'Complex', 'IncreaseAmount', 'Phosphorylation']
    emmaa_inhibits = ['Inhibition', 'DecreaseAmount', 'Dephosphorylation']
    print("Parsing Human Regulatory Network...")
    with open(kwargs['human_regulatory_network']) as f:
        network = json.load(f)
        
        for interaction in network:
            if interaction['type'] in emmaa_activates:
                weight = float(interaction['belief'])
            elif interaction['type'] in emmaa_inhibits:
                weight = -float(interaction['belief'])
            else:
                #TODO: unknown option
                #continue
                weight = 0
        
            if 'subj' in interaction and 'obj' in interaction:
                subject_name = interaction['subj']['name'].replace("\n", "")
                object_name = interaction['obj']['name'].replace("\n", "")
            elif 'sub' in interaction and 'enz' in interaction:
                subject_name = interaction['sub']['name'].replace("\n", "")
                object_name = interaction['enz']['name'].replace("\n", "")
            elif 'members' in interaction:
                subject_name = interaction['members'][0]['name'].replace("\n", "")
                object_name = interaction['members'][1]['name'].replace("\n", "")
            else:
                continue
           
            if subject_name in human_ppi_nodes:
                subject_node = human_ppi_nodes[subject_name]
            else:
                subject_node = Node(subject_name, subject_name, 'human')
                human_ppi_nodes[subject_name] = subject_node
                
            if object_name in human_ppi_nodes:
                object_node = human_ppi_nodes[object_name]
            else:
                object_node = Node(object_name, subject_name, 'human')
                human_ppi_nodes[object_name] = object_node

            evidence = []
            for e in interaction['evidence']:
                if 'pmid' in e and 'text' in e:
                    evidence.append({'source': 'EMMAA', 'pmid': e['pmid'], 'text': e['text']})

            edge = Edge(object_node, weight, interaction['type'], evidence)
            subject_node.edges.add(edge)

    # read file of sources (e.g., predictions with scores/ranks)
    
    drugbank_activates = ['activator', 'adduct', 'agonist', 'carrier', 'chaperone', 'catalytic activity',
        'cofactor', 'cross-linking/alkylation', 'inducer', 'intercalation', 'partial agonist',
        'positive allosteric modulator', 'positive modulator', 'potentiator', 'protector',
        'stabilization', 'stimulator', 'transporter']
    drugbank_inhibits = [
        'aggregation inhibitor', 'antagonist', 'antisense oligonucleotide', 'blocker', 'chelator'
        'cleavage', 'conversion inhibitor', 'degradation', 'desensitize the target',
        'disruptor', 'downregulator', 'inactivator', 'incorporation into and destabilization',
        'inhibition of synthesis', 'inhibitor', 'inhibitory allosteric modulator',
        'inhibits downstream inflammation cascades', 'inverse agonist', 'metabolizer',
        'negative modulator', 'neutralizer', 'nucleotide exchange blocker',
        'partial antagonist', 'reducer', 'suppressor', 'translocation inhibitor', 'weak inhibitor']
    
    print("Parsing Drug Targets...")
    with open(kwargs['drug_targets']) as f:
        drugs = xml.parse(f)

        for drug in drugs.getroot():

            drug_name = drug.find('{http://www.drugbank.ca}name').text
            drug_id = drug.find('{http://www.drugbank.ca}drugbank-id').text

            if drug_name in drug_target_nodes:
                drug_node = drug_target_nodes[drug_name]
            else:
                drug_node = Node(drug_name, drug_id, 'drug')
                drug_target_nodes[drug_name] = drug_node

            for target in drug.find('{http://www.drugbank.ca}targets'):
                weight = 0
                interaction = "unknown"
                for action in target.find('{http://www.drugbank.ca}actions'):
                    if action.text in drugbank_activates:
                        weight = 1
                        interaction = action.text
                    elif action.text in drugbank_inhibits:
                        weight = -1
                        interaction = action.text

                evidence = []
                references = target.find('{http://www.drugbank.ca}references')
                for article in references.find('{http://www.drugbank.ca}articles'):
                    pmid = article.find('{http://www.drugbank.ca}pubmed-id').text
                    text = article.find('{http://www.drugbank.ca}citation').text
                    evidence.append({'source': 'DrugBank', 'pmid': pmid, 'text': text})

                protein = target.find('{http://www.drugbank.ca}polypeptide')
                if protein:
                #TODO: unknown option
                #if protein and not interaction == 'unknown':
                    protein_id = protein.find('{http://www.drugbank.ca}gene-name').text
                    if protein_id in drug_target_nodes:
                        protein_node = drug_target_nodes[protein_id]
                    else:
                        protein_node = Node(protein_id, protein_id, 'human')
                        drug_target_nodes[protein_id] = protein_node
                    
                    edge = Edge(protein_node, weight, interaction, evidence)
                    drug_node.edges.add(edge)

    # read file of targets (e.g., human proteins that interact with SARS-CoV-2 proteins)
    # TODO: read this file name from a config file.
    print('Parsing Virus Human Protein Interactions...')
    virus_human_ppi = pd.read_csv(kwargs['virus_human_ppis'], sep='\t')
    virus_human_ppi["#Bait"] = virus_human_ppi["#Bait"]
    for index, row in virus_human_ppi.iterrows():
        virus_protein = row["#Bait"]
        human_protein = row["PreyGene"]

        if human_protein in virus_human_ppi_nodes:
            human_node = virus_human_ppi_nodes[human_protein]
        else:
            human_node = Node(human_protein, human_protein, 'human')
            virus_human_ppi_nodes[human_protein] = human_node

        if virus_protein in virus_human_ppi_nodes:
            virus_node = virus_human_ppi_nodes[virus_protein]
        else:
            virus_node = Node(virus_protein, virus_protein, 'virus')
            virus_human_ppi_nodes[virus_protein] = virus_node

        evidence = [{'source': 'Krogan', 'pmid': '32353859', 'text': 'Gordon, D.E., Jang, G.M., Bouhaddou, M. et al. A SARS-CoV-2 protein interaction map reveals targets for drug repurposing. Nature (2020). https://doi.org/10.1038/s41586-020-2286-9'}]

        edge = Edge(virus_node, 1, 'Human-Virus PPI', evidence)
        human_node.edges.add(edge)

    if 'ACE2' in virus_human_ppi_nodes:
        ace2_node = virus_human_ppi_nodes['ACE2']
    else:
        ace2_node = Node('ACE2', 'ACE2', 'human')
        virus_human_ppi_nodes['ACE2'] = ace2_node

    if 'SARS-CoV2 Spike' in virus_human_ppi_nodes:
        spike_node = virus_human_ppi_nodes['SARS-CoV2 Spike']
    else:
        spike_node = Node('SARS-CoV2 Spike', 'SARS-CoV2 Spike', 'virus')
        virus_human_ppi_nodes['SARS-CoV2 Spike'] = spike_node

    edge = Edge(spike_node, 1, 'Human-Virus PPI', [])
    ace2_node.edges.add(edge)

    if (kwargs['pathlinker_output']) :
        summary = {'file': [], 'activating': [], 'inhibiting': [], 'unknown': [], 'score': []} 
        for input in os.listdir(kwargs['pathlinker_output']):
            if '_k-100-paths' in input:
                shortest_paths = pd.read_csv(kwargs['pathlinker_output'] + '/' + input, sep='\t', quoting=3)
                activating = {}
                inhibiting = {}
                unknowns = {}
                for index, row in shortest_paths.iterrows():
                    path = row['path'].split('|')
                    sign = 1
                    annotated = None
                    evidence = []
                    unknown = False
                    drug = drug_target_nodes[path[0].split('"')[1]]
                    
                    for edge in drug.edges:
                        if edge.node.name == path[1].split('"')[1]:
                            annotated = path[0] + "--(" + edge.description + ")-->" + path[1] 
                            if not (edge.description in drugbank_inhibits or edge.description in drugbank_activates):
                                unknown = True
                            if edge.description in drugbank_inhibits:
                                sign = -sign
                            evidence.append(json.dumps(edge.evidence))
                    
                    if not annotated:
                        protein = human_ppi_nodes[path[0].split('"')[1]]
                        for edge in protein.edges:
                            if edge.node.name == path[1].split('"')[1]:
                                annotated = path[0] + "--(" + edge.description + ")-->" + path[1]
                                if not (edge.description in emmaa_inhibits or edge.description in emmaa_activates):
                                    unknown = True
                                if edge.description in emmaa_inhibits:
                                    sign = -sign
                                evidence.append(json.dumps(edge.evidence))
 
                    for i in range(1, len(path) - 2):
                        protein = human_ppi_nodes[path[i].split('"')[1]]
                        protein_target = False
                        for edge in protein.edges:
                            if edge.node.name == path[i + 1].split('"')[1]:
                                protein_target = True
                                annotated += "--(" + edge.description + ")-->" + path[i + 1]
                                if not (edge.description in emmaa_inhibits or edge.description in emmaa_activates):
                                    unknown = True
                                if edge.description in emmaa_inhibits:
                                    sign = -sign
                                evidence.append(json.dumps(edge.evidence))

                        if not protein_target:
                            drug = drug_target_nodes[path[i].split('"')[1]]
                            for edge in drug.edges:
                                if edge.node.name == path[i + 1].split('"')[1]:
                                    annotated += "--(" + edge.description + ")-->" + path[i + 1]
                                    if not (edge.description in drugbank_inhibits or edge.description in drugbank_activates):
                                        unknown = True
                                    if edge.description in drugbank_inhibits:
                                        sign = -sign
                                    evidence.append(json.dumps(edge.evidence))

                    virus = virus_human_ppi_nodes[path[-2].split('"')[1]]
                    for edge in virus.edges:
                        if edge.node.name == path[-1].split('"')[1]:
                            annotated += "--(" + edge.description + ")-->" + path[-1]
                        evidence.append(json.dumps(edge.evidence))
                    
                    shortest_paths.at[index, 'annotated'] = annotated
                    length = str(len(path) -1)

                    if unknown:
                        shortest_paths.at[index, 'result'] = 'Unknown'

                        if length in unknowns:
                            unknowns[length] += 1
                        else:
                            unknowns[length] = 1
                    elif sign >= 0:
                        shortest_paths.at[index, 'result'] = 'Activation'
                        
                        if length in activating:
                            activating[length] += 1
                        else:
                            activating[length] = 1
                    else:
                        shortest_paths.at[index, 'result'] = 'Inhibition'
                    
                        if length in inhibiting:
                            inhibiting[length] += 1
                        else:
                            inhibiting[length] = 1
                     
                    shortest_paths.at[index, 'evidence'] = "|".join(evidence) 
                    shortest_paths.at[index, 'drug_id'] = drug.id
                
                shortest_paths.to_csv(kwargs['output'] + input.replace('_k-100-paths', '_k-100-annotated-paths'), sep = '\t', index=False, quoting=3)
                summary['file'].append(input)
                summary['activating'].append(activating)
                summary['inhibiting'].append(inhibiting)
                summary['unknown'].append(unknowns)
                score = 0
                alpha = 2 
                for length in inhibiting:
                    score += inhibiting[length] / (alpha**(int(length)-2))
                for length in activating:
                    score -= activating[length] / (alpha**(int(length)-2))
                summary['score'].append(score)
                
        pd.DataFrame(summary, columns = ['file', 'activating', 'inhibiting', 'unknown', 'score']).sort_values('score', ascending = False).to_csv(kwargs['output'] + 'pathlinker-summary.tsv', sep = '\t', index=False)
    #TODO nicer condition / separate script?
    if (kwargs['pathlinker_output']) :
        exit();

    print("Network summaries:")

    print("Number of nodes in EMMAA Human PPI: " + str(len(human_ppi_nodes.keys())))
    num_edges = 0
    for node in human_ppi_nodes.values():
        num_edges += len(node.edges)
    print("Number of edges in EMMAA Human PPI: " + str(num_edges) + "\n")

    print("Number of nodes in DrugBank drug targets: " + str(len(drug_target_nodes.keys())))
    num_edges = 0
    for node in drug_target_nodes.values():
        num_edges += len(node.edges)
    print("Number of edges in DrugBank drug targets: " + str(num_edges) + "\n")

    print("Number of nodes in Krogan Human Virus PPI: " + str(len(virus_human_ppi_nodes.keys())))
    num_edges = 0
    for node in virus_human_ppi_nodes.values():
        num_edges += len(node.edges)
    print("Number of edges in Krogan Human Virus PPI: " + str(num_edges) + "\n")         

    print("Number of nodes in intersection of EMMAA, Krogan, and DrugBank: " + str(len(set(human_ppi_nodes.keys()) & set(drug_target_nodes.keys()) & set(virus_human_ppi_nodes.keys()))))
    print("Number of nodes in intersection of EMMAA and Krogan: " + str(len(set(human_ppi_nodes.keys()) & set(virus_human_ppi_nodes.keys()))))
    print("Number of nodes in intersection of EMMAA and DrugBank: " + str(len(set(human_ppi_nodes.keys()) & set(drug_target_nodes.keys()))))
    print("Number of nodes in intersection of Krogan and DrugBank: " + str(len(set(virus_human_ppi_nodes.keys()) & set(drug_target_nodes.keys()))))
    

    print("Generating PathLinker input...")
    relevant_drug_targets = (set(human_ppi_nodes.keys()) & set(drug_target_nodes.keys())).union( 
                            (set(virus_human_ppi_nodes.keys()) & set(drug_target_nodes.keys())))

    with open(kwargs['output'] + '/network.tsv', 'w') as network:
        for node in human_ppi_nodes:
            for edge in human_ppi_nodes[node].edges:
                network.write("\t".join(['"' + human_ppi_nodes[node].name + '"', '"' + edge.node.name + '"', str(edge.weight)]) + '\n')

    drug_sources = []
    with open(kwargs['output'] + '/network.tsv', 'a') as network:
        for node in drug_target_nodes:
            drug_source = False
            for target in drug_target_nodes[node].edges:
                 if target.node.name in relevant_drug_targets:
                     network.write("\t".join(['"' + drug_target_nodes[node].name + '"', '"' + target.node.name + '"', str(edge.weight)]) + '\n')
                     drug_source = True
            
            if drug_source:
                drug_sources.append(node)
    
    virus_targets = []
    with open(kwargs['output'] + '/network.tsv', 'a') as network:
        for node in virus_human_ppi_nodes:
            if virus_human_ppi_nodes[node].type == 'virus':
                virus_targets.append(node)
            for edge in virus_human_ppi_nodes[node].edges:
                network.write("\t".join(['"' + virus_human_ppi_nodes[node].name + '"', '"' + edge.node.name + '"', str(edge.weight)]) + '\n') # '1']) + '\n')

#TODO break up into methods: parseDrug, parseKrogan, generatePathLinker input, etc.
#TODO drug filter config
    for drug in drug_sources:
        with open(kwargs['output'] + '/' + drug.replace(" ", "_").replace("/","_") +'.tsv', 'w') as types:
            types.write("\t".join(['"' + drug + '"', 'source']) + '\n')
            for target in virus_targets:
                types.write("\t".join(['"' + target + '"', 'target']) + '\n')

    # the sign of a path is the product of the signs of the edges in it. We assing "+" to "activation" and "-" to inhibition. For now, we are ignoring other edges.
    
    # read DFA for paths with positive sign.
    # read DFA for paths with negative sign.
    
    # for each source, compute k shortest regular-language constrained paths to each target where the sign of the path is positive.

    # for each source, compute k shortest regular-language constrained paths to each target where the sign of the path is negative.


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)
    
