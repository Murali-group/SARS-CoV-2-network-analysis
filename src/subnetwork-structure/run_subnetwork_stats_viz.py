import networkx as nx
import gzip
import sys
import collections
import matplotlib.pyplot as plt

from graphspace_python.api.client import GraphSpace
from graphspace_python.graphs.classes.gsgraph import GSGraph

krogan_positive_file = '../../fss_inputs/pos-neg/2020-03-sarscov2-human-ppi/2020-03-24-sarscov2-human-ppi.txt'
interactome_files = {'stringv11':'../../datasets/networks/stringv11/9606-uniprot-links-full-v11.txt.gz',
 	'tissuenet_v2-hpa-rnaseq-lung':'../../datasets/networks/tissuenet-v2/hpa-rnaseq/lung.tsv.gz',
 	'tissuenet_v2-hpa-protein-lung':'../../datasets/networks/tissuenet-v2/hpa-protein/lung.tsv.gz',
 	'tissuenet_v2-gtex-rnaseq-lung':'../../datasets/networks/tissuenet-v2/gtex-rnaseq/lung.tsv.gz' }
interactome_score_col = {'stringv11':15,
 	'tissuenet_v2-hpa-rnaseq-lung':2,
 	'tissuenet_v2-hpa-protein-lung':2,
 	'tissuenet_v2-gtex-rnaseq-lung':2}
uniprot_mapping_file = '../../datasets/mappings/human/uniprot-reviewed-status.tab.gz'

def main(username,password):
	graphspace = connect_to_graphspace(username,password)

	## positves
	positives = get_positives()

	## mapped names
	mapped = get_mapped_names()

	##interactome list
	interactomes = ['tissuenet_v2-hpa-protein-lung','tissuenet_v2-gtex-rnaseq-lung','tissuenet_v2-hpa-rnaseq-lung','stringv11']

	graphs = {}
	for ppi_name in interactomes:
		print('processing',ppi_name)
		G = nx.Graph()
		G.add_nodes_from(positives)
		with gzip.open(interactome_files[ppi_name], 'rb') as fin:
			for line in fin:
				row = line.decode().strip().split()
				if row[0] in positives and row[1] in positives:
					score_col = interactome_score_col[ppi_name]
					if ppi_name != 'stringv11' or float(row[score_col])>400:
						G.add_edge(row[0],row[1],weight=row[score_col])
		if ppi_name == 'stringv11':
			post_to_graphspace(graphspace,G,mapped,'%s > 400 (Krogan Positives)' % (ppi_name),gs_group='SARS-CoV-2-network-analysis')
		else:
			post_to_graphspace(graphspace,G,mapped,'%s (Krogan Positives)' % (ppi_name),gs_group='SARS-CoV-2-network-analysis')
		graphs[ppi_name] = G
		
	fig = plt.figure(figsize=(8,5))
	ax1 = plt.subplot(1,2,1)
	ax2 = plt.subplot(1,2,2)
	ax2.set_yscale('log')
	ax2.set_xscale('log')
	for ppi_name in graphs:
		G = graphs[ppi_name]
		degreeCount  = get_stats(ppi_name,G)
		deg, cnt = zip(*degreeCount.items())
		ax1.plot(deg,cnt,'o--',ms=7,label=ppi_name)
		ax2.plot(deg,cnt,'o--',ms=7,label=ppi_name)
	ax1.set_title('Krogan-Induced Subgraph')
	ax2.set_title('Krogan-Induced Subgraph (Log-Log)')
	ax1.set_ylabel('#')

	ax2.set_ylabel('Log #')
	ax1.set_xlabel('Degree')
	plt.legend()
	ax2.set_xlabel('Log Degree')
	plt.tight_layout()
	plt.savefig('degree_distributions.png')

	return

def get_stats(ppi_name,G):
	print('%s Graph has %d nodes and %d edges' % (ppi_name,nx.number_of_nodes(G),nx.number_of_edges(G)))
	print('%s Graph has %d connected components (largest contains %d nodes)' % (ppi_name,nx.number_connected_components(G),len(max(nx.connected_components(G), key=len))))
	# from https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_degree_histogram.html
	degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
	degreeCount = collections.Counter(degree_sequence)
	
	#for deg in degreeCount:
	#	print(' %d nodes have degree %d' % (degreeCount[deg],deg))
	print()
	return degreeCount


def post_to_graphspace(graphspace,network,mapped,title,gs_group=None):
	G = GSGraph()
	G.set_name(title)

	## get max degree
	max_degree = max([a[1] for a in network.degree()])
	print('max degree is %d' % (max_degree))

	## color nodes by degree distribution
	for node in network.nodes():
		this_degree = network.degree(node)
		G.add_node(node,label=mapped.get(node,node),popup='<a href="https://www.uniprot.org/uniprot/%s" target="_blank" style="color:#0000FF;">UniProtID: %s</a><br>Degree: %d' % (node,node,this_degree))
		
		color = rgb_to_hex(1-this_degree/max_degree/2,0.8,1-this_degree/max_degree)
		#print(node,this_degree,color)
		G.add_node_style(node,shape='rectangle',color=color,width=90,height=45)
	
	for u,v in network.edges():
		G.add_edge(u,v,popup='Weight: %s' % (network[u][v]['weight']))
		G.add_edge_style(u,v,width=2)

	graph = post(G,graphspace,gs_group)
	print('posted graph "%s"' % (title))
	if gs_group:
		print('shared graph with group "%s"' % (gs_group))

	return

def rgb_to_hex(r,g,b):
	return '#%02x%02x%02x'.upper() % (int(r*255),int(g*255),int(b*255))

def get_positives():
	positives = set()
	with open(krogan_positive_file) as fin:
		for line in fin:
			positives.add(line.strip())
	print('%d positives from Krogan et al.' % (len(positives)))
	return positives


def get_mapped_names():
	mapped = {}
	with gzip.open(uniprot_mapping_file, 'rb') as fin:
		for line in fin:
			row = line.decode().strip().split()
			if row == 'Entry':
				continue
			mapped[row[0]] = row[2]
			#print(row[0],row[2])
	return mapped


def connect_to_graphspace(username,password):
	graphspace = GraphSpace(username,password)
	return graphspace

def post(G,gs,gs_group):
	try:
		graph = gs.update_graph(G)
	except:
		graph = gs.post_graph(G)
		if gs_group:
			gs.share_graph(graph=graph,group_name=gs_group)
	return graph


if __name__ == '__main__':
 	main(sys.argv[1],sys.argv[2])