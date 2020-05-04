import networkx as nx
import gzip
import sys
import collections
import matplotlib.pyplot as plt
import time
import argparse
import copy 

from graphspace_python.api.client import GraphSpace
from graphspace_python.graphs.classes.gsgraph import GSGraph

import PathLinker as pl 
import ksp_Astar as ksp

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
drug_target_file = '../../datasets/drug-targets/pharmgkb/prot-chem-relationships.tsv'

def parse_args():
	parser = argparse.ArgumentParser(description='viz and analysis some simple network statistics about the Krogan nodes, the PPIs, and the drug-protein interactions.')

	parser.add_argument('--gs',action='store_true',help='Make GraphSpace Graphs.')
	parser.add_argument('--username',type=str,help='GraphSpace username. Required if --gs option is specified.')
	parser.add_argument('--password',type=str,help='GraphSpace password. Required if --gs option is specified.')
	parser.add_argument('--krogan-subgraph',action='store_true',help='Print and plot statistics about the Krogan induced subgraph.')
	parser.add_argument('--drugs',action='store_true',help='Run PathLinker on the drug-ppi networks.')
	parser.add_argument('--drugs_direct',action='store_true',help='Extract direct neighbors of drug-ppi networks (corresponds to PL paths of 1-3)')

	args = parser.parse_args()
	if args.gs and not (args.username or args.password):
		sys.exit('Error: --username and --password required if --gs option is specified. Exiting.')
	return args

def main(krogan_subgraph=None,drugs=None,drugs_direct=None,gs=None,username=None,password=None):
	if gs:
		graphspace = connect_to_graphspace(username,password)

	## positves
	positives = get_positives()

	## mapped names
	mapped = get_mapped_names()

	if drugs or drugs_direct:
		drug_edges,mapped_drugs = get_drug_edges()

	##interactome list
	interactomes = ['tissuenet_v2-hpa-protein-lung','tissuenet_v2-gtex-rnaseq-lung','tissuenet_v2-hpa-rnaseq-lung']#,'stringv11']

	induced_subgraphs = {}
	for ppi_name in interactomes:
		print('processing',ppi_name)
		score_col = interactome_score_col[ppi_name]

		G = nx.Graph()
		G.add_nodes_from(positives)
		if drugs or drugs_direct:  ## make full netwokr (directed)
			fullG = nx.DiGraph()

		with gzip.open(interactome_files[ppi_name], 'rb') as fin:
			for line in fin:
				row = line.decode().strip().split()
				if drugs or drugs_direct:
					if ppi_name != 'stringv11' or float(row[score_col])>400:
						fullG.add_edge(row[0],row[1],weight=float(row[score_col]))
						fullG.add_edge(row[1],row[0],weight=float(row[score_col]))
				if row[0] in positives and row[1] in positives:
					if ppi_name != 'stringv11' or float(row[score_col])>400:
						G.add_edge(row[0],row[1],weight=float(row[score_col]))
		if krogan_subgraph and gs:
			if ppi_name == 'stringv11':
				post_to_graphspace(graphspace,G,mapped,'%s > 400 (Krogan Positives)' % (ppi_name), gs_group='SARS-CoV-2-network-analysis')
			else:
				post_to_graphspace(graphspace,G,mapped,'%s (Krogan Positives)' % (ppi_name), gs_group='SARS-CoV-2-network-analysis')
		
		induced_subgraphs[ppi_name] = G

		if drugs or drugs_direct: ## add drugs and run PathLinker
			## NOTE: if string, we need tom odify to log-transform edges.
			print(' Before adding drugs: Graph has %d nodes and %d edges' % (nx.number_of_nodes(fullG),nx.number_of_edges(fullG)))
			incident_edges = set([e for e in drug_edges if e[0] in fullG.nodes()])
			print(' adding %d node-drug edges' % (len(incident_edges)))
			for u,v,w in incident_edges:
				fullG.add_edge(u,v,weight=w)
			print(' After adding drugs: Graph has %d nodes and %d edges' % (nx.number_of_nodes(fullG),nx.number_of_edges(fullG)))
			to_remove = set([e for e in G.edges() if e[1] in positives])
			print(' removing %d edges coming into Krogan nodes' % (len(to_remove)))
			fullG.remove_edges_from(to_remove)
			print(' Graph has %d nodes and %d edges' % (nx.number_of_nodes(fullG),nx.number_of_edges(fullG)))
			sources = positives
			targets = set([e[1] for e in incident_edges]) # drugs

		if drugs:
			## add supersource and supersink
			print(' Adding supersource and supersink')
			for s in sources:
				fullG.add_edge('source',s,weight=1)
			for t in targets:
				fullG.add_edge(t,'target',weight=1)
			print(' Final: Graph has %d nodes and %d edges' % (nx.number_of_nodes(fullG),nx.number_of_edges(fullG)))

			relevant_nodes = sources.union(targets)
			relevant_nodes.add('source')
			relevant_nodes.add('target')
			#induced_subgraph = fullG.subgraph(relevant_nodes).copy()
			
			## run KSP
			pathgraph,k = run_KSP(fullG,'PL-%s' % (ppi_name),kval=500)
			#run_KSP(induced_subgraph,'PL-%s' % (ppi_name))

			if gs:
				#if k <= 200:
				#	post_ksp_to_graphspace(graphspace,pathgraph,sources,targets,mapped,mapped_drugs,'PL %s (k=%d) %f' % (ppi_name,k,time.time()),gs_group=None)
				num_s = len(sources.intersection(pathgraph.nodes()))
				num_t = len(targets.intersection(pathgraph.nodes()))
				title='%s: %d shortest paths from %d Krogan proteins (%.2f of total) to %d drugs (%.2f of total)' % (ppi_name,k,num_s,num_s/len(sources),num_t,num_t/len(targets))
				post_ksp_to_graphspace(graphspace,pathgraph,sources,targets,mapped,mapped_drugs,title,gs_group='SARS-CoV-2-network-analysis',simplify=True)

		if drugs_direct:
			neighbor_nodes = set()
			for s in sources:
				if s in fullG:
					neighbor_nodes.update(fullG.successors(s))
			for t in targets:
				if t in fullG:
					neighbor_nodes.update(fullG.predecessors(t))
			print(' %d sources, %d targets, and %d neighbors'% (len(sources),len(targets),len(neighbor_nodes)))
			induced_subgraph = nx.DiGraph(fullG.subgraph(sources.union(targets).union(neighbor_nodes))).to_undirected()
			print(' subgraph: Graph has %d nodes and %d edges' % (nx.number_of_nodes(induced_subgraph),nx.number_of_edges(induced_subgraph)))
			## remove neighbor_nodes that don't have at least one source and one target.
			to_remove = set()
			to_remove_edges = set()
			first = True
			while len(to_remove)+len(to_remove_edges) > 0 or first:
				first = False
				to_remove = set()
				to_remove_edges = set()
				for n in induced_subgraph.nodes():
					neighs = set(induced_subgraph.neighbors(n))
					if n in sources or n in targets:
						if len(neighs)==0:
							to_remove.add(n)
						for p in neighs:
							if n in sources and p in sources:
								to_remove_edges.add((p,n))
						continue
					
					if len(sources.intersection(neighs)) == 0 or len(targets.intersection(neighs)) == 0:
						to_remove.add(n)
					else:
						for p in neighs:
							if p in neighbor_nodes:
								to_remove_edges.add((p,n))
		
				induced_subgraph.remove_nodes_from(to_remove)
				induced_subgraph.remove_edges_from(to_remove_edges)
				print(' after removing %d nodes and %d addt\'l edges: Graph has %d nodes and %d edges' % (len(to_remove),len(to_remove_edges),nx.number_of_nodes(induced_subgraph),nx.number_of_edges(induced_subgraph)))
			title = '%s: Krogan-drug links (path lengths of 1 or 2)' % (ppi_name)
			post_ksp_to_graphspace(graphspace,induced_subgraph,sources,targets,mapped,mapped_drugs,title,gs_group='SARS-CoV-2-network-analysis',simplify=True, simplify_links=True)


	if krogan_subgraph:
		plot_degree_dist(interactomes,induced_subgraphs)

	return

def run_KSP(fullG,outprefix, kval=500): ## modified from PL's run.y

	paths = ksp.k_shortest_paths_yen(fullG, 'source', 'target', kval, weight='weight', clip=True,verbose=True)
	#print(paths)

	# Prepare the k shortest paths for output to flat files
	pathgraph = nx.DiGraph()
	ksp_ids = {}
	for k,path in enumerate(paths, 1):

		# Process the edges in this path
		edges = []
		for i in range(len(path)-1):
			t = path[i][0]
			h = path[i+1][0]

			# Skip edges that have already been seen in an earlier path
			if pathgraph.has_edge(t, h):
				continue

			# Skip edges that include our artificial supersource or
			# supersink
			if t=='source' or h=='target':
				continue

			# This is a new edge. Add it to the list and note which k it
			# appeared in.
			else:
				edges.append( (t,h,{
					'ksp_id':k, 
					'ksp_weight':fullG[t][h]['weight'],
					'path_cost': path[-1][1]}) )

		# Add all new, good edges from this path to the network
		pathgraph.add_edges_from(edges)

		# Each node is ranked by the first time it appears in a path.
		# Identify these by check for any nodes in the graph which do
		# not have 'ksp_id' attribute, meaning they were just added
		# from this path.
		for n in pathgraph.nodes():
			if 'ksp_id' not in ksp_ids:
				ksp_ids[n] = k
	nx.set_node_attributes(pathgraph,ksp_ids,'ksp_id')


	## Write out the results to file    

	# Write a list of all edges encountered, ranked by the path they
	# first appeared in.
	kspGraphOutfile = '%s-k-%d-ranked-edges.txt' %(outprefix, kval)
	pl.printKSPGraph(kspGraphOutfile, pathgraph)
	print('\nKSP results are in "%s"' %(kspGraphOutfile))

	# Write a list of all paths found by the ksp algorithm, if
	# requested.
	kspOutfile = '%sk-%d-paths.txt' %(outprefix, kval)
	pl.printKSPPaths(kspOutfile, paths)
	print('KSP paths are in "%s"' %(kspOutfile))

	return pathgraph,kval

def get_drug_edges():
	edges = set()
	drug2name = {}
	with open(drug_target_file) as fin:
		for line in fin:
			row = line.strip().split()
			if row[0] == 'Entity1_id':
				continue
			edges.add((row[0],row[3],1))
			drug2name[row[3]] = row[4]
	print('%d protein-drug edges and %d unique drugs' % (len(edges),len(drug2name)))
	return edges,drug2name

def post_to_graphspace(graphspace,network,mapped,title,gs_group=None):
	print(' Posting graph with %d nodes and %d edges' % (nx.number_of_nodes(network),nx.number_of_edges(network)))
	G = GSGraph()
	G.set_name(title)

	## get max degree
	max_degree = max([a[1] for a in network.degree()])
	print('max degree is %d' % (max_degree))

	## color nodes by degree distribution
	for node in network.nodes():
		this_degree = network.degree(node)
		color = rgb_to_hex(this_degree/max_degree,0.7,1-this_degree/max_degree)
		G.add_node(node,label=mapped.get(node,node),popup='<a href="https://www.uniprot.org/uniprot/%s" target="_blank" style="color:#0000FF;">UniProtID: %s</a><br>Degree: %d<br>Color: %s' % (node,node,this_degree,color))
		
		
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

def post_ksp_to_graphspace(graphspace,orig,sources,targets,mapped,mapped_drugs,title,simplify=False,simplify_links=False,gs_group=None):
	print(' Posting graph with %d nodes and %d edges' % (nx.number_of_nodes(orig),nx.number_of_edges(orig)))
	simplified_nodes = {}
	if simplify:
		pathgraph = copy.deepcopy(orig)
		
		edges_to_remove = set()
		edges_to_add = set()
		nodes_to_remove = set()
		for n in pathgraph.nodes():
			if n in targets: # skip drugs
				continue
			neighbors = pathgraph.neighbors(n)
			neighbors_to_collapse = set([c for c in neighbors if c in targets and pathgraph.degree(c)==1])
			if len(neighbors_to_collapse)>1:
				new_node = '%s-neighbors' % (n)
				simplified_nodes[new_node]=neighbors_to_collapse
				edges_to_add.add((n,new_node))
				for c in neighbors_to_collapse:
					edges_to_remove.add((n,c))
					nodes_to_remove.add(c)
		pathgraph.add_edges_from(edges_to_add)
		pathgraph.remove_edges_from(edges_to_remove)
		pathgraph.remove_nodes_from(nodes_to_remove)
		print(' After simplifying, Posting graph with %d nodes and %d edges' % (nx.number_of_nodes(pathgraph),nx.number_of_edges(pathgraph)))
	else:
		pathgraph=orig


	if simplify_links:
		edges_to_add = set()
		nodes_to_remove = set()
		s2t ={}
		for s in sources:
			s2t[s] = {t:set() for t in targets}
		for n in pathgraph.nodes():
			if n in sources or n in targets or '-neighbors' in n:
				continue

			neighbors = list([val for val in pathgraph.neighbors(n) if '-neighbors' not in val])
			if len(neighbors)==2:
				#print('THIS NODE:',n,n in sources,n in targets)
				#print('NEIGHBOR #0:',neighbors[0],neighbors[0] in sources,neighbors[0] in targets)
				#print('NEIGHBOR #1:',neighbors[1],neighbors[1] in sources, neighbors[1] in targets)
				#print()
				if neighbors[0] in sources and neighbors[1] in targets:
					s2t[neighbors[0]][neighbors[1]].add(n)
				elif neighbors[1] in sources and neighbors[0] in targets:
					s2t[neighbors[1]][neighbors[0]].add(n)
		for s in sources:
			for t in targets:
				if len(s2t[s][t]) > 1: # collapse
					new_node = '%s-%s-links' % (s,t)
					simplified_nodes[new_node]=s2t[s][t]
					edges_to_add.add((s,new_node))
					edges_to_add.add((new_node,t))
					nodes_to_remove.update(set(s2t[s][t]))
		pathgraph.add_edges_from(edges_to_add)
		pathgraph.remove_nodes_from(nodes_to_remove)
		print(' After simplifying links, Posting graph with %d nodes and %d edges' % (nx.number_of_nodes(pathgraph),nx.number_of_edges(pathgraph)))

	print(' Posting graph with %d nodes and %d edges' % (nx.number_of_nodes(pathgraph),nx.number_of_edges(pathgraph)))
	#sys.exit()
	G = GSGraph()
	G.set_name(title)

	for node in pathgraph.nodes():
		if node not in targets and node not in simplified_nodes: ## proteins - rectangls
			shape='rectangle'
			width=90
			height=45
			label=mapped.get(node,node)
			popup='<a href="https://www.uniprot.org/uniprot/%s" target="_blank" style="color:#0000FF;">UniProtID: %s</a>' % (node,node)
			if node in sources: # krogan nodes
				color='#00B2FE'
			elif node in simplified_nodes: ## multi human proteins
				color='#888888'
				label='%d human proteins' % (len(simplified_nodes[node]))
				print(label)
				popup = ''
				for c in simplified_nodes[node]:
					popup+='%s <a href="https://www.uniprot.org/uniprot/%s" target="_blank" style="color:#0000FF;">UniProtID: %s</a><br>' % (mapped.get(c,c),c,c)
			else:
				color='#AAAAAA'

		elif node in targets or node in simplified_nodes: ## drugs - red diamonds
			shape='diamond'
			height=75
			width=90
			if node in targets:
				color='#FF5555'
				label=mapped_drugs.get(node,node)
				popup='<a href="https://www.pharmgkb.org/chemical/%s" target="_blank" style="color:#0000FF;">PharmGKB: %s</a>' % (node,node)
			else: # simplified node
				color='#AA0000'
				label='%d drugs' % (len(simplified_nodes[node]))
				popup = ''
				for c in simplified_nodes[node]:
					popup+='%s <a href="https://www.pharmgkb.org/chemical/%s" target="_blank" style="color:#0000FF;">(PharmGKB: %s)</a><br>' % (mapped_drugs.get(c,c),c,c)
		
		G.add_node(node,label=label,popup=popup)
		G.add_node_style(node,shape=shape,color=color,width=width,height=height)
	
	for u,v in pathgraph.edges():
		G.add_edge(u,v,popup='K-Shortest Paths ID: %d' % (pathgraph[u][v].get('ksp_id',-1)))
		G.add_edge_style(u,v,width=2)

	graph = post(G,graphspace,gs_group)
	print('posted graph "%s"' % (title))
	if gs_group:
		print('shared graph with group "%s"' % (gs_group))

	return

def rgb_to_hex(r,g,b):
	return '#%02x%02x%02x'.upper() % (int(r*255),int(g*255),int(b*255))

def plot_degree_dist(interactomes,graphs):
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
	print('Saved to degree_distributions.png')
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
	args = parse_args()
	kwargs = vars(args)
	main(**kwargs)
