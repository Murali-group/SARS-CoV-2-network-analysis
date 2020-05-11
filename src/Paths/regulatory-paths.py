# standalone script to parse EMMAA files from NDEx and analyse them for logical paths from a source to a target.


from ndex2.nice_cx_network import NiceCXNetwork
import ndex2.client as nc
import ndex2
import networkx as nx
import pandas as pd
import os



def main(config_map, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """
    # convert file into cx format (used by NDEx). TODO: read this file name from a config file.
    emmaa_covid_digraph_cx = ndex2.create_nice_cx_from_file('../datasets/regulatory-networks/2020-04-02-bachman-emmaa-covid19.cx')
    emmaa_covid_digraph_cx.print_summary()
    # convert cx-format graph to NetworkX format. notice that second last letter is 'n' and not 'c', as in the previous variable.
    emmaa_covid_digraph_nx = nx.DiGraph()
    ndex2.nice_cx_network.DefaultNetworkXFactory().get_graph(emmaa_covid_digraph_cx, emmaa_covid_digraph_nx)
    # I want to check how large the edgelist file is. 103 MB as a edge list vs. 176MB in cx format. TODO: read this file name from a config file.
    nx.write_edgelist(emmaa_covid_digraph_nx, '../datasets/regulatory-networks/2020-04-02-bachman-emmaa-covid19.edgelist')

    # read file of sources (e.g., predictions with scores/ranks)
    # read file of targets (e.g., human proteins that interact with SARS-CoV-2 proteins)

    # the sign of a path is the product of the signs of the edges in it. We assing "+" to "activation" and "-" to inhibition. For now, we are ignoring other edges.
    
    # read DFA for paths with positive sign.
    # read DFA for paths with negative sign.
    
    # for each source, compute k shortest regular-language constrained paths to each target where the sign of the path is positive.

    # for each source, compute k shortest regular-language constrained paths to each target where the sign of the path is negative.
    


if __name__ == "__main__":
    config_map, kwargs = [0, 1] #parse_args()
    main(config_map) #, kwargs)
    
