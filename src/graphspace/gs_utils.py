import os, sys
import pandas as pd
import networkx as nx

METHODS_TO_IGNORE = set()


##########################################################
def evidenceToHTML(t,h,evidencelist):
    annotation = '<dl>'
    sources = sorted(evidencelist)

    for source in sources:
        if source == 'STRING':
            # Make it a link: https://string-db.org/cgi/network.pl?identifiers=P05412%0dP01100
            url = f"https://string-db.org/cgi/network.pl?identifiers={t}%0d{h}"
            desc = f'<a style="color:blue" href="{url}" target="STRING">{source}</a>'
            annotation += '<dt>%s</dt>' % (desc)
            for string_channel in sorted(evidencelist[source]):
                score = list(evidencelist[source][string_channel].keys())[0]
                annotation += '&bull;&nbsp&nbsp%s:&nbsp%s<br>' % (string_channel, score)
            continue
        annotation += '<dt>%s</dt>' % (source)
        # TODO add interaction type color
        for interactiontype in evidencelist[source]:
            if interactiontype != '' and interactiontype != "None":
                # use bull instead of li to save on white space
                annotation += '&bull;&nbsp&nbsp%s <br>' % interactiontype
            for detectionmethod in evidencelist[source][interactiontype]:
                annotation += '&nbsp&nbsp&nbsp&nbsp&bull;&nbsp&nbsp'
                if detectionmethod != '':
                    filtered = False
                    for m in METHODS_TO_IGNORE:  # CSBDB
                        if m in detectionmethod:
                            filtered = True
                            break
                    # add detection method filter (i.e. gray out detection methods that we're ignoring)
                    if filtered:
                        annotation += '<FONT COLOR="gray">%s</FONT>  ' % (detectionmethod)
                    else:
                        annotation += '%s  ' % detectionmethod

                # now add the pubmed IDs. &nbsp is the html for a non-breaking space
                pub_ids = evidencelist[source][interactiontype][detectionmethod]
                #KEGG doesn't have pub ids. It has a pathway map and entry (evidence)
                try:
                    # How do we want to sort the pmid, imex, doi and MINT and such? pmid first?
                    pubmed_ids = [pubmed_id for pubmed_id in pub_ids if pubmed_id.split(':')[0] == 'pubmed' and 'None' not in pubmed_id]
                    # just sort the pubmed_ids and put them first
                    #pubmed_ids = sortPubs(pubmed_ids)
                    # add the rest of the ids
                    pub_ids = pubmed_ids + [other_id for other_id in pub_ids if other_id.split(':')[0] != 'pubmed']
                    # now get the html for each of the links
                    pub_ids = [parseCSBDBpubs(pub_id) for pub_id in pub_ids if parseCSBDBpubs(pub_id) != '']
                except ValueError:
                    print("ERROR when parsing pub_ids from:")
                    print(t,h, source, interactiontype, detectionmethod, pub_ids)
                    raise

                # use a non-breaking space with a comma so they all stay on the same line
                annotation += ',&nbsp'.join(pub_ids)
                annotation += "<br>"
        annotation += '</li></ul><br>'

    return annotation


def getEvidence(edges, evidence_file=None):
    """
    *edges*: a set of edges for which to get the evidence for
    returns a multi-level dictionary with the following structure
    edge: 
      db/source: 
        interaction_type: 
          detection_method: 
            publication / database ids
    For NetPath, KEGG and SPIKE, the detection method is the pathway name 
    NetPath, KEGG, and Phosphosite also have a pathway ID in the database/id set
    which follows convention of id_type:id (for example: kegg:hsa04611).
    For STRING, the sub-channel is in the interaction_type col and the score is in the detection method
    """
    # dictionary of an edge mapped to the evidence for it
    evidence = {} 
    edge_types = {}
    edge_dir = {}

    if evidence_file is None:
        print("evidence_file not given. Returning empty sets")
        return evidence, edge_types, edge_dir
        # TODO use a default version or something

    print("Reading evidence file %s" % (evidence_file))
    #evidence_lines = utils.readColumns(evidence_file, 1,2,3,4,5,6,7)
    df = pd.read_csv(evidence_file, sep='\t')
    print("\t%d lines" % (len(df)))
    print(df.head())

    # create a graph of the passed in edges 
    G = nx.Graph()
    G.add_edges_from(edges)

    # initialize the dictionaries
    for t,h in G.edges():
        evidence[(t,h)] = {}
        evidence[(h,t)] = {}
        edge_types[(t,h)] = set()
        edge_types[(h,t)] = set()
        edge_dir[(t,h)] = False
        edge_dir[(h,t)] = False

    for u,v, directed, interactiontype, detectionmethod, pubid, source in df.values:
        # header line of file
        #uniprot_a  uniprot_b   directed    interaction_type    detection_method    publication_id  source
        directed = True if directed == "True" else False

        # We only need to get the evidence for the edges passed in, so if this edge is not in the list of edges, skip it
        # G is a Graph so it handles undirected edges correctly
        if not G.has_edge(u,v):
            continue

        evidence = addToEvidenceDict(evidence, (u,v), directed, source, interactiontype, detectionmethod, pubid)
        edge_types = addEdgeType(edge_types, (u,v), directed, source, interactiontype)

        if directed is True:
            edge_dir[(u,v)] = True
        else:
            # if they are already directed, then set them as undirected
            if (u,v) not in edge_dir:
                edge_dir[(u,v)] = False
                edge_dir[(v,u)] = False

    return evidence, edge_types, edge_dir


def addEdgeType(edge_types, e, directed, source, interactiontype):
    if e not in edge_types:
        edge_types[e] = set()
    # add the edge type as well
    # direction was determined using the csbdb_interface.psimi_interaction_direction dictionary in CSBDB. 
    if not directed:
        # it would be awesome if we knew which edges are part of complex formation vs other physical interactions
        # is there an mi term for that?
        t,h = e
        edge_types[(t,h)].add('physical')
        if (h,t) not in edge_types:
            edge_types[(h,t)] = set()
        edge_types[(h,t)].add('physical')

    # include this to be able to reproduce evidence where there is no spike interaction type
    elif source == "SPIKE" and interactiontype == '':
        edge_types[e].add('spike_regulation')

    elif source == "KEGG" or source == "SPIKE":
        interactiontype = interactiontype.lower()
        if "phosphorylation" in interactiontype and "dephosphorylation" not in interactiontype:
            # Most of them are phosphorylation
            edge_types[e].add('phosphorylation')
        if "activation" in interactiontype:
            edge_types[e].add('activation')
        if "inhibition" in interactiontype:
            edge_types[e].add('inhibition')
        for edgetype in ["dephosphorylation", "ubiquitination", "methylation", "glycosylation"]:
            if edgetype in interactiontype:
                # TODO for now, just call the rest of the directed edges enzymatic. 
                edge_types[e].add('enzymatic')
    else:
        # the rest is CSBDB
        #if "(phosphorylation reaction)" in interactiontype:
        # MI:0217 is safer because that will only ever match phosphorylation
        if "MI:0217" in interactiontype:
            # Most of the directed edges are phosphorylation
            edge_types[e].add('phosphorylation')
        else:
            # TODO for now, just call the rest of the directed edges enzymatic. 
            edge_types[e].add('enzymatic')

    return edge_types


def addToEvidenceDict(evidence, e, directed, source, interactiontype, detectionmethod, pubid):
    """ add the evidence of the edge to the evidence dictionary
    *pubids*: publication id to add to this edge. 
    """
    #if e not in evidence:
    #    evidence[e] = {}
    if source not in evidence[e]:
        evidence[e][source] = {}
    if interactiontype not in evidence[e][source]:
        evidence[e][source][interactiontype] = {}
    if detectionmethod not in evidence[e][source][interactiontype]:
        evidence[e][source][interactiontype][detectionmethod] = set()
    evidence[e][source][interactiontype][detectionmethod].add(pubid)

    if not directed:
        # add the evidence for both directions
        evidence = addToEvidenceDict(evidence, (e[1],e[0]), True, source, interactiontype, detectionmethod, pubid)

    return evidence


##########################################################
def parseCSBDBpubs(publication_id):
    row = publication_id.split(':')
    if row[0] == 'pubmed':
        pubmedurl = 'http://www.ncbi.nlm.nih.gov/pubmed/%s' % (row[1])
        desc = '<a style="color:blue" href="%s" target="PubMed">pmid:%s</a>' % (pubmedurl,row[1])
    elif row[0] == 'doi':
        doiurl = 'http://dx.doi.org/%s' % (row[1])
        # replace the dash with a non-linebreaking hyphen
        desc = '<a style="color:blue" href="%s" target="DOI">doi:%s</a>' % (doiurl,row[1].replace('-','&#8209;'))
    elif row[0] == 'omim':
        omimurl = 'http://omim.org/entry/%d' % (int(row[1]))
        desc = '<a style="color:blue" href="%s" target="OMIM">omim:%s</a>' % (omimurl,row[1])
    elif row[0] == 'imex':
        imexurl = 'http://www.ebi.ac.uk/intact/imex/main.xhtml?query=%s' % (row[1])
        # replace the dash with a non-linebreaking hyphen
        desc = '<a style="color:blue" href="%s" target="IMEX">imex:%s</a>' % (imexurl,row[1].replace('-','&#8209;'))
    elif row[0] == 'phosphosite':
        phosphourl = 'http://www.phosphosite.org/siteAction.action?id=%s' % (row[1])
        desc = '<a style="color:blue" href="%s" target="PSP">phosphosite:%s</a>' % (phosphourl,row[1])
    elif row[0] == 'mint' or row[0] == 'dip':
        # MINT links are taking too long to load. The website often times out.
        # MINT links are often accompanied by IntAct links which seem to have the same info
        # http://mint.bio.uniroma2.it/mint/search/interaction.do?ac=MINT-8200651
        # not sure what the dip links are but they are also accompanied by pubmed IDs
        desc = ''
    elif row[0] == 'kegg':
        # links to KEGG pathway map
        kegg_map_link = 'http://www.kegg.jp/kegg-bin/show_pathway?'
        # links to KEGG pathway entry (evidence)
        kegg_entry_link = 'http://www.kegg.jp/dbget-bin/www_bget?pathway+'
        pathway_map_link = '<a style="color:blue" href="%s%s" target="KEGG">map</a>' % (kegg_map_link,row[1])
        pathway_entry_link = '<a style="color:blue" href="%s%s" target="KEGG">evidence</a>' % (kegg_entry_link,row[1])
        desc = "%s,&nbsp%s"%(pathway_map_link,pathway_entry_link)
    elif row[0] == 'netpath':
        # add a link to the netpath page
        # http://www.netpath.org/reactions?path_id=NetPath_12
        netpath_url = "http://www.netpath.org/reactions?path_id=%s" % (row[1])
        # links to KEGG pathway entry (evidence)
        desc = '<a style="color:blue" href="%s" target="NetPath">netpath:%s</a>' % (netpath_url, row[1])
    else:
        print("Unkown publication id '%s'," % publication_id,)
        # replace the dash with a non-linebreaking hyphen
        #desc = publication_id.replace('-','&#8209;')
        desc = ''

    return desc


