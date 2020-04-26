!#/bin/bash

#filter out Krogan positive examples from output files
python filter.py

#Do GO enrichment analysis with clusterProfiler
Rscript prediction_GoEnrichment.R

#combine ouput of cluster profiler for each algorithm(separately) across all networks 
python combine_enrichment_result.py
