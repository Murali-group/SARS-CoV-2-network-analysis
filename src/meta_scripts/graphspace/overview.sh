# term nodes only:
# UPDATE to use most recent results
#simplified_terms_file="outputs/enrichment/simplified/2020-05-11/string_k332_GO-BP_simplified-correct-order.tsv"
#enriched_terms_file="outputs/enrichment/simplified/2020-05-11/string--k332-with-p_adjust.tsv"
simplified_terms_file="outputs/enrichment/simplified/2020-06-03/string-k332-GO-BP-simplified-manual-ordered.tsv"
enriched_terms_file="outputs/enrichment/simplified/2020-06-03/string--k332-with-p_adjust.tsv"

cmd="""python src/graphspace/overview_terms_only.py \
    --config fss_inputs/config_files/string-tissuenet-wace2/string.yaml \
    --enriched-terms-file $enriched_terms_file \
    --simplified-terms-file $simplified_terms_file \
    --parent-nodes \
    --edge-weight-cutoff 900 \
    --user jeffl@vt.edu --pass <pass> \
    --out-pref fss_inputs/graphspace/graphs/overview \
    --name-postfix=-2020-06-03$1 \
    --virus-nodes \
    --out-pref=outputs/graphspace/simplified/ \
    --apply-layout 2020-03-sarscov2-human-ppi-ace2-400-2020-06-03-3   \
    --layout-name layout1 \

    """
echo $cmd
$cmd


# now make copies of the virus nodes for each group of terms
# I manually made this file to split the terms into groups
term_groups_file="outputs/enrichment/simplified/2020-06-03/manual-groups.txt"
cmd="""python overview_terms_copy_virus_nodes.py \
    --net-json-file outputs/graphspace/simplified/2020-03-sarscov2-human-ppi-ace2-400-2020-06-03$1.json \
    --term-groups-file $term_groups_file  \
    --user jeffl@vt.edu --pass <pass> --out-pref outputs/graphspace/simplified-manual/ \
    --graph-name=2020-03-sarscov2-human-ppi-ace2-400-2020-06-03-copy-virnodes \

    """
echo $cmd
$cmd


## ALL NODES overview
##simplified_terms_file="outputs/enrichment/simplified/2020-05-11/string_k332_GO-BP_simplified.tsv"
#simplified_terms_file="outputs/enrichment/simplified/2020-05-11/string_k332_GO-BP_simplified-correct-order.tsv"
#
#cmd="""python src/graphspace/overview_gs.py \
#    --config fss_inputs/config_files/string-tissuenet-wace2/string.yaml \
#    --enriched-terms-file outputs/enrichment/simplified/2020-05-11/string--k332-with-p_adjust.tsv \
#    --simplified-terms-file $simplified_terms_file \
#    --parent-nodes \
#    --edge-weight-cutoff 900 \
#    --user jeffl@vt.edu \
#    --pass <pass> \
#    --out-pref fss_inputs/graphspace/graphs/overview \
#    --name-postfix=-test$1 \
#
#    """
#echo $cmd
#$cmd
#
##    --simplified-terms-file  \
##    --edge-evidence-file fss_inputs/networks/stringv11/400/9606-uniprot-links-full-v11-evidence.tsv.gz \
##    --node-list-file fss_inputs/graphspace/gene_lists/top-cilium-assembly.txt \
##    --drug-list-file fss_inputs/graphspace/gene_lists/cilium-assembly-drugs.txt \
##    --drug-targets-only \
##    -T GO:0060271 \
##    --alg genemaniaplus \
##    --graph-attr-file fss_inputs/graphspace/gene_lists/cilium-assembly-styles.txt \
##    --apply-layout GM+-sarscov2-cilium-asembly-2 \
##    --layout-name layout4 \
