
cmd="""python src/graphspace/sars_cov2_post_to_gs.py \
    --config fss_inputs/config_files/string-tissuenet-wace2/string.yaml \
    --drug-id-mapping-file datasets/drug-targets/drugbank-v5.1.6/additional-data/drugbank-toxicity.tsv \
    --drug-targets-file datasets/drug-targets/drugbank-v5.1.6/prot-drug-itxs.tsv \
    --drug-target-info-file datasets/drug-targets/drugbank-v5.1.6/additional-data/drugbank-targets.tsv \
    --edge-evidence-file fss_inputs/networks/stringv11/400/9606-uniprot-links-full-v11-evidence.tsv.gz \
    --enriched-terms-file outputs/enrichment/combined-krogan-0_01/GM+/string-k332-BP.csv \
    --node-list-file fss_inputs/graphspace/gene_lists/top-cilium-assembly.txt \
    --drug-list-file fss_inputs/graphspace/gene_lists/cilium-assembly-drugs.txt \
    --drug-targets-only \
    -T GO:0060271 \
    --alg genemaniaplus \
    --parent-nodes \
    --edge-weight-cutoff 900 \
    --user jeffl@vt.edu \
    --pass <password> \
    --graph-attr-file fss_inputs/graphspace/gene_lists/cilium-assembly-styles.txt \
    --out-pref fss_inputs/graphspace/graphs/protein-folding-in-er \
    --name-postfix=-string900-4 \
    --apply-layout GM+-sarscov2-cilium-asembly-2 \
    --layout-name layout4 \

    """
echo $cmd
$cmd

# leave this off while testing
    #--drug-list-file fss_inputs/graphspace/gene_lists/protein-folding-in-er-drugs.txt \
