#simplified_terms_file="outputs/enrichment/networks/stringv11/400/2020-03-sarscov2-human-ppi-ace2/GM+/pred-scores-a0_01-tol1e-05-filtered-p0_05/enrich-BP-simplified0.5.csv"
#simplified_terms_file="outputs/enrichment/simplified/terms.txt"
simplified_terms_file="outputs/enrichment/simplified/more-terms.txt"
#enriched_terms_file="outputs/enrichment/combined-krogan-0_01/GM+/string-k332-BP.csv"
#for alg in genemaniaplus; do 
enriched_terms_file="outputs/enrichment/combined-krogan-0_01/SVM/string-k332-BP.csv"
for alg in svm; do 
    for term in `cut -f 1 -d ',' $simplified_terms_file | tail -n +2 | sed "s/\"//g"`; do
#    for term in "GO:0048208"; do
cmd="""python src/graphspace/sars_cov2_post_to_gs.py \
    --config fss_inputs/config_files/string-tissuenet-wace2/string.yaml  \
    --drug-id-mapping-file datasets/drug-targets/drugbank-v5.1.6/additional-data/drugbank-toxicity.tsv \
    --drug-targets-file datasets/drug-targets/drugbank-v5.1.6/prot-drug-itxs.tsv   \
    --drug-target-info-file datasets/drug-targets/drugbank-v5.1.6/additional-data/drugbank-targets.tsv \
    --drug-list-file fss_inputs/graphspace/drug_lists/accepted_investigational.txt \
    --drug-targets-only \
    --enriched-terms-file $enriched_terms_file -T $term \
    --edge-evidence-file fss_inputs/networks/stringv11/400/9606-uniprot-links-full-v11-evidence.tsv.gz \
    --alg $alg   \
    --parent-nodes \
    --name-postfix=-s0.5   \
    --edge-weight-cutoff 700 \
    --user jeffl@vt.edu \
    --pass f1fan     \
    --group SARS-CoV-2-network-testing \
    --tag simplified
"""
        echo $cmd
        $cmd
        #break
    done
done


# also keep track of the other terms I post here:
# GO:0006506 (GPI anchor biosynthetic process)
# GO:0036503 (ERAD pathway)
# GO:0048208 (COPII vesicle coating)
