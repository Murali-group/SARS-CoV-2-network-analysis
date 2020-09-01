#drug_targets_only="--drug-targets-only"
#simplified_terms_file="outputs/enrichment/networks/stringv11/400/2020-03-sarscov2-human-ppi-ace2/GM+/pred-scores-a0_01-tol1e-05-filtered-p0_05/enrich-BP-simplified0.5.csv"
#simplified_terms_file="outputs/enrichment/simplified/terms.txt"
#simplified_terms_file="outputs/enrichment/simplified/more-terms.txt"
#simplified_terms_file="outputs/enrichment/simplified/2020-06-03/string-k332-GO-BP-simplified-manual.tsv"
simplified_terms_w_layouts_file="outputs/enrichment/simplified/2020-06-03/layouts/string_k332_GO-BP_simplified.tsv"
#alg_name="RL"
alg_name="SVM"
enriched_terms_file="outputs/enrichment/combined-krogan-0_01/$alg_name/string-k332-BP.csv"
#for alg in genemaniaplus; do 
for alg in svm; do 
    #for term in `cut -f 1 $simplified_terms_file | tail -n +2 | sed "s/\"//g"`; do
    for term in `cut -f 1,3,6,7 $simplified_terms_w_layouts_file | tail -n +2 | grep "$alg_name" | sed "s/\t/|--term-color=/" | sed "s/\t/|--apply-layout=/" | sed "s/\t/|--layout-name=/"`; do
        term=$(echo "$term" | sed "s/|/ /g")
cmd="""python src/graphspace/sars_cov2_post_to_gs.py \
    --config fss_inputs/config_files/string-tissuenet-wace2/string.yaml  \
    --drug-id-mapping-file datasets/drug-targets/drugbank-v5.1.6/additional-data/drugbank-toxicity.tsv \
    --drug-targets-file datasets/drug-targets/drugbank-v5.1.6/prot-drug-itxs.tsv   \
    --drug-target-info-file datasets/drug-targets/drugbank-v5.1.6/additional-data/drugbank-targets.tsv \
    --drug-list-file fss_inputs/graphspace/drug_lists/accepted_investigational.txt \
    --enriched-terms-file $enriched_terms_file -T $term \
    --edge-evidence-file fss_inputs/networks/stringv11/400/9606-uniprot-links-full-v11-evidence.tsv.gz \
    $drug_targets_only \
    --alg $alg   \
    --k-to-test 332 \
    --parent-nodes \
    --name-postfix=-string700   \
    --edge-weight-cutoff 700 \
    --user jeffl@vt.edu \
    --pass <pass> \
    --group SARS-CoV-2-network-testing \
    --tag simplified --tag svm --tag covid19

"""
        echo $cmd
        $cmd
        #break
    done
done


# also keep track of the other terms I post here:
# GO:0006829 (zinc ion transport)
# GO:0006506 (GPI anchor biosynthetic process)
# GO:0036503 (ERAD pathway)
# GO:0048208 (COPII vesicle coating)
