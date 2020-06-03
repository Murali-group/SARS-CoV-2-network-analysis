#date="2020-05-11"
date="2020-06-03"
# Make two versions of the predictions table: 

# First with only the nodes with a pval < 0.05
python src/write_pred_table.py \
    --config fss_inputs/config_files/string-tissuenet-wace2/string.yaml  \
    --alg genemaniaplus \
    --alg svm  \
    --id-mapping-file datasets/mappings/human/uniprot-reviewed-status.tab.gz \
    --out-pref outputs/stats/${date}/string-pred-p0_05-wdrug-  \
    --sample-neg-examples-factor 5.0 \
    --num-pred-to-write -1 \
    --stat-sig-cutoff 0.05 \
    --apply-cutoff \
    --drug-target-info-file datasets/drug-targets/drugbank-v5.1.6/additional-data/drugbank-targets.tsv \
    --drug-info-file datasets/drug-targets/drugbank-v5.1.6/additional-data/drugbank-toxicity.tsv \
    --prot-drug-targets \
    --go-pos-neg-table-file datasets/go//2020-05/pos-neg/isa-partof-pos-neg-cc-50.tsv.gz -T GO:0005886 -T GO:0005783

python src/closest_viral_prots.py \
    --config fss_inputs/config_files/string-tissuenet-wace2/string.yaml   \
    --pred-table outputs/stats/${date}/string-pred-p0_05-wdrug-genemaniaplus-svm.tsv


# Second with all nodes
python src/write_pred_table.py \
    --config fss_inputs/config_files/string-tissuenet-wace2/string.yaml  \
    --alg genemaniaplus \
    --alg svm  \
    --id-mapping-file datasets/mappings/human/uniprot-reviewed-status.tab.gz \
    --out-pref outputs/stats/${date}/string-pred-wdrug-  \
    --sample-neg-examples-factor 5.0 \
    --num-pred-to-write -1 \
    --stat-sig-cutoff 0.05 \
    --drug-target-info-file datasets/drug-targets/drugbank-v5.1.6/additional-data/drugbank-targets.tsv \
    --drug-info-file datasets/drug-targets/drugbank-v5.1.6/additional-data/drugbank-toxicity.tsv \
    --prot-drug-targets \
    --go-pos-neg-table-file datasets/go//2020-05/pos-neg/isa-partof-pos-neg-cc-50.tsv.gz -T GO:0005886 -T GO:0005783

python src/closest_viral_prots.py \
    --config fss_inputs/config_files/string-tissuenet-wace2/string.yaml   \
    --pred-table outputs/stats/${date}/string-pred-wdrug-genemaniaplus-svm.tsv
