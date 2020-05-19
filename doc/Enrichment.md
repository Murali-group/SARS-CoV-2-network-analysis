

To test the enrichment of the Krogan nodes:
```
python src/Enrichment/enrichment.py \
    --config fss_inputs/config_files/stringv11/400-nf5-nr100.yaml \
    --prot-list-file fss_inputs/pos-neg/2020-03-sarscov2-human-ppi/2020-03-24-sarscov2-human-ppi.txt \
    --prot-universe-file fss_inputs/networks/stringv11/400/sparse-nets/c400-node-ids.txt \
    --out-pref outputs/enrichment/krogan/p1_0 \
    --pval-cutoff 1.0 \
    --qval-cutoff 1.0 \
    --add-prot-list-to-prot-universe  \
    --force-run
```

To test the enrichment of the FSS algorithm predictions, and compare to the krogan term enrichment results generated above:
```
python src/Enrichment/fss_enrichment.py \
    --config fss_inputs/config_files/string-tissuenet-wace2/string.yaml \
    --stat-sig-cutoff 0.05  \
    --alg svm \
    --k-to-test 332  \
    --compare-krogan-terms outputs/enrichment/krogan/p1_0/
```

> Caution: the `--file-per-alg` option currently doesn't work with the `--compare-krogan-terms` option.
