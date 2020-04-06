# these are the proteins known to interact with SARS-COV-2
cut -f 2 ../../../datasets/protein-networks/2020-03-biorxiv-krogan-sars-cov-2-human-ppi.tsv | tail -n +2 > 2020-03-24-sarscov2-human-ppi.txt

# then to generate the pos-neg file:
#cd ../../../
#python src/setup_pos_neg_file.py \
#    --pos-examples-file inputs/pos-neg/2020-03-sarscov2-human-ppi/2020-03-24-sarscov2-human-ppi.txt \
#    --name 2020-03-sarscov2-human-ppi \
#    --prot-universe-file inputs/networks/stringv11/9606-stringv11-700-prots.txt \
#    --sample-neg-examples-factor 10 \
#    --out-file inputs/pos-neg/2020-03-sarscov2-human-ppi/pos-neg-sample10.txt

# to generate the "pos.txt (positives) file, I just cut the second column of 2020-03-24-sarscov2-human-ppi.txt 
# and add a column of 1s after it
