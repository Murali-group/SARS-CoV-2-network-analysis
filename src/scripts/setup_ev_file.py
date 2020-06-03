# The goal of the script is to convert the STRING file into an evidence-style file for posting to GS

import os, sys
import pandas as pd
import numpy as np
import gzip

string_file = "fss_inputs/networks/stringv11/400/9606-uniprot-links-full-v11.txt.gz"
out_file = string_file.replace('.txt.gz', '-evidence.tsv.gz')
#string_file = "fss_inputs/networks/stringv11/400/head.txt"
#out_file = "fss_inputs/networks/stringv11/400/head.txt.gz"
cutoff = 400

string_cols = ['protein1', 'protein2', 'neighborhood', 'neighborhood_transferred', 'fusion', 'cooccurence', 'homology', 'coexpression', 'coexpression_transferred', 'experiments', 'experiments_transferred', 'database', 'database_transferred', 'textmining', 'textmining_transferred', 'combined_score']
df = pd.read_csv(string_file, sep='\t', names=string_cols)
df.replace(0, np.nan, inplace=True)
df = df[df['combined_score'] > cutoff]
print(df.head())
df.drop(columns='combined_score', inplace=True)

df2 = pd.melt(
    df, id_vars=string_cols[:2], value_vars=string_cols[2:-1],
    var_name="string_channel", value_name="score").dropna(how='any')
print(df2.head())
print(len(df2))
print(df2['string_channel'].value_counts())

# now setup the table to match the "evidence file"
# the desired columns are: #uniprot_a  uniprot_b   directed    interaction_type    detection_method    publication_id  source
df2.columns = ['#uniprot_a', 'uniprot_b', 'interaction_type', 'detection_method']
df2.insert(2, 'directed', False)
df2['publication_id'] = np.nan
df2['source'] = "STRING"

print(df2.head())
print("writing %s" % (out_file))
df2.to_csv(out_file, sep='\t', compression='gzip', index=False)
