# this is a small set of commands to copy the prediction files from jeff's directory to another

jeff_dir="/data/jeff-law/projects/2020-03-covid-19/SARS-CoV-2-network-analysis"
net_version="stringv11/400"
exp_name="2020-03-sarscov2-human-ppi-ace2"
out_dir="outputs/networks/$net_version/$exp_name"
stat_sig_dir="outputs/viz/networks/$net_version/$exp_name"

# first copy the config file
config_dir="fss_inputs/config_files/string-tissuenet-wace2/"
mkdir -p $config_dir
echo "cp $jeff_dir/$config_dir/*.yaml  $config_dir/"
cp $jeff_dir/$config_dir/*.yaml  $config_dir/

# now copy the prediction and statistical significance files
for alg in genemaniaplus svm; do
    mkdir -p $out_dir/$alg
    echo "cp  $jeff_dir/$out_dir/$alg/pred-scores*.txt $out_dir/$alg"
    cp  $jeff_dir/$out_dir/$alg/pred-scores*.txt $out_dir/$alg

    mkdir -p $stat_sig_dir/$alg/
    echo "cp $jeff_dir/$stat_sig_dir/$alg/*.tsv $stat_sig_dir/$alg"
    cp  $jeff_dir/$stat_sig_dir/$alg/*.* $stat_sig_dir/$alg
done

