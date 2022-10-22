# Params Comparisons
These config files contain the different parameter combinations we tested.

## deepNF
For deepNF, we have a different way of handling the STRING network. 
In their paper, they first process each of the individual channels, and then combine them in the autoencoder.
We accomplish that here by setting `string_nets: all`, which means all string channels will be passed to the method.

By running only deepNF with this dataset (i.e., '2020-03-sarscov2-human-ppi-ace2-all') and not the other dataset (i.e., '2020-03-sarscov2-human-ppi-ace2'),
we are able to have the plotting scripts combine the results into a single plot.

### command
```
python src/meta_scripts/start_jobs_baobab.py \
  --config ann_pred_inputs/config_files/stringv11/params-comparisons/params-testing-deepnf.yaml \
  --job-per-param \
  --alg deepnf \
  --qsub \
  --pass-to-script --only-eval -C 5 --early-prec 0.3
```
