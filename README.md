# SARS-CoV-2-network-analysis
Analysis of SARS-CoV-2 molecular networks

## Getting Started
- Required Python packages: networkx, numpy, scipy, pandas, sklearn, pyyaml, rpy2, wget, tqdm, graphspace_python
- Required R packages: PPROC
- Recommended R packages: clusterProfiler, org.Hs.eg.db

We recommend using [Anaconda](https://www.anaconda.com/) for Python, especially to access the needed R packages. To setup your environment, use the following commands:

```
conda create -n sarscov2-net python=3.7 r=3.6
conda activate sarscov2-net
pip install -r requirements.txt
```
To install the R packages:
```
R -e "install.packages('https://cran.r-project.org/src/contrib/PRROC_1.3.1.tar.gz', type = 'source')"
conda install -c bioconda bioconductor-clusterprofiler
```

## Download Datasets
The SARS-CoV-2 - Human PPI "Krogan" dataset is available in the [datasets/protein-networks](https://github.com/Murali-group/SARS-CoV-2-network-analysis/tree/master/datasets/protein-networks) folder. 

To download additional datasets and map various namespaces to UniProt IDs, use the following command(s)
```
python src/masterscript.py --config config-files/master-config.yaml --download-only
python src/masterscript.py --config config-files/biogrid.yaml --download-only
python src/masterscript.py --config config-files/huri.yaml --download-only
```
More details of how biogrid was processed are in the [biogrid folder](https://github.com/Murali-group/SARS-CoV-2-network-analysis/tree/use_annotation_prediction/datasets/networks/biogrid).

The YAML config file contains the list of datasets to download and is self-documented. The following types of datasets are supported:
  - Networks
  - Gene sets
  - Drug targets

To download additional datasets, copy one of the existing dataset sections in the config file and modify the fields accordingly. If your dataset is not yet supported, add to an existing issue ([#5](https://github.com/Murali-group/SARS-CoV-2-network-analysis/issues/5)), or make an issue and we'll try to add it as soon as we can. 

## Run the annotation_prediction Pipeline
This master script will generate a config file specific to the [annotation_prediction pipeline](https://github.com/Murali-group/annotation_prediction/tree/no-ontology), which will then be used to generate predictions, and/or run cross validation.

> Note that the annotation_prediction code was added as a [git-subrepo](https://github.com/ingydotnet/git-subrepo), so to make changes to that code, please commit them to that repository directly, and then pull them with `git subrepo pull src/annotation_prediction/`, or follow the suggestions in the git-subrepo documentation.

### Generate predictions
The script will automatically generate predictions from each of the given methods in the config file. The default number of predictions stored is 10. To write more, either set the `num_pred_to_write` or `fator_pred_to_write` flags under `annotation_prediction_pipeline_settings -> eval_settings`, or, after the config file is generated, call the `run_eval_algs.py` script directly and add either the `--num-pred-to-write` or `--factor-pred-to-write` options (see `python src/annotation_prediction/run_eval_algs.py --help`)

Example 1, with the appropriate flags set in the master config file:
```
python src/masterscript.py --config config-files/master-config.yaml 
```

Example 2:
```
python src/annotation_prediction/run_eval_algs.py  \
  --config ann_pred_inputs/config_files/stringv11/400-nf5-nr100.yaml \
  --num-pred-to-write -1
```

#### Test for enrichment of top predictions with various gene sets ([#6](https://github.com/Murali-group/SARS-CoV-2-network-analysis/issues/6))
After the predictions have been generated, you can test for the enrichment of various genesets among the top _k_ predictions.

The following command will test for enrichment of the top 332 predictions of each algorithm, and on each dataset/network in the config file:
```
python src/Enrichment/fss_enrichment.py \
    --config ann_pred_inputs/config_files/stringv11/400-nf5-nr100.yaml \
    --k-to-test 332 --file-per-alg
```

TODO Currently only tests for enrichment of GO terms (BP, MF, CC).

To test for enrichment of any given list of genes (e.g., Krogan nodes), use the following type of command:
```
python src/Enrichment/enrichment.py \
  --prot-list-file ann_pred_inputs/pos-neg/2020-03-sarscov2-human-ppi/2020-03-24-sarscov2-human-ppi.txt \
  --prot-universe-file ann_pred_inputs/networks/stringv11/400/sparse-nets/c400-node-ids.txt \
  --out-pref outputs/enrichment/krogan \
  --add-prot-list-to-prot-universe
```

#### Post sub-network of the top predictions and their scores to GraphSpace
https://graphspace.org/ allows you to visualize and interact with the predictions made by the methods. In addition, you can share graphs with a group to easily allow others to do the same. 

Here's an example of how to post the top 5 predictions made by GM+ when using STRING, and the shortest paths from those to the virus proteins:
```
python src/graphspace/sars_cov2_post_to_gs.py \
  --config ann_pred_inputs/config_files/stringv11/400-nf5-nr100.yaml \
  --sarscov2-human-ppis datasets/protein-networks/2020-03-biorxiv-krogan-sars-cov-2-human-ppi.tsv \
  --user <email> --pass <password> \
  --k-to-test 5 --name-postfix=-test \
  --alg genemaniaplus \
  --parent-nodes
```

<!---
When making predictions for drugs, you can a
--->

### Cross Validation
Similar to the previous section, the options to run cross validation can either be set in the config file under `annotation_prediction_pipeline_settings -> eval_settings`, or passed directly to `run_eval_algs.py`. The relevant options are below. See `python src/annotation_prediction/run_eval_algs.py --help` for more details.
  - `cross_validation_folds`
    - Number of folds to use for cross validation. Specifying this parameter will also run CV
  - `sample_neg_examples_factor`
    - ratio of negatives to positives to randomly sample from the nodes of the given network
  - `num_reps`
    - Number of times to repeat CV and the sampling of negative examples
  - `cv_seed`
    - The seed to use when generating the CV splits. 

Example:
```
python src/annotation_prediction/run_eval_algs.py \
  --config ann_pred_inputs/config_files/stringv11/400-nf5-nr100.yaml \
  --cross-validation-folds 5
```

After CV has finished, to visualize the results, use the `plot.py` script (TODO add to `masterscript.py`). For example:
```
python annotation_prediction/plot.py --config ann_pred_inputs/config_files/stringv11/400-nf5-nr100.yaml --box --measure fmax
```

I used the jupyter notebook `src/jupyter-notebooks/plot_net_collections.ipynb` to visualize the CV results (e.g., Fmax) on the large collections of TissueNet v2 (TODO make a script for it).
