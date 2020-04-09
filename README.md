# SARS-CoV-2-network-analysis
Analysis of SARS-CoV-2 molecular networks

## Getting Started
Required packages: networkx, numpy, scipy, pandas, sklearn, pyyaml, wget, tqdm

To install the required packages:

```
pip3 install --user -r requirements.txt
```
  
Optional: create a virtual environment with anaconda
```
conda create -n sarscov2-net python=3.7
conda activate sarscov2-net
pip install -r requirements.txt
```

## Download Datasets
The SARS-CoV-2 - Human PPI "Krogan" dataset is available in the [datasets/protein-networks](https://github.com/Murali-group/SARS-CoV-2-network-analysis/tree/master/datasets/protein-networks) folder. 

To download additional datasets and map various namespaces to UniProt IDs, use the following command
```
python src/masterscript.py --config config-files/master-config.yaml --download-only
```

The YAML config file contains the list of datasets to download and is self-documented. The following types of datasets are supported:
  - Networks
  - Gene sets
  - Drug targets

To download additional datasets, copy one of the existing dataset sections in the config file and modify the fields accordingly. If your dataset is not yet supported, add to an existing issue ([#5](https://github.com/Murali-group/SARS-CoV-2-network-analysis/issues/5)), or make an issue and we'll try to add it as soon as we can. 

## Run the FastSinkSource Pipeline
This master script will generate a config file specific to the [FastSinkSource pipeline](https://github.com/jlaw9/FastSinkSource/tree/no-ontology), which will then be used to generate predictions, and/or run cross validation.

> Note that the FastSinkSource code was added as a [git-subrepo](https://github.com/ingydotnet/git-subrepo), so to make changes to that code, please commit them to that repository directly, and then pull them with `git subrepo pull src/FastSinkSource/`, or follow the suggestions in the git-subrepo documentation.

### Generate predictions
The script will automatically generate predictions from each of the given methods in the config file. The default number of predictions stored is 10. To write more, either set the `num_pred_to_write` or `fator_pred_to_write` flags under `fastsinksource_pipeline_settings -> eval_settings`, or, after the config file is generated, call the `run_eval_algs.py` script directly and add either the `--num-pred-to-write` or `--factor-pred-to-write` options (see `python src/FastSinkSource/run_eval_algs.py --help`)

Example 1, with the appropriate flags set in the master config file:
```
python src/masterscript.py --config config-files/master-config.yaml 
```

Example 2:
```
python src/FastSinkSource/run_eval_algs.py  --config fss_inputs/config_files/stringv11/400-cv5-nf1.yaml --num-pred-to-write -1
```

#### TODO Compare overlap of top predictions with various gene sets ([#6](https://github.com/Murali-group/SARS-CoV-2-network-analysis/issues/6))
#### TODO Post sub-network of the top predictions and their scores to GraphSpace
### Cross Validation
Similar to the previous section, the options to run cross validation can either be set in the config file under `fastsinksource_pipeline_settings -> eval_settings`, or passed directly to `run_eval_algs.py`. The relevant options are below. See `python src/FastSinkSource/run_eval_algs.py --help` for more details.
  - `cross_validation_folds`
    - Number of folds to use for cross validation. Specifying this parameter will also run CV
  - `sample_neg_examples_factor`
    - ratio of negatives to positives to randomly sample from the nodes of the given network
  - `num_reps`
    - Number of times to repeat CV and the sampling of negative examples
  - `cv_seed`
    - The seed to use when generating the CV splits. 

After CV has finished, to visualize the results, use the `plot.py` script (TODO add to `masterscript.py`). For example:
```
python FastSinkSource/plot.py --config fss_inputs/config_files/stringv11/400-cv5-nf1.yaml --box --measure fmax
```

I used the jupyter notebook `src/jupyter-notebooks/plot_net_collections.ipynb` to visualize the CV results (e.g., Fmax) on the large collections of TissueNet v2 (TODO make a script for it).
