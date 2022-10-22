# annotation-prediction
Pipeline for running and evaluating algorithms for gene/protein annotation prediction.
This `no-ontology` branch is for cases where the "annotations" are simply node labels
(e.g., human proteins that directly interact with SARS-CoV-2 proteins)
that do not belong to a specific ontology (e.g., Gene Ontology, Human Phenotype Ontology)


## Installation

- Required Python packages: `networkx`, `numpy`, `scipy`, `pandas`, `sklearn`, `pyyaml`, `tqdm`, `rpy2`
- Required R packages: PPROC

To install the required packages:
```
conda create -n ann-pred python=3.7 r=3.6 --file requirements.txt
conda activate ann-pred
```
To install the R packages:
```
R -e "install.packages('https://cran.r-project.org/src/contrib/PRROC_1.3.1.tar.gz', type = 'source')"
```

> If you are unable to install the the R package for computing the AUPRC and AUROC, 
> the code will use sklearn instead, which is not as accurate in some cases.

## Usage 
### Generate predictions
The script will automatically generate predictions from each of the given methods with `should_run: [True]` in the config file. The default number of predictions stored is 10. To write more, use either the `--num-pred-to-write` or `--factor-pred-to-write options` (see python run_eval_algs.py --help). For example:
```
python run_eval_algs.py  --config config.yaml --num-pred-to-write -1
```

### Cross Validation
The relevant options are below. See `python run_eval_algs.py --help` for more details.
  - `cross_validation_folds`
    - Number of folds to use for cross validation. Specifying this parameter will also run CV
  - `cv_seed`
    - Can be used to specify the seed to use when generating the CV splits. 
    
Example:
```
python run_eval_algs.py  --config config.yaml --cross-validation-folds 5 --only-eval
```

#### Sample negative examples
Negative examples are used by some algorithms, and they are used to measure precision and recall during CV.

If no negative examples are given, they can be set with these options:
  - `sample_neg_examples_factor`
    - ratio of negatives to positives to sample uniformly at random without replacement from the nodes of the given network
  - `num_reps`
    - Number of times to repeat CV and the sampling of negative examples

#### Plot
After CV has finished, to visualize the results, use the `plot.py` script. For example:
```
python plot.py --config config.yaml --box --measure fmax
```

## Cite
If you use FastSinkSource or other methods in this package, please cite:

Jeffrey N. Law, Shiv D. Kale, and T. M. Murali. [Accurate and Efficient Gene Function Prediction using a Multi-Bacterial Network](https://doi.org/10.1093/bioinformatics/btaa885), _Bioinformatics_ (2020). 
