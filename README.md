# SARS-CoV-2-network-analysis
Analysis of SARS-CoV-2 molecular networks

## Getting Started
Required packages: networkx, numpy, scipy, pandas, sklearn, pyyaml, tqdm

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

The YAML config file contains the list of datasets to download and is self-documented. Currently the following types of datasets are supported:
  - networks

To add more datasets, copy one of the existing datasets and modify the fields accordingly. If your dataset is not yet supported, add to an existing issue ([#5](https://github.com/Murali-group/SARS-CoV-2-network-analysis/issues/5)), or make an issue and we'll try to add it as soon as we can. 

## Run the FastSinkSource Pipeline
  Coming soon...
### Generate predictions
### Cross Validaiton
