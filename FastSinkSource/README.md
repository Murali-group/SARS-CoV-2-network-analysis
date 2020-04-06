# FastSinkSource
This is the main repository for the paper "Accurate and Efficient Gene Function Prediction using a Multi-Bacterial Network".

## Installation
These scripts requires Python 3 due to the use of obonet to build the GO DAG.

Required packages: `networkx`, `numpy`, `scipy`, `pandas`, `sklearn`, `obonet`, `pyyaml`, `tqdm`

To install the required packages:
```
pip3 install -r requirements.txt
```

Optional: use a virtual environment
```
virtualenv -p /usr/bin/python3 py3env
source py3env/bin/activate
pip install -r requirements.txt
```

## Cite
If you use FastSinkSource or other methods in this package, please cite:

Jeffrey Law, Shiv D. Kale, and T. M. Murali. [Accurate and Efficient Gene Function Prediction using a Multi-Bacterial Network](https://doi.org/10.1101/646687), _bioRxiv_ (2019). doi.org/10.1101/646687
