#!/bin/bash
conda activate sarscov2-net-new
python src/FastSinkSource/run_eval_algs.py --config $1 --num-pred-to-write -1 --forcenet --forcealg
python src/scripts/prediction/alg_parameter_selection.py --config $1
