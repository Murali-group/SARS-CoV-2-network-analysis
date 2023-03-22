#!/bin/bash

python src/FastSinkSource/run_eval_algs.py --config $1 --num-pred-to-write -1

python src/scripts/diffusion/effective_diffusion_node_path.py --force-contr --balancing-alpha-only   --config $1
python src/scripts/diffusion/path_effective_diffusion_pathtype_supersink.py --balancing-alpha-only  --config $1
python src/scripts/diffusion/plot_path_based_effective_diffusion_eppstein_balancing_alpha.py --balancing-alpha-only  --config $1
