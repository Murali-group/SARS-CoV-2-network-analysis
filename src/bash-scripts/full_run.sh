#!/bin/bash
conda activate sarscov2-net-new

python src/FastSinkSource/run_eval_algs.py --config $1 --num-pred-to-write -1
python src/scripts/prediction/alg_parameter_selection.py --config $1
python src/scripts/diffusion/effective_diffusion_node_path.py --force-contr --balancing-alpha-only --config $1 --pos-k
python src/scripts/diffusion/path_effective_diffusion_pathtype_supersink.py --balancing-alpha-only --config $1 --pos-k
python src/scripts/diffusion/plot_path_based_effective_diffusion_eppstein_balancing_alpha.py --balancing-alpha-only --config $1 --pos-k
python src/scripts/diffusion/compare_across_networks_ed_types_balancing_alpha.py --config $1 --pos-k
python src/scripts/betweenness/betweenness_src_spec.py  --balancing-alpha-only --config $1 # default --pos-k
python src/scripts/betweenness/preds_src_spec.py --balancing-alpha-only --config $1  #default --pos-k

