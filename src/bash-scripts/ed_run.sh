#!/bin/bash
conda activate sarscov2-net-new
python src/scripts/diffusion/effective_diffusion_node_path.py --force-contr --balancing-alpha-only   --config $1
python src/scripts/diffusion/path_effective_diffusion_pathtype_supersink.py --balancing-alpha-only  --config $1 --n $2
python src/scripts/diffusion/plot_path_based_effective_diffusion_eppstein_balancing_alpha.py --balancing-alpha-only  --config $1
