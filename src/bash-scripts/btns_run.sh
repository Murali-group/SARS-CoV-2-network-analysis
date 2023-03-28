#!/bin/bash
conda activate sarscov2-net-new
python src/scripts/betweenness/betweenness_src_spec.py  --balancing-alpha-only --config $1 --force-run
#python src/scripts/betweenness/preds_src_spec.py --balancing-alpha-only --config $1
