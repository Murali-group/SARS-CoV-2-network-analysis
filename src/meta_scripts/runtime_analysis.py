import pstats
from pstats import SortKey
import sys
sys.path.insert(0,"/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/")
p = pstats.Stats('profile_diffusion_path_analysis.txt')
p.sort_stats(SortKey.CUMULATIVE).print_stats(100)