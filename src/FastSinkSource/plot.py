
import os
import sys
import src.plot.plot_utils as plot_utils
import src.plot.combine_plots as combine_plots


# had to setup the scripts this way because Python's relative imports were giving me problems

if __name__ == "__main__":
    config_map, kwargs = plot_utils.parse_args()

    if config_map.get('combine_plots'):
        combine_plots.main(config_map, **kwargs)
    else:
        plot_utils.main(config_map, **kwargs)
