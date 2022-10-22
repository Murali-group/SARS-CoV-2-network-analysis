# this script is for running the pipeline

from src import main


if __name__ == "__main__":
    config_map, kwargs = main.parse_args()
    main.run(config_map, **kwargs)
