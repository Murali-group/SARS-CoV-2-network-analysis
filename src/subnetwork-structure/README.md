
Some code to visualize and calculate simple statistics on the Krogan human proteins and drugs:

```
usage: run_subnetwork_stats_viz.py [-h] [--gs] [--username USERNAME] [--password PASSWORD] [--krogan-subgraph] [--drugs]

viz and analysis some simple network statistics about the Krogan nodes, the PPIs, and the drug-protein interactions.

optional arguments:
  -h, --help           show this help message and exit
  --gs                 Make GraphSpace Graphs.
  --username USERNAME  GraphSpace username. Required if --gs option is specified.
  --password PASSWORD  GraphSpace password. Required if --gs option is specified.
  --krogan-subgraph    Print and plot statistics about the Krogan induced subgraph.
  --drugs              Run PathLinker on the drug-ppi networks.
  ```

## Dependencies

Requires [`GraphSpace` python module](http://manual.graphspace.org/projects/graphspace-python/en/latest/) and `PathLinker` with modifications to make it compatible with `networkx v2`. See [Anna's forked Pathlinker directory](https://github.com/annaritz/PathLinker) (this should be verified and modified in the Murali group repo).

## Example Runs

- Get induced subgraph of the 332 Krogan human proteins in the TissueNet Lung and STRING networks. Print simple statistics and plot the degree distribution of these induced subnetworks.  Post these graphs to GraphSpace.

```
python3 run_subnetwork_stats_viz.py --krogan-subgraph --gs --username [USERNAME] --password [PASSWORD]
```

- Do same as above but don't post to GS.

```
python3 run_subnetwork_stats_viz.py --krogan-subgraph
```

- Get the induced subgraph of Krogan proteins and drugs and visualize (commented out).

- Use PathLinker to compute many short paths from the Krogan proteins to drugs.  Post these graphs to GraphSpace.

```
python3 run_subnetwork_stats_viz.py --drugs --gs --username [USERNAME] --password [PASSWORD]
```

- Get induced subgraph of krogan & drug nodes, along with neighbors that link krogan and drug nodes (path len at most 3).

```
python3 run_subnetwork_stats_viz.py --drugs_direct --gs --username [USERNAME] --password [PASSWORD]
```