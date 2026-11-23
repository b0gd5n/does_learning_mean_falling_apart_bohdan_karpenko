# Integrat

Code for the project "Does learning mean falling apart? The behaviour of φₛ and Φ during training with gradient descent" (Bohdan Karpenko).

An Integrat is a recurrent network made of stochastic binary nodes. It is trained with gradient descent, and its integration according to Integrated Information Theory (IIT 4.0) is computed exactly with the PyPhi library. The project compares ring and random initialisation on cores of 2, 3 and 4 nodes, seven tasks and five delays, 840 networks in total.

## Files

| File | What it does |
|---|---|
| `integrats.py` | The `Integrat` class, forward and backward pass, AdamW optimiser, computation of φₛ and Φ with PyPhi |
| `research.py` | All experiments. Atlas of untrained cores, training of 840 networks and weight shuffling. Writes CSV files to the `data` folder, one set of files per task |
| `make_plots.py` | Draws the plots from the CSV files in `data` and saves them to the `plots` folder in English and German |
| `statistics.py` | Computes all statistics of the paper from the CSV files in `data`, namely the Wilcoxon tests and the Pearson and Spearman correlations |
| `requirements.txt` | Required libraries |

## Installation

```
python -m pip install -r requirements.txt
```

PyPhi is installed from the `feature/iit-4.0` branch, pinned to commit `b78d0e342`, because the regular PyPhi release does not include IIT 4.0.

## Running

```
python research.py atlas
python research.py and
python research.py or
python research.py xor
python research.py one_back_xor
python research.py replay
python research.py replay_twice
python research.py noise
python make_plots.py
python statistics.py
```

`research.py atlas` computes the untrained cores, every other command trains all networks of one task. The tasks are independent, so they can run at the same time in separate terminals. Training runs for about a day, because φₛ and Φ are computed exactly for every state of the core. Before running it again, copy the old files from `data` somewhere else, otherwise they will be overwritten. `statistics.py` prints all p values and correlation coefficients of the paper.

## Results

The `data` folder contains

- `atlas.csv` with φₛ and Φ of untrained cores of 2, 3, 4 and 5 nodes
- `training_accuracy_<task>.csv` with the accuracy of every network before training, every 1024 epochs and after each of the 8 weight shuffles
- `training_integration_<task>.csv` with φₛ and Φ of the whole core for every state at the same moments
- `training_weights_<task>.csv` with the weights of every network at the end of training

The `plots` folder contains the plots in the subfolders `en` and `de` and the table `summary.csv` with the means before training, after training and after shuffling.

The paper is in the `article` folder in English and German.

## Reproducibility

All random numbers are set by the generator seeds 0, 1, 2 and 3. Test sequences come from a separate generator with seed 12345, so measuring accuracy does not affect training. With the same library versions the results should match the paper.

## Use of AI

The backward pass and the scripts for training, plots and statistics were written with the help of Claude (Anthropic). All computations were run on the author's computer.

## License

GPL-3.0, see `LICENSE`.
