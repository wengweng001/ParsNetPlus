# ParsNetPlus Pytorch
Online Weakly Supervised Learning Approach for Quality Monitoring of Complex Manufacturing Process, Complexity, 2021

derived from "Weakly Supervised Deep Learning Approach in Streaming Environments. IEEE BigData 2019: 1195-1202"


# Requirements
The current version of the code has been tested with:
* `pytorch 1.3.1`
* `torchvision 0.4.2`

# Running the experiment
All experiments with injection molding dataset can be run with `NextBatchPredict.py`, `CurrentBatchPredict.py`, `InfiniteDelayNext.py`, `InfiniteDelay.py`, and `NextBatchPredict_proportion.py`. The link between individual experiments and scripts are listed as follow:
Sporadic access next batch prediction -- `NextBatchPredict.py`
Sporadic access current batch prediction -- `CurrentBatchPredict.py`
Infinite delay next batch prediction -- `InfiniteDelayNext.py`
Infinite delay current batch prediction -- `InfiniteDelay.py`
Effect of different label proportion -- `NextBatchPredict_proportion.py`
A comparison of classification performance on ParsNet++ included in the paper can be run with `plot-cur.ipynb` and `plot-next.ipynb` (Figure 4).
