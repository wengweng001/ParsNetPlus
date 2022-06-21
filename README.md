# ParsNetPlus Pytorch
Online Weakly Supervised Learning Approach for Quality Monitoring of Complex Manufacturing Process, Complexity, 2021

derived from "Weakly Supervised Deep Learning Approach in Streaming Environments. IEEE BigData 2019: 1195-1202"


# Requirements
The current version of the code has been tested with:
* `pytorch 1.9.0`
* `torchvision 0.10.0`

# Running the experiment
The link between individual experiments and scripts are listed as follow:  
Sporadic access current batch prediction -- `CurrentBatchPredict.py`  
Infinite delay current batch prediction -- `InfiniteDelay.py`

Next batch prediction experiments can be simulated with replacing the dataloader with respective scripts from `dataprep_weakly.load_sensor(labeled_proportion, batchSize, next=False)` to `dataprep_weakly.load_sensor(labeled_proportion, batchSize, next=True)`.

A comparison of classification performance on ParsNet++ included in the paper can be run with `plot-cur.ipynb` and `plot-next.ipynb` (Figure 4).

# SIMTech Injection Molding Dataset
The dataset we used in this project came from the Singapore Institute of Manufacturing Technology (SIMTech).
