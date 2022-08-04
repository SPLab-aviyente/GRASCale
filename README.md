# GRASCale

Python implementation of simultaneous GRAph Signal Clustering And graph LEarning (GRASCale) algorithm presented in [1]. 

## Installation
Once you download the repo, go to the repo directory and start a terminal. First, create an 
environment and then install the required packages listed in `requirements.txt`. This can be done as
follows for conda:
```sh
conda create -n grascale python=3.10
conda activate grascale
```
This will create a conda environment. To install the requirements:
```sh
pip install -r requirements.txt
```

## Usage 
Please see `experiment1.py` under scripts folder for an illustration about how to use the code. 
The script includes data generation process of experiment 1 from the paper and shows how to run 
GRASCale on the generated simulated data. To run the script first start a terminal on the repo 
directory, then:
```sh
conda activate grascale
python scripts/experiment1.py
```
It will print F1 score, NMI and density of the learned graphs associated with clusters. 
The script for the experiment 2 of the paper will also be published soon. 

### References
  - [1] [Karaaslanli, Abdullah, and Selin Aviyente. "Simultaneous Graph Signal Clustering and Graph Learning." International Conference on Machine Learning. PMLR, 2022.](https://proceedings.mlr.press/v162/karaaslanli22a.html)
