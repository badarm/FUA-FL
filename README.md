# Fairness and Utility Enhancing Agnostic Federated Learning Framework
Federated learning (FL) is an emerging communication-efficient and collaborative learning paradigm of Machine Learning (ML) with privacy guarantees. As these advancements unfold, adapting FL for fairness-aware learning becomes crucial.  In this context, we propose a pre-processing Fairness and Utility (balanced accuracy) enhancing Agnostic Federated Learning framework (Fed-FUEL) that mitigates discrimination embedded in the non-independent identically distributed (non-IID) data while improving the utility of the framework. We contribute a novel adaptive data manipulation method that mitigates discrimination embedded in the data at client side during optimization, resulting in an optimized and fair centralized server. This pre-processing approach, by design, abstracts the model architecture from the equation, offering a significant advantage in the federated environment. This abstraction not only facilitates a broader application across diverse model architectures without necessitating modifications but also sidesteps the potential complexities and inefficiencies associated with model-specific in-processing methods. Extensive experiments with a range of publicly available datasets demonstrate that our method outperforms the competing baselines in terms of both discrimination mitigation and predictive performance.
## The datsets used in this project
* [Adult Census](https://archive.ics.uci.edu/dataset/2/adult)
* [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* [Default](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
* [Law School](https://github.com/iosifidisvasileios/FABBOO/blob/master/Data/law_dataset.arff)

## Code
### Dataset Processing Scripts

The `datasets` directory contains all the datasets used in this project. Below is a description of python scripts written to process each dataset:

- `load_data.py`: Utility script for loading all the datasets.
- `load_adult_data.py`: Script for pre-processing and loading different distributions of Adult dataset.
- `load_bank_data.py`: Script for pre-processing and loading different distributions of Bank dataset.
- `load_default_data.py`: Script for pre-processing and loading different distributions of Default dataset.
- `load_law_data.py`: Script for pre-processing and loading different distributions of Law School dataset.

### Utility Scripts
- `find_potential_outcomes_utilities.py`: Script for finding potential outcomes.
- `find_disc_score.py`: Script for computing discriminations scores (statistical parity, equal opportuinity, FACE).
- 'test_local_and_server': Script for testing local learners and global server.


### Fed-FUEL main scripts
The following scripts constitute the complete methodology of Fed-FUEL
- `fed_fuel-main.py`: Main script for the 'Fed-FUEL' framework that orchestrates the fairness aware federated learning process on different datasets.
- `utilities.py`: The script contains functions related to the 'SOTE' and 'SDTE' algorithms for adaptive data augmnetation for discrimination mitigation.

## Prerequisites

Before running the script, ensure you have the following Python libraries installed:

- numpy
- pandas
- sklearn
- scipy
- matplotlib
- tensorflow
- pytorch
- scikit-multiflow
