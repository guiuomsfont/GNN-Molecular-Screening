
GNN Drug Target Interaction
=================

A repository of scripts and utilities built by Grupo AIA and Guiu Oms Font in 2022 for the purpose of predicting the interaction between different chemicals and a target protein. It has also been used as Final Degree Project.

The code is derived from the article Predicting Drug-target Interaction Using a Novel Graph Neural Network with 3D Structure-embedded Graph Representation. Starting from this, improvements are applied throughout the execution pipeline, optimizing the preprocessing of the data and evaluating different possible options, redesigning part of the structure of the network, preparing the code and restructuring it so that it is understandable and usable in a way quickly in real applications and packaged in such a way that it acquires robustness and manageability.

Normal use of this package starts with running docking via Smina a fork of AutoDock Vina (run_docking_receptor). Once the files are obtained, rf-score-4 and a script to cut the binding pocket (prepare_data.sh) must be executed. Next we already have the data ready to enter the GNN and we only have to create the train and test partitions with a script (create_keys.py). Finally we must use the train_test scripts to train or predict with the neural network.

***(c) 2022 Grupo AIA***  
**omsg@aia.es**



