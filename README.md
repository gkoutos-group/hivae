# hivae2

This repository contains a modular wrapper of the "Heterogeneous Incomplete Variational Autoencoder model (HI-VAE)".  The previous implementation branch (v1) has been used in: 

Karwath et al., Redefining β-blocker response in heart failure patients with sinus rhythm and atrial fibrillation: a machine learning cluster analysis, Lancet, 2021 [paper](https://doi.org/10.1016/S0140-6736(21)01638-X). 

This modular wrapper was written by Andreas Karwath (a.karwath@bham.ac.uk) and Fathy Shalaby.

The original coding was done by Alfredo Nazabal (anazabal@turing.ac.uk) et al. written in Python and details can be found in this [paper](https://doi.org/10.1016/j.patcog.2020.107501). 

This is an extenstion of implementations as easy to use Python library, upgraded for tensorflow2.

Please cite both papers if you should use this code/library for your own research.


## Examples

See examples directory for usage


## Files description

*(outdated) **HIVAE.py**: The main script of the library, it needs to imported to work with the library and is connected to all the other scripts.
* **loglik_ models_ missing_normalize.py**: In this file, the different likelihood models for the different types of variables considered (real, positive, count, categorical and ordinal) are included.
* **model_ HIVAE_inputDropout.py**: Contains the HI-VAE with input dropout encoder model.
* **model_ HIVAE_factorized.py**: Contains the HI-VAE with factorized encoder model

## Contact

* **For questions regarding algorithm --> Alfredo Nazabal**: anazabal@turing.ac.uk
* **For bugs or suggestion regarding this code --> Andreas Karwath**: a.karwath@bham.ac.uk

## Comments

This version required tf2. For apple silcone users, please follow : https://developer.apple.com/metal/tensorflow-plugin/


## Comments on general_example.py (might be outdated!)


main_directory: where is the project folder

dataset_name: the name of the database (if you want)

types_list_d: a dictionary where the key is the dataset name, which contains a list with tuples that indicates the column names, types, the number of dimensions and classes 

types:

•	count: real values

•	cat: categorical 0 or 1

•	pos: positive real values

•	ordinal: ordinal number

number of dimensions:

•	number of possibilities in the categorical variables or 1 in numerical

number of classes:

•	number of options (same of number of dimensions for categorical variables)

dataset_path: this is the folder of the csv files

results_path: the output folder for results

network_path: where the models are going to be stored

types_list: the specific type for the dataset you are going to use
data_file: the full dataset
train_file/ test_file: if the dataset was already splitted

train_data/test_data: pandas dataframes

dim_y: the depth of the network

dim_s/dim_z: dimensions of the embedding
