# hivae

This repository contains the Modular reimplemenation of the Heterogeneous Incomplete Variational Autoencoder model (HI-VAE)written by Alfredo Nazabal and co-workers.
The package provided here is to a large part baseed on this implementation, but adheres to a more pythonic way, omitting the need for supplying parameters via I/O , as well as aligning the modelling more with sklearn.
It was written by A. Karwath (a.karwath@bham.ac.uk) and F. Shalaby.


The details of this model are included in this [paper](https://arxiv.org/abs/1807.03653). 

## Install

The package can be installed using pip:

```pip install hivae```


## Examples

Once checked out, there are a number of example datasets (Wine, Adult and Diabetes), which can be found in ./hivae/examples/data. To evaluate the package, please use ./hivae/examples/hivae_general_example.py. The example should give a general explaination of how to use the package. More details will folow.


## Files description

* **hivae.py**: The main script of the library, it needs to imported to work with the library and is connected to all the other scripts.
* **loglik_ models_ missing_normalize.py**: In this file, the different likelihood models for the different types of variables considered (real, positive, count, categorical and ordinal) are included.
* **model_ HIVAE_inputDropout.py**: Contains the HI-VAE with input dropout encoder model.
* **model_ HIVAE_factorized.py**: Contains the HI-VAE with factorized encoder model

## Contact

* **For questions regarding algorithm --> Alfredo Nazabal**: anazabal@turing.ac.uk

## More details regarding the hivae_general_example.py and use of the model (please note that this is under construction)

main_directory: project folder

dataset_name: the name of the database (required)

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

results_path: the output folder for results (currently not used)

network_path: where the models are going to be stored

types_list: the specific type for the dataset you are going to use
data_file: the full dataset
train_file/ test_file: if the dataset was already splitted

train_data/test_data: pandas dataframes

dim_y: the depth of the network

dim_s/dim_z: dimensions of the embedding
