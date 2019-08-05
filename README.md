# hivae

This repository contains the Modular reimplemenation of the Heterogeneous Incomplete Variational Autoencoder model (HI-VAE)written by Alfredo Nazabal (anazabal@turing.ac.uk). It was written in Python, using Tensorflow.

The details of this model are included in this [paper](https://arxiv.org/abs/1807.03653). Please cite it if you use this code/library for your own research.

## Databse description

There are three different example datasets found in the library (Wine, Adult and Diabetes). Majority of the datasets( Wine and Adult) have each their own folder, containing:

* **data.csv**: the dataset
* **data_types.csv(NOT REQUIRED, LOOK AT THE EXAMPLE(working_example)**: a csv containing the types of that particular dataset. Every line is a different attribute containing three paramenters:
	* type: real, pos (positive), cat (categorical), ord (ordinal), count
	* dim: dimension of the variable
	* nclass: number of categories (for cat and ord)
* **Missingxx_y.csv**: a csv containing the positions of the different missing values in the data. Each "y" mask was generated randomly, containing a "xx" % of missing values.

You can add your own datasets as long as they follow this structure.


## Files description

* **HIVAE.py**: The main script of the library, it needs to imported to work with the library and is connected to all the other scripts.
* **loglik_ models_ missing_normalize.py**: In this file, the different likelihood models for the different types of variables considered (real, positive, count, categorical and ordinal) are included.
* **model_ HIVAE_inputDropout.py**: Contains the HI-VAE with input dropout encoder model.
* **model_ HIVAE_factorized.py**: Contains the HI-VAE with factorized encoder model

## Contact

* **For questions regarding algorithm --> Alfredo Nazabal**: anazabal@turing.ac.uk
* **For bugs or suggestion regarding code --> Fathy Shalaby**: fathy.mshalaby@outlook.com
