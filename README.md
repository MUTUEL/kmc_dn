# Kinetic Monte Carlo simulation tool for dopant networks

## Dependencies
+ python
+ numpy
+ matplotlib
+ numba
+ fenics, installation notes are [here](https://fenics.readthedocs.io/en/latest/installation.html)

## Recommended installation procedure

I would recommend to create a new conda environment to run this simulation code in. For the sake of example, let's name it *kmc*.
Run the following command to create the environment:
```
conda create -n kmc
```
Now activate the environment:
```
conda activate kmc
```
And run the following installs:
```
conda install numba
conda install -c conda-forge fenics
```
Now your environment is set up and all dependencies should be met.
