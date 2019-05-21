![Kinetic Monte Carlo simulation tool for dopant networks](https://github.com/brambozz/kmc_dn/blob/master/misc/logo.png "Kinetic Monte Carlo simulation tool for dopant networks")

Welcome to the GitHub page for kmc_dn (Kinetic Monte Carlo for Dopant
Networks). This page houses a simulation tool that was specifically 
built to simulate Silicon Dopant networks. In practice, this tool
simulates variable range hopping between an arbitrary number of 
acceptor sites in a domain surrounded by an arbitrary number of 
electrodes. It is mostly written in Python, with (soon) some of the
more computationally heavy parts written in Go.

A good introduction to the code and the original experimental context
can be found in the main author's Master thesis (WHERE?).

To get started with the code, please have a look at the scripts in the
examples folder. They are well documented examples that should tell you
enough to get going.

# Installation

Installing the tool boils mainly down to meeting a few dependencies. A
suggested workflow (based on conda) is given afterwards.

## Dependencies
+ python 3.6+
+ numpy
+ matplotlib
+ numba
+ ffmpeg (for animations)
+ fenics, installation notes are [here](https://fenics.readthedocs.io/en/latest/installation.html)
+ logging
+ go

## Recommended installation procedure

The recommended way to manage the dependencies is through a virtual
environment in conda. The code can run perfectly without conda, but it is
a neat way to isolate the dependencies and to make sure the code will
always run.
For the sake of example, let's create an environment named *kmc*.
Run the following command (in the terminal) to create the environment:

```
conda create -n kmc
```

Now activate the environment:

```
conda activate kmc
```

And run the following installs:

```
conda install numpy
conda install matplotlib
conda install numba
conda install -c conda-forge fenics
```

Lastly, to make sure that kmc_dopant_networks is always found when 
running code in the kmc environment, 
look for the system path that looks like this:

```
~/anaconda3/envs/kmc/lib/python3.6/site-packages/
```

Then make a new file `kmc.pth` which contains the absolute path to the 
repo.

To use go functionalities you have to compile and build a library accessible by python inside the goSimulation folder.

```
go build -o libSimulation.so -buildmode=c-shared simulationWrapper.go simulation.go probabilitySimulation.go
```

## Get help

You are now all set to start simulating! If documentation in the code 
and the example files are not enough, please do not hesitate to contact
any of the authors below.

### Authors
Bram de Wilde / [brambozz](https://github.com/brambozz); b.dewilde-1@student.utwente.nl
