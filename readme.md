# Code release to "Efficient Rollout Strategies for Bayesian Optimization", published in UAI 2020

---
This repository contains the code used to produce the results in the paper [Efficient Rollout Strategies for Bayesian Optimization](https://arxiv.org/abs/2002.10539). This code is largely based off the [Metric Optimization Engine (MOE)](https://github.com/wujian16/Cornell-MOE) open-source Bayesian optimization library, albeit simplified in many areas. We note that this code is not meant to be a fully-fledged Bayesian optimization software package (e.g., it does not support categorical variables), and is primarily intended to illustrate the more important concepts of our paper. 

# Requirements
To keep things simple, we recommend the latest version of [Anaconda](https://www.anaconda.com/). Otherwise, our requirements are:

* Python >= 3.6
* The latest version of numpy, scipy, jupyter and torch. 

# Installation
Run `python setup.py install` in the command line. 

# Directory Structure
The `lookahead/` directory is subdivided as follows:

* `lookahead/acquisitions/` contains implementations of all acquisition functions. This includes expected improvement, upper confidence bound, knowledge gradient, rollout of EI, and policy search. 
* `lookahead/model/` contains our GP implementation. By default, this implementation uses a constant mean function and the Matern 5/2 kernel. 
* `lookahead/runners/` contains the BO runners, which are invoked to run a full BO loop.
* `lookahead/test_problems/` contains implementations of synthetic functions, which are used to benchmark BO acquisition functions. 

# Demos 
We have a few simple demos in the `demos/` folder, which you should run to get started. 