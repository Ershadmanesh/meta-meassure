# Overview


This repository hosts the data and code necessary to replicate the findings of our research on "Metacognitive Efficiency in Learned Value-based Choice."

# Installations

To set up the required environment, execute:

`pip install -r requirements.txt`

This code uses data from an external study, accessible at: https://github.com/kdesende/Two-armed-bandit-confidence. Ensure to create a `data/` directory at the root level and transfer the dataset there for full functionality.

# Code organization

The code is organized according to the sections of the paper.

Code for:
- The `notebooks/` directory contains Jupyter notebooks for reproducing the study's results. For example, `Figure_3_4_5.ipynb` contains code for producing figures 3, 4, and 5 in the paper.
- Essential functions for model simulation and fitting are located within the `functions/` folder. Additionally, this folder contains utility functions for plotting and preprocessing the data, enhancing the analysis and visualization of results.
- Results from simulations and model fittings can be found in the `results/` folder.
 

If you have any questions, please feel free to contact corresponding author.


