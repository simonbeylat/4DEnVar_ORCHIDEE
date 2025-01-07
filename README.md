4DEnVar ORCHIDEE
=========================

<!---
"[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10592299.svg)](https://doi.org/10.5281/zenodo.10592299)
-->


An application of 4DEnVar to assimilate Atmospheric CO2 Concentration Data for parameter calibration in ORCHIDEE Land Surface Model

Installation
============

We recommend to install and use the conda environment provided in this repository. 

    $ conda install -c conda-forge --name 4DEnVar --file requirements.txt 
    $ conda activate 4DEnVar
    
Usage
=====

This repository contains two Python scripts that execute the 4DEnVar method. The `FourDEnVar.py` script is the original script employed in the ORCHIDAS software. 
To test the script, a simplified version has been adapted so that the optimisation can be run using the output file present in this repository without the need for the ORCHIDAS software.


Following the unzipping of the ZIP file, which is located in the `data` directory, 

    $ unzip VCMAX.zip
    $ unzip 5P.zip
  
the `Example.py` script can be run to initiate the optimisation process.

    $ python Example.py

Reference publications
======================
  
Contributors
============
Simon Beylat
This script was inspired by the script used in [Lavender](https://github.com/pyearthsci/lavendar) available on github.

