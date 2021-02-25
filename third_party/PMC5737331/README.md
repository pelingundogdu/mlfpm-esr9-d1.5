# README

### Create environment

#### This paper use 'theano' package. Thus, we need a python environment in 2.7 version. We checked the package in 'requirement.txt' file and based on that document we create an environment in Python 2.7 with required libraries, some of them have specific version.

conda create -n PMC5737331_2 python=2.7 theano numpy scipy matplotlib seaborn palettable statistics ipykernel

#### Import R environment into conda

conda activate PMC5737331_2

conda install -c conda-forge -c bioconda r-irkernel bioconductor-hipathia

#### Specific versions;
pip install keras==1.1.0

conda install -c conda-forge tensorflow=1.4

conda install -c conda-forge scikit-learn=0.19

#### Exporting environment into text file.
pip freeze > requirements_PMC5737331.txt


### Used during the execution of 'deep_net.py' code
$ KERAS_BACKEND=theano python deep_net.py




