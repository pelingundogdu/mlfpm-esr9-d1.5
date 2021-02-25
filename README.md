# MLFPM - ESR9 - Deliverable 1.5
### Software implementing the method developed

This repository consists of python scripts, jupyter notebooks, R scripts, datasets, source codes, figures, evaluation metrics of the network which are created or obtained for this project.

==========================================================================================

Project Organization
------------------------

    ├── README.md              <- Project details.
    ├── data
    │   ├── EXPERIMENT         <- The index information
    │   ├── geneSCF            <- Pathway information from geneSCF tool. (https://github.com/genescf)
    │   ├── pathways           <- The details of pathway information.
    │   └── weights            <- The prior biological knowledge which includes into first hidden layer.
    │
    ├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                             and a short `-` delimited description, e.g.
    │                            `1.0-initial-data-exploration`.
    │
    ├── retrieval_analysis.sh  <- retrieval analysis
    │
    ├── source                 <- External data sources
    │   └── README.md          <- The explanation of data source
    │
    ├── tgpu.yml               <- Ptyhon environment
    │
    └── third_party            <- Source code from reference papers.
        ├── PMC5737331         <- Reference paper details.
        └── third_party.txt    <- Scripts to download or generate data
------------------------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
==========================================================================================

Tutorial
------------------------

1. Create environment

```
$ conda env create -f tgpu.yml
```

2. To execute R and Pyhon script in **_notebooks_** folder
3. To perform retrival analysis by
```
$ ./ retrieval_analysis.sh
```
