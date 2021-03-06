# MLFPM - ESR9 - Deliverable 1.5
### Software implementing the method developed

This repository consists of python scripts, jupyter notebooks, R scripts, datasets, source codes, figures, evaluation metrics of the network which are created or obtained for this project.

=================================================

Project Organization
------------------------

    proposed_model             <- Project folder
    │
    ├── data
    │   ├── EXPERIMENT         <- The index information
    │   ├── geneSCF            <- Pathway information from geneSCF platform in https://github.com/genescf
    │   ├── pathways           <- The details of pathway information
    │   └── weights            <- The prior biological knowledge which includes into first hidden layer
    │
    ├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                             and a short `-` delimited description, e.g.
    │                            `1.0-initial-data-exploration`
    │
    ├── source                 <- External data sources
    │   └── README.md          <- The explanation of data source
    │
    ├── third_party            <- Source code from reference papers
    │   ├── PMC5737331         <- Reference paper code details
    │   └── third_party.txt    <- reference paper link information
    │
    ├── README.md              <- Project details
    │
    ├── retrieval_analysis.sh  <- retrieval analysis
    │
    └── tgpu.yml               <- Ptyhon environment
    
------------------------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
=================================================

Tutorial
------------------------

1. Create environment
```
... $ conda env create -f tgpu.yml
```

2. To execute R and Pyhon script in **_notebooks_** folder

3. To perform retrival analysis by
```
.../proposed_model$ ./ retrieval_analysis.sh
``` 

4. Retrieval Analysis Result
```
.../proposed_model$ python notebooks/load_retrieval_summary.py retrieval_analysis
```

Funding
------------------------
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the **Marie Sklodowska-Curie grant agreement no 813533**.

More detail in [MLFPM webpage](https://mlfpm.eu/)
