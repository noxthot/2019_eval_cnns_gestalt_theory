# Evaluating CNNs on the gestalt principle of closure

The (hacky) code in this repository was used to create the training images used for the paper **Evaluating CNNs on the gestalt principle of closure** [[1]](#1).

## Setup (Ubuntu)
### Conda
Use conda to install all required modules (default environment: `eval_cnns_gestalt_theory`):
```
conda env create -f environment.yml
```

Before you work in this repository, do not forget to activate the environment:
```
conda activate eval_cnns_gestalt_theory
```

In case you already got the environment and only need to update to the latest `environment.yml` use:
```
conda activate eval_cnns_gestalt_theory
conda env update --file environment.yml --prune
```

#### Adding a package
To create a minimized environment.yml (for the sake of platform independency) we use `conda-minify`. It has to be installed into the `base` environment:
```
conda activate base
conda install conda-minify
```

After manually adding a package, update the `environment.yml` using this command:
```
conda run --name base conda-minify -n eval_cnns_gestalt_theory -f environment.yml
```

## References

<a id="1">[1]</a> 
Ehrensperger, Gregor, Sebastian Stabinger, and Antonio Rodríguez Sánchez. "Evaluating CNNs on the gestalt principle of closure." International Conference on Artificial Neural Networks. Springer, Cham, 2019.` (URL: [https://arxiv.org/abs/1904.00285](https://arxiv.org/abs/1904.00285))