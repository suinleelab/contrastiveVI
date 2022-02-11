# contrastiveVI

<center>
    <img src="./sketch.png?raw=true" width="750">
</center>

contrastiveVI is a generative model designed to isolate factors of variation specific to 
a group of "target" cells (e.g. from specimens with a given disease) from those shared
with a group of "background" cells (e.g. from healthy specimens). contrastiveVI is
implemented in [scvi-tools](https://scvi-tools.org/).

## User guide

### Installation

To install the latest version of contrastiveVI via pip

```
pip install contrastive-vi
```

### What you can do with contrastiveVI

* If you have a dataset with cells in a background condition (e.g. from healthy
controls) and a target condition (e.g. from diseased patients), you can train
contrastiveVI to isolate latent factors of variation specific to the target cells
from those shared with a background into separate latent spaces.
* Run clustering algorithms on the target-specific latent space to discover sub-groups
of target cells
* Perform differential expression testing for discovered sub-groups of target cells 
using a procedure similar to that of [scVI
](https://www.nature.com/articles/s41592-018-0229-2).

### Colab Notebook Examples

* [Applying contrastiveVI to separate mouse intestinal epithelial cells
infected with different pathogens by pathogen type
](https://colab.research.google.com/drive/1z0AcKQg7juArXGCx1XKj6skojWKRlDMC?usp=sharing)
* [Applying contrastiveVI to better understand the results of a MIX-Seq
small-molecule drug perturbation experiment
](https://colab.research.google.com/drive/1cMaJpMe3g0awCiwsw13oG7RvGnmXNCac?usp=sharing)



## Development guide

### Set up the environment
1. Git clone this repository.
2. `cd contrastive-vi`.
3. Create and activate the specified conda environment by running
    ```
    conda env create -f environment.yml
    conda activate contrastive-vi-env
    ```
4. Install the `constrative_vi` package and necessary dependencies for
development by running `pip install -e ".[dev]"`.
5. Git pre-commit hooks (https://pre-commit.com/) are used to automatically
check and fix formatting errors before a Git commit happens. Run
`pre-commit install` to install all the hooks.
6. Test that the pre-commit hooks work by running `pre-commit run --all-files`.

### Testing
It's a good practice to include unit tests during development.
Run `pytest tests` to verify existing tests.


## References

If you find contrastiveVI useful for your work, please consider citing our preprent:

```
@article{contrastiveVI,
  title={Isolating salient variations of interest in single-cell transcriptomic data with contrastiveVI},
  author={Weinberger, Ethan and Lin, Chris and Lee, Su-In},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```
