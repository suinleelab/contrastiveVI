# contrastive-vi

Contrastive variational inference for single-cell transcriptomic data.

## User guide

### Set up the environment
1. Git clone this repository.
2. `cd contrastive-vi`.
3. Create and activate the specified conda environment by running
    ```
    conda env create -f environment.yml
    conda activate contrastive-vi-env
    ```
4. Install the `contrastive_vi` package by running `pip install .`.

### Obtain results
After setting up the environment, do the following to train and evaluate contrastiveVI
and baseline models.
1. Download and preprocess data using `scripts/preprocess_data.py`.
2. Modify the optimization and input/output global variables in `scripts/constants.py`.
3. Train a model on one dataset using `scripts/run_experiment.py` or on multiple
datasets using `scripts/run_many_experiments.py`.
4. Evaluate the trained models using `scripts/evaluate_performance.py`.

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
