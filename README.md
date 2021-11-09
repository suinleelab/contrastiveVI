# contrastive-vi

Contrastive single-cell variational inference.

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
