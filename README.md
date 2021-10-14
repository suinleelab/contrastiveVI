# contrastive-vi

Contrastive single-cell variational inference.

## Development guide

### Set up the environment
1. Git clone this repository.
2. Activate your virtual environment (e.g. conda). Make sure that it has a 
working version of `pip`.
3. `cd contrastive-vi`
4. Install the `constrative-vi` package and necessary dependencies for 
development by running `pip install -e ".[dev]"`.
5. Git pre-commit hooks (https://pre-commit.com/) are used to automatically 
check and fix formatting errors before a Git commit happens. Run 
`pre-commit install` to install all the hooks.
6. Test that the pre-commit hooks work by running `pre-commit run --all-files`.