"""Data preprocessing utilities."""
import os
from typing import Tuple

import numpy as np
import requests
from anndata import AnnData


def download_binary_file(file_url: str, output_path: str) -> None:
    """
    Download binary data file from a URL.

    Args:
    ----
        file_url: URL where the file is hosted.
        output_path: Output path for the downloaded file.

    Returns
    -------
        None.
    """
    request = requests.get(file_url)
    with open(output_path, "wb") as f:
        f.write(request.content)
    print(f"Downloaded data from {file_url} at {output_path}")


def get_library_log_means_and_vars(adata: AnnData) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the mean and variance of log library size for each experimental batch.

    Args:
    ----
        adata: AnnData object that has been registered via `setup_anndata`.

    Returns
    -------
        A tuple of numpy array `library_log_means` and `library_log_vars` for the mean
        and variance, respectively. Each has shape `(1, n_batch)`.
    """
    count_data_registry = adata.uns["_scvi"]["data_registry"]["X"]
    if count_data_registry["attr_name"] == "layers":
        count_data = adata.layers[count_data_registry["attr_key"]]
    else:
        count_data = adata.X

    library_log_means = []
    library_log_vars = []
    batches = adata.obs["_scvi_batch"].unique()
    for batch in batches:
        if len(batches) > 1:
            library = count_data[adata.obs["_scvi_batch"] == batch].sum(1)
        else:
            library = count_data.sum(1)
        library_log = np.ma.log(library)
        library_log = library_log.filled(0.0)  # Fill invalid log values with zeros.
        library_log_means.append(library_log.mean())
        library_log_vars.append(library_log.var())
    library_log_means = np.array(library_log_means)[np.newaxis, :]
    library_log_vars = np.array(library_log_vars)[np.newaxis, :]
    return library_log_means, library_log_vars


def save_preprocessed_adata(adata: AnnData, output_path: str) -> None:
    """
    Save given AnnData object with preprocessed data to disk using our dataset file
    naming convention.

    Args:
    ----
        adata: AnnData object containing expression count data as well as metadata.
        output_path: Path to save resulting file.

    Returns
    -------
        None. Provided AnnData object is saved to disk in a subdirectory called
        "preprocessed" in output_path.
    """
    preprocessed_directory = os.path.join(output_path, "preprocessed")
    os.makedirs(preprocessed_directory, exist_ok=True)
    n_genes = adata.shape[1]
    filename = os.path.join(
        preprocessed_directory,
        f"adata_top_{n_genes}_genes.h5ad",
    )
    adata.write_h5ad(filename=filename)
