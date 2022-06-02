"""Data preprocessing utilities."""
import os

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
