"""Data preprocessing utilities."""
import os
from typing import Tuple

import numpy as np
import pandas as pd
import requests
from anndata import AnnData
from rpy2 import robjects
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects.packages import importr
from rpy2.robjects.pandas2ri import rpy2py


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


def read_seurat_raw_counts(file_path: str) -> pd.DataFrame:
    """
    Read raw expression count data from a Seurat R object.

    Args:
    ----
        file_path: Path to RDS file containing Seurat R object.

    Returns
    -------
        A (pandas) dataframe containing the count data stored in the Seurat object. This
        data frame has cell identification barcodes as column names and gene IDs as
        indices.
    """
    try:
        readRDS = robjects.r["readRDS"]
        base = importr("base")
        seurat_object = importr("SeuratObject")
    except RRuntimeError:
        raise ImportError(
            "Unable to load R environment with SeuratObject package. Please ensure you "
            "have a working installation of R with the SeuratObject "
            "package installed before continuing."
        )

    rds_object = readRDS(file_path)

    r_df = base.as_data_frame(
        base.as_matrix(seurat_object.GetAssayData(object=rds_object, slot="counts"))
    )

    # Converts the R dataframe object to a pandas dataframe via rpy2
    pandas_df = rpy2py(r_df)
    return pandas_df


def read_seurat_cell_metadata(file_path: str) -> pd.DataFrame:
    """
    Read cell metadata from a Seurat R object.

    Args:
    ----
        file_path: Path to RDS file containing Seurat R object.

    Returns
    -------
        A (pandas) dataframe containing metadata for each cell in the Seurat object.
        For this dataframe rows are cells while columns represent metadata features.
    """
    try:
        readRDS = robjects.r["readRDS"]
    except RRuntimeError:
        raise ImportError(
            "Unable to load R environment. Please ensure you "
            "have a working installation of R before continuing."
        )
    rds_object = readRDS(file_path)

    metadata_r_df = rds_object.slots["meta.data"]
    metadata_pandas_df = rpy2py(metadata_r_df)
    return metadata_pandas_df


def read_seurat_feature_metadata(file_path: str) -> pd.DataFrame:
    """
    Read feature metadata from a Seurat R object.

    Args:
    ----
        file_path: Path to RDS file containing Seurat R object.

    Returns
    -------
        A (pandas) dataframe containing metadata for each gene feature in the Seurat
        object. For this dataframe rows are genes while columns represent metadata
        features.
    """
    try:
        readRDS = robjects.r["readRDS"]
        dollar_sign = robjects.r["$"]
        double_bracket = robjects.r["[["]
    except RRuntimeError:
        raise ImportError(
            "Unable to load R environment. Please ensure you "
            "have a working installation of R before continuing."
        )
    rds_object = readRDS(file_path)

    # This line is equivalent to the R code `rds_object$RNA[[]]`, which is used to
    # access metadata on the features stored in the Seurat object
    feature_metadata_r_df = double_bracket(dollar_sign(rds_object, "RNA"))

    feature_metadata_pandas_df = rpy2py(feature_metadata_r_df)
    return feature_metadata_pandas_df


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
