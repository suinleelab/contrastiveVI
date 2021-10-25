"""Download, read, and preprocess Blish et al. (2020) expression data.

Single-cell expression data from Blish et al. A single-cell atlas of the peripheral
immune response in patients with severe COVID-19. Nature Medicine (2020). """
import os

import pandas as pd
import scanpy as sc
from anndata import AnnData
from rpy2 import robjects
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects.packages import importr
from rpy2.robjects.pandas2ri import rpy2py


def download_blish_2020(output_path: str) -> None:
    """For this data, due to limitations with the Chan-Zuckerberg Biohub website,
    we can't download the data file programatically. Instead, this function redirects
    the user to the webpage where the file can be downloaded.

    Args:
        output_path: Path where raw data file should live.

    Returns:
        None. This function redirects the user to the Chan-Zuckerberg Biohub to download
        the data file if it doesn't already exist.
    """
    if not os.path.exists(os.path.join(output_path, "local.rds")):
        raise FileNotFoundError(
            "File cannot be downloaded automatically. Please download"
            "RDS file from "
            "https://cellxgene.cziscience.com/collections"
            "/a72afd53-ab92-4511-88da-252fb0e26b9a and place it in"
            "{output_path} to continue."
        )


def read_blish_2020(file_directory: str) -> pd.DataFrame:
    """Read the expression data for Blish et al. 2020 in the given directory.

    Args:
        file_directory: Directory containing Haber et al. 2017 data.

    Returns:
        A data frame containing single-cell gene expression count, with cell
        identification barcodes as column names and gene IDs as indices.
    """
    # Load in required R packages to handle Seurat object file
    try:
        base = importr("base")
        seurat_object = importr("SeuratObject")
    except RRuntimeError:
        raise ImportError(
            "Unable to load R environment with SeuratObject package. Please ensure you "
            "have a working installation of R with the SeuratObject "
            "package installed before continuing."
        )
    readRDS = robjects.r["readRDS"]

    rds_object = readRDS(os.path.join(file_directory, "local.rds"))

    r_df = base.as_data_frame(
        base.as_matrix(seurat_object.GetAssayData(object=rds_object, slot="counts"))
    )

    # Converts the R dataframe object to a pandas dataframe via rpy2
    pandas_df = rpy2py(r_df)

    return pandas_df


def preprocess_blish_2020(download_path: str, n_top_genes: int) -> AnnData:
    """Preprocess expression data from Blish et al., 2020.

    Args:
        download_path: Path containing the downloaded Blish et al. 2020 data file.
        n_top_genes: Number of most variable genes to retain.

    Returns:
        An AnnData object containing single-cell expression data. The layer
        "count" contains the count data for the most variable genes. The X
        variable contains the total-count-normalized and log-transformed data
        for the most variable genes (a copy with all the genes is stored in
        .raw).
    """

    df = read_blish_2020(download_path)
    df = df.transpose()

    readRDS = robjects.r["readRDS"]
    rds_object = readRDS(os.path.join(download_path, "local.rds"))
    metadata_r_df = rds_object.slots["meta.data"]
    metadata_pandas_df = rpy2py(metadata_r_df)

    adata = AnnData(X=df.values, obs=metadata_pandas_df)
    adata.layers["count"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(
        adata, flavor="seurat_v3", n_top_genes=n_top_genes, layer="count", subset=True
    )
    return adata
