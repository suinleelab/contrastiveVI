"""
Download, read, and preprocess Norman et al. (2019) expression data.

Single-cell expression data from Norman et al. Exploring genetic interaction
manifolds constructed from rich single-cell phenotypes. Science (2019).
"""

import gzip
import os
import re

import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.io import mmread
from scipy.sparse import coo_matrix

from contrastive_vi.data.utils import download_binary_file

# Gene program lists obtained by cross-referencing the heatmap here
# https://github.com/thomasmaxwellnorman/Perturbseq_GI/blob/master/GI_optimal_umap.ipynb
# with Figure 2b in Norman 2019
G1_CYCLE = [
    "CDKN1C+CDKN1B",
    "CDKN1B+ctrl",
    "CDKN1B+CDKN1A",
    "CDKN1C+ctrl",
    "ctrl+CDKN1A",
    "CDKN1C+CDKN1A",
    "CDKN1A+ctrl",
]

ERYTHROID = [
    "BPGM+SAMD1",
    "ATL1+ctrl",
    "UBASH3B+ZBTB25",
    "PTPN12+PTPN9",
    "PTPN12+UBASH3A",
    "CBL+CNN1",
    "UBASH3B+CNN1",
    "CBL+UBASH3B",
    "UBASH3B+PTPN9",
    "PTPN1+ctrl",
    "CBL+PTPN9",
    "CNN1+UBASH3A",
    "CBL+PTPN12",
    "PTPN12+ZBTB25",
    "UBASH3B+PTPN12",
    "SAMD1+PTPN12",
    "SAMD1+UBASH3B",
    "UBASH3B+UBASH3A",
]

PIONEER_FACTORS = [
    "ZBTB10+SNAI1",
    "FOXL2+MEIS1",
    "POU3F2+CBFA2T3",
    "DUSP9+SNAI1",
    "FOXA3+FOXA1",
    "FOXA3+ctrl",
    "LYL1+IER5L",
    "FOXA1+FOXF1",
    "FOXF1+HOXB9",
    "FOXA1+HOXB9",
    "FOXA3+HOXB9",
    "FOXA3+FOXA1",
    "FOXA3+FOXL2",
    "POU3F2+FOXL2",
    "FOXF1+FOXL2",
    "FOXA1+FOXL2",
    "HOXA13+ctrl",
    "ctrl+HOXC13",
    "HOXC13+ctrl",
    "MIDN+ctrl",
    "TP73+ctrl",
]

GRANULOCYTE_APOPTOSIS = [
    "SPI1+ctrl",
    "ctrl+SPI1",
    "ctrl+CEBPB",
    "CEBPB+ctrl",
    "JUN+CEBPA",
    "CEBPB+CEBPA",
    "FOSB+CEBPE",
    "ZC3HAV1+CEBPA",
    "KLF1+CEBPA",
    "ctrl+CEBPA",
    "CEBPA+ctrl",
    "CEBPE+CEBPA",
    "CEBPE+SPI1",
    "CEBPE+ctrl",
    "ctrl+CEBPE",
    "CEBPE+RUNX1T1",
    "CEBPE+CEBPB",
    "FOSB+CEBPB",
    "ETS2+CEBPE",
]

MEGAKARYOCYTE = [
    "ctrl+ETS2",
    "MAPK1+ctrl",
    "ctrl+MAPK1",
    "ETS2+MAPK1",
    "CEBPB+MAPK1",
    "MAPK1+TGFBR2",
]

PRO_GROWTH = [
    "CEBPE+KLF1",
    "KLF1+MAP2K6",
    "AHR+KLF1",
    "ctrl+KLF1",
    "KLF1+ctrl",
    "KLF1+BAK1",
    "KLF1+TGFBR2",
]


def download_norman_2019(output_path: str) -> None:
    """
    Download Norman et al. 2019 data and metadata files from the hosting URLs.

    Args:
    ----
        output_path: Output path to store the downloaded and unzipped
        directories.

    Returns
    -------
        None. File directories are downloaded to output_path.
    """

    file_urls = (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl"
        "/GSE133344_filtered_matrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl"
        "/GSE133344_filtered_genes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl"
        "/GSE133344_filtered_barcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl"
        "/GSE133344_filtered_cell_identities.csv.gz",
    )

    for url in file_urls:
        output_filename = os.path.join(output_path, url.split("/")[-1])
        download_binary_file(url, output_filename)


def read_norman_2019(file_directory: str) -> coo_matrix:
    """
    Read the expression data for Norman et al. 2019 in the given directory.

    Args:
    ----
        file_directory: Directory containing Norman et al. 2019 data.

    Returns
    -------
        A sparse matrix containing single-cell gene expression count, with rows
        representing genes and columns representing cells.
    """

    with gzip.open(
        os.path.join(file_directory, "GSE133344_filtered_matrix.mtx.gz"), "rb"
    ) as f:
        matrix = mmread(f)

    return matrix


def preprocess_norman_2019(download_path: str, n_top_genes: int) -> AnnData:
    """
    Preprocess expression data from Norman et al. 2019.

    Args:
    ----
        download_path: Path containing the downloaded Norman et al. 2019 data file.
        n_top_genes: Number of most variable genes to retain.

    Returns
    -------
        An AnnData object containing single-cell expression data. The layer
        "count" contains the count data for the most variable genes. The .X
        variable contains the normalized and log-transformed data for the most variable
        genes. A copy of data with all genes is stored in .raw.
    """
    matrix = read_norman_2019(download_path)

    # List of cell barcodes. The barcodes in this list are stored in the same order
    # as cells are in the count matrix.
    cell_barcodes = pd.read_csv(
        os.path.join(download_path, "GSE133344_filtered_barcodes.tsv.gz"),
        sep="\t",
        header=None,
        names=["cell_barcode"],
    )

    # IDs/names of the gene features.
    gene_list = pd.read_csv(
        os.path.join(download_path, "GSE133344_filtered_genes.tsv.gz"),
        sep="\t",
        header=None,
        names=["gene_id", "gene_name"],
    )

    # Dataframe where each row corresponds to a cell, and each column corresponds
    # to a gene feature.
    matrix = pd.DataFrame(
        matrix.transpose().todense(),
        columns=gene_list["gene_id"],
        index=cell_barcodes["cell_barcode"],
        dtype="int32",
    )

    # Dataframe mapping cell barcodes to metadata about that cell (e.g. which CRISPR
    # guides were applied to that cell). Unfortunately, this list has a different
    # ordering from the count matrix, so we have to be careful combining the metadata
    # and count data.
    cell_identities = pd.read_csv(
        os.path.join(download_path, "GSE133344_filtered_cell_identities.csv.gz")
    ).set_index("cell_barcode")

    # This merge call reorders our metadata dataframe to match the ordering in the
    # count matrix. Some cells in `cell_barcodes` do not have metadata associated with
    # them, and their metadata values will be filled in as NaN.
    aligned_metadata = pd.merge(
        cell_barcodes,
        cell_identities,
        left_on="cell_barcode",
        right_index=True,
        how="left",
    ).set_index("cell_barcode")

    adata = AnnData(matrix)
    adata.obs = aligned_metadata

    # Filter out any cells that don't have metadata values.
    rows_without_nans = [
        index for index, row in adata.obs.iterrows() if not row.isnull().any()
    ]
    adata = adata[rows_without_nans, :]

    # Remove these as suggested by the authors. See lines referring to
    # NegCtrl1_NegCtrl0 in GI_generate_populations.ipynb in the Norman 2019 paper's
    # Github repo https://github.com/thomasmaxwellnorman/Perturbseq_GI/
    adata = adata[adata.obs["guide_identity"] != "NegCtrl1_NegCtrl0__NegCtrl1_NegCtrl0"]

    # We create a new metadata column with cleaner representations of CRISPR guide
    # identities. The original format is <Guide1>_<Guide2>__<Guide1>_<Guide2>_<number>
    adata.obs["guide_merged"] = adata.obs["guide_identity"]

    control_regex = re.compile(r"NegCtrl(.*)_NegCtrl(.*)+NegCtrl(.*)_NegCtrl(.*)")
    for i in adata.obs["guide_merged"].unique():
        if control_regex.match(i):
            # For any cells that only had control guides, we don't care about the
            # specific IDs of the guides. Here we relabel them just as "ctrl".
            adata.obs["guide_merged"].replace(i, "ctrl", inplace=True)
        else:
            # Otherwise, we reformat the guide label to be <Guide1>+<Guide2>. If Guide1
            # or Guide2 was a control, we replace it with "ctrl".
            split = i.split("__")[0]
            split = split.split("_")
            for j, string in enumerate(split):
                if "NegCtrl" in split[j]:
                    split[j] = "ctrl"
            adata.obs["guide_merged"].replace(i, f"{split[0]}+{split[1]}", inplace=True)

    guides_to_programs = {}
    guides_to_programs.update(dict.fromkeys(G1_CYCLE, "G1 cell cycle arrest"))
    guides_to_programs.update(dict.fromkeys(ERYTHROID, "Erythroid"))
    guides_to_programs.update(dict.fromkeys(PIONEER_FACTORS, "Pioneer factors"))
    guides_to_programs.update(
        dict.fromkeys(GRANULOCYTE_APOPTOSIS, "Granulocyte/apoptosis")
    )
    guides_to_programs.update(dict.fromkeys(PRO_GROWTH, "Pro-growth"))
    guides_to_programs.update(dict.fromkeys(MEGAKARYOCYTE, "Megakaryocyte"))
    guides_to_programs.update(dict.fromkeys(["ctrl"], "Ctrl"))

    # We only keep cells whose guides were either controls or are labeled with a
    # specific gene program
    adata = adata[adata.obs["guide_merged"].isin(guides_to_programs.keys())]
    adata.obs["gene_program"] = [
        guides_to_programs[x] for x in adata.obs["guide_merged"]
    ]

    adata.obs["good_coverage"] = adata.obs["good_coverage"].astype(bool)

    adata.layers["count"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(
        adata, flavor="seurat_v3", n_top_genes=n_top_genes, layer="count", subset=True
    )
    adata = adata[adata.layers["count"].sum(1) != 0]  # Remove cells with all zeros.
    return adata
