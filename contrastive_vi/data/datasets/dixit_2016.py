"""Download, read, and preprocess Dixit et al. (2016) expression data.

Single-cell expression data from Dixit et al. Perturb-Seq: Dissecting Molecular
Circuits with Scalable Single-Cell RNA Profiling of Pooled Genetic Screens. Cell (2020)
"""
import os

import pandas as pd
import scanpy as sc
from anndata import AnnData
from typing import Tuple
import tarfile

from contrastive_vi.data.utils import download_binary_file


def download_dixit_2016(output_path: str) -> None:
    """Download Dixit et al. 2017 data from the hosting URL and extracts the files
    in the tarball to `output_path`.

    Args:
        output_path: Output path to store the extracted files.

    Returns:
        None. Extracted files are saved in output_path.
    """
    url = (
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE90063&format=file"
    )

    tar_filename = os.path.join(output_path, "dixit_2016.tar")
    download_binary_file(url, tar_filename)

    tar_handle = tarfile.open(tar_filename)
    tar_handle.extractall(output_path)
    tar_handle.close()


def _read_perturbseq_df(directory: str, file_prefix: str) -> pd.DataFrame:
    # Using scanpy here since it will take care of decompressing the gz file
    counts = sc.read_mtx(
        os.path.join(directory, f"{file_prefix}.mtx.txt.gz")).T.X.todense()

    barcodes = pd.read_csv(
        os.path.join(directory, f"{file_prefix}_cellnames.csv.gz"),
        index_col=0,
        names=['Barcode'],
        header=None,
        skiprows=1
    )

    genes = pd.read_csv(
        os.path.join(directory, f"{file_prefix}_genenames.csv.gz"),
        index_col=0,
        names=['Concat_Symbol_ID'],
        skiprows=1
    )
    gene_ids = [x.split('_')[0] for x in genes['Concat_Symbol_ID']]
    df = pd.DataFrame(counts, columns=gene_ids, index=barcodes['Barcode'])
    return df


def _read_perturbation_information(directory: str, file_prefix: str) -> pd.DataFrame:
    perturbations = pd.read_csv(
        os.path.join(directory, file_prefix),
        header=None,
        names=['Perturbed_Gene', 'barcodes']
    )

    perturb_dict = {}
    perturb_dict_concat = {}

    for i in range(len(perturbations)):
        perturbed_gene = perturbations.iloc[i, 0]
        cells = perturbations.iloc[i, 1].split(', ')
        for cell in cells:
            perturb_dict.setdefault(cell, list()).append(perturbed_gene)

    for k, v in perturb_dict.items():
        perturb_dict_concat[k] = ' '.join(sorted(v))

    perturb_df = pd.DataFrame(perturb_dict_concat.values(), perturb_dict_concat.keys(),
                              columns=['Perturbed_Genes'])
    return perturb_df


def read_dixit_2016(file_directory: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read the expression data for Download Dixit et al. 2016 the given directory.

    Args:
        file_directory: Directory containing Dixit et al. 2016 data.

    Returns:
        TODO: Fill this in.
    """
    control_df = _read_perturbseq_df(file_directory, file_prefix="GSM2396857_dc_0hr")
    stim_df = _read_perturbseq_df(file_directory, file_prefix="GSM2396856_dc_3hr")

    return control_df, stim_df

def preprocess_dixit_2016(download_path: str, n_top_genes: int) -> AnnData:
    control_df, stim_df = read_dixit_2016(download_path)

    import pdb
    pdb.set_trace()

    control_perturbation_df = _read_perturbation_information(
        download_path,
        "GSM2396857_dc_0hr_cbc_gbc_dict.csv.gz"
    )
    stim_perturbation_df = _read_perturbation_information(
        download_path,
        "GSM2396856_dc_3hr_cbc_gbc_dict_strict.csv.gz"
    )

    control_adata = AnnData(control_df)
    stim_adata = AnnData(control_df)

    control_adata.obs['perturbation'] = control_perturbation_df['Perturbed Genes']
    stim_adata.obs['perturbation'] = stim_perturbation_df['Perturbed Genes']

    merged = control_adata.concatenate(stim_adata)
    merged.obs['perturbation_simple'] = ['Multi' if len(x.split()) > 1 else x for x in
                                         merged.obs.perturbation]
    merged.obs['perturbation_gene'] = [x.split('_')[1] if '_' in x else 'None' for x in
                                       merged.obs['perturbation_simple']]
    merged.obs['MOI'] = [len(x.split()) if x != 'None' else 0 for x in
                         merged.obs.perturbation]

    sc.pp.normalize_total(merged)
    sc.pp.log1p(merged)
    merged.raw = merged
    sc.pp.highly_variable_genes(
        merged, flavor="seurat_v3", n_top_genes=n_top_genes, layer="count", subset=True
    )
