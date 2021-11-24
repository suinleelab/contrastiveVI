"""Run multiple model training experiments."""
import argparse
import os

dataset_list = ["mcfarland_2020", "zheng_2017", "haber_2017", "fasolino_2021"]
latent_size_list = [2, 10, 32, 64]

parser = argparse.ArgumentParser()
parser.add_argument(
    "method",
    type=str,
    choices=[
        "contrastiveVI",
        "TC_contrastiveVI",
        "mmd_contrastiveVI",
        "scVI",
        "PCPCA",
        "cPCA",
        "CPLVM",
        "cVAE",
    ],
    help="Which model to train",
)
parser.add_argument(
    "-use_gpu", action="store_true", help="Flag for enabling GPU usage."
)
parser.add_argument(
    "--gpu_num",
    type=int,
    help="If -use_gpu is enabled, controls which specific GPU to use for training.",
)
args = parser.parse_args()
for dataset in dataset_list:
    for latent_size in latent_size_list:
        command = f"python scripts/run_experiment.py {dataset} {args.method}"
        command += f" --latent_size {latent_size}"
        if args.use_gpu:
            command += " -use_gpu"
        if args.gpu_num is not None:
            command += f" --gpu_num {args.gpu_num}"
        os.system(command)
