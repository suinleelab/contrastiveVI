"""Global constants for project scripts."""

DATASET_LIST = [
    "zheng_2017",
    "haber_2017",
    "blish_2020",
    "xiang_2020",
    "fasolino_2021",
    "mcfarland_2020",
]
DEFAULT_DATA_PATH = "/projects/leelab/data/single-cell"
DEFAULT_RESULTS_PATH = "/projects/leelab/contrastiveVI/results/"
DEFAULT_SEEDS = [123, 42, 789, 46, 999]

DATASET_SPLIT_LOOKUP = {
    "zheng_2017": {
        "split_key": "condition",
        "background_value": "healthy",
        "label_key": "condition",
    },
    "haber_2017": {
        "split_key": "condition",
        "background_value": "Control",
        "label_key": "condition",
    },
    "fasolino_2021": {
        "split_key": "disease_state",
        "background_value": "Control",
        "label_key": "disease_state",
    },
    "mcfarland_2020": {
        "split_key": "condition",
        "background_value": "DMSO",
        "label_key": "TP53_mutation_status",
    },
}
