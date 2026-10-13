DATASETS = {
    "dataset1": {
        "title": "Dataset 1 - Credit Card Default",
        "csv": "data/dataset1.csv",
        "target": "default",
        "drop": ["rownames"],
        "labels": {},
    },
    "dataset2": {
        "title": "Dataset 2 - Breast Biopsy (benign/malignant)",
        "csv": "data/dataset2.csv",
        "target": "class",
        "drop": ["rownames", "ID"],
        "labels": {},
    },
}