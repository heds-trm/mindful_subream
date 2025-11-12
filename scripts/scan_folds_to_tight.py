import pandas as pd
from pathlib import Path
import argparse

from mindful_core.utils.data_constants import SCAN_ID


def remove_lesion_index(scan_id: str) -> str:
    return scan_id[:-2]


def update_scan_filepath(scan_filepath: str) -> str:
    return scan_filepath.replace("50.0_mm",
                                 "tight")


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", required=True, type=str)
    args = arg_parser.parse_args()
    path = Path(args.path)

    folds_paths = []
    for filepath in path.iterdir():
        if filepath.match("*_fold_*.csv"):
            folds_paths.append(filepath)

    for fold_path in folds_paths:
        fold = pd.read_csv(fold_path)
        fold[SCAN_ID] = fold[SCAN_ID].apply(remove_lesion_index)
        fold["ScanFilepath"] = fold["ScanFilepath"].apply(update_scan_filepath)
        fold: pd.DataFrame = fold.drop_duplicates(SCAN_ID, keep="first")
        fold.to_csv(fold_path, index=False)


if __name__ == "__main__":
    main()
