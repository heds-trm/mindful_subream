import pandas as pd
from pathlib import Path
import argparse

from mindful_core.utils.data_constants import SCAN_ID


def remove_lesion_index(scan_id: str):
    return "_".join(scan_id.split("_")[:-1])


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source", required=True, type=str)
    arg_parser.add_argument("--destination", required=True, type=str)
    args = arg_parser.parse_args()

    source_path = Path(args.source)
    destination_path = Path(args.destination)

    data_frame = pd.read_csv(source_path, index_col=SCAN_ID)
    data_frame.index = data_frame.index.map(remove_lesion_index)
    data_frame = data_frame[~data_frame.index.duplicated(keep="first")]

    data_frame.to_csv(destination_path)


if __name__ == "__main__":
    main()
