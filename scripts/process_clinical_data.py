import pandas as pd
from pathlib import Path
import argparse

from mindful_core.utils.data_constants import SCAN_ID


def get_lesion_attributes(lesion_attributes_path: str) -> pd.DataFrame:
    lesion_attributes = pd.read_csv(lesion_attributes_path)

    def get_lesion_id(mask_filename: str) -> str:
        mask_filepath = Path(mask_filename)
        patient_id, *_, lesion_index, _ = mask_filepath.name.split("_")
        return "{}_{}".format(patient_id, lesion_index)

    lesion_attributes_index = lesion_attributes["mask_filename"].apply(get_lesion_id)
    lesion_attributes = lesion_attributes.drop(["mask_filename", "type"], axis="columns")
    lesion_attributes.index = lesion_attributes_index

    return lesion_attributes


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", required=True, type=str)
    arg_parser.add_argument("--reference_fold", required=True, type=str)
    arg_parser.add_argument("--lesion_attributes", required=False, type=str, default=None)
    args = arg_parser.parse_args()

    filepath = Path(args.path)
    # noinspection PyTypeChecker
    clinical_data = pd.read_excel(filepath, sheet_name="clinical", index_col="Code")
    clinical_data.index = clinical_data.index.map(str)
    clinical_data = clinical_data.replace(".", "")

    known_scalars = ["age"]
    base_categorical_data = clinical_data[[column for column in clinical_data.columns if column not in known_scalars]]
    base_scalar_data = clinical_data[[column for column in clinical_data.columns if column in known_scalars]]

    reference_fold = pd.read_csv(args.reference_fold, index_col=SCAN_ID)

    if args.lesion_attributes is not None:
        lesion_attributes = get_lesion_attributes(args.lesion_attributes)
        lesion_attributes_rows = {}
    else:
        lesion_attributes, lesion_attributes_rows = None, None

    categorical_data_rows = {}
    scalar_data_rows = {}
    for scan_id, row in reference_fold.iterrows():
        scan_id: str
        patient_code = scan_id.split("_")[0]
        lesion_code = "_".join(scan_id.split("_")[:2])
        if patient_code in clinical_data.index:
            categorical_data_rows[scan_id] = base_categorical_data.loc[patient_code]
            scalar_data_rows[scan_id] = base_scalar_data.loc[patient_code]
            if (lesion_attributes is not None) and (lesion_code in lesion_attributes.index):
                lesion_attributes_rows[scan_id] = lesion_attributes.loc[lesion_code]

    categorical_filepath = filepath.parent / "categorical_features.csv"
    scalar_filepath = filepath.parent / "scalar_features.csv"

    categorical_data = pd.DataFrame.from_dict(categorical_data_rows, orient="index")
    scalar_data = pd.DataFrame.from_dict(scalar_data_rows, orient="index")

    if lesion_attributes is not None:
        lesion_attributes_data = pd.DataFrame.from_dict(lesion_attributes_rows, orient="index")
        scalar_data = pd.concat([scalar_data, lesion_attributes_data], axis=1)

    categorical_data.index.name = SCAN_ID
    scalar_data.index.name = SCAN_ID

    categorical_data.to_csv(categorical_filepath)
    scalar_data.to_csv(scalar_filepath)


if __name__ == "__main__":
    main()
