import pandas as pd
from pathlib import Path
import argparse

from mindful_core.scripts.outcomes.compute_average_generalized_score import compute_average_generalized_dice
from mindful_subream.dataset.lesion_attributes_extraction import process_folder as extract_lesion_attributes


def process_fold_segmentations(input_folder: Path,
                               ground_truth_folder: Path,
                               segmentation_folder: Path
                               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = {
        "folder": input_folder,
        "binarize": True,
        "min_pixel_count": 50,
        "attributes_filter": "min_distance_to_center",
        "save_objects": True
    }
    ground_truth_attributes = extract_lesion_attributes(filename_filter="*mask*.mha",
                                                        objects_folder=ground_truth_folder, **shared)
    segmentation_attributes = extract_lesion_attributes(filename_filter="*segmentation*.mha",
                                                        objects_folder=segmentation_folder, **shared)
    return ground_truth_attributes, segmentation_attributes


def process_experiment(experiment_path: Path,
                       ground_truth_folder: Path,
                       segmentation_folder: Path
                       ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    ground_truth_attributes, segmentation_attributes = [], []
    for fold_run_folder in experiment_path.iterdir():
        if (not fold_run_folder.match("version_*")) or not fold_run_folder.is_dir():
            continue

        input_folder = fold_run_folder / "segmentations"
        attributes = process_fold_segmentations(input_folder, ground_truth_folder, segmentation_folder)
        ground_truth_attributes.append(attributes[0])
        segmentation_attributes.append(attributes[1])

    return ground_truth_attributes, segmentation_attributes


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("experiment_path")
    arg_parser.add_argument("--scalar_features", default=None)
    args = arg_parser.parse_args()

    experiment_path = Path(args.experiment_path)
    scalar_features_path: str | None = args.scalar_features

    # region Check the experiment path
    if experiment_path.name.startswith("version_"):
        experiment_path = experiment_path.parent
    elif experiment_path.name != "lightning_logs":
        experiment_path = experiment_path / "lightning_logs"

    if not experiment_path.exists():
        raise FileNotFoundError(experiment_path)
    # endregion

    # region Additional paths
    export_folder = experiment_path.parent

    ground_truth_folder = export_folder / "ground_truth_objects"
    segmentation_folder = export_folder / "segmentation_objects"

    ground_truth_folder.mkdir(exist_ok=True)
    segmentation_folder.mkdir(exist_ok=True)
    # endregion

    ground_truth_attributes, segmentation_attributes = process_experiment(experiment_path=experiment_path,
                                                                          ground_truth_folder=ground_truth_folder,
                                                                          segmentation_folder=segmentation_folder)

    # region DICE scores
    dice_scores = compute_average_generalized_dice(ground_truth_folder=ground_truth_folder,
                                                   ground_truth_prefix="mask_",
                                                   ground_truth_suffix="_objects.mha",
                                                   segmentation_folder=segmentation_folder,
                                                   segmentation_prefix="segmentation_",
                                                   segmentation_suffix="_objects.mha")
    dice_scores.loc["Average"] = dice_scores.mean()
    print(dice_scores)
    dice_scores.to_csv(export_folder / "dice.csv")
    # endregion

    # region Lesion attributes
    ground_truth_attributes = pd.concat(ground_truth_attributes, axis="index")
    segmentation_attributes = pd.concat(segmentation_attributes, axis="index")

    ground_truth_attributes.to_csv(export_folder / "ground_truth_attributes.csv")
    segmentation_attributes.to_csv(export_folder / "segmentation_attributes.csv")

    if scalar_features_path is not None:
        scalar_features = pd.read_csv(scalar_features_path, index_col="ScanID")
        segmentation_features = {column: {} for column in scalar_features.columns}
        copied_columns = ["age",
                          #   "bounding_box_center_x","bounding_box_center_y","bounding_box_center_z",
                          #   "center_of_gravity_x","center_of_gravity_y","center_of_gravity_z"
                          ]
        for column in scalar_features.columns:
            if column in copied_columns:
                segmentation_features[column] = dict(scalar_features[column])
                continue

            for sample_id, value in segmentation_attributes[column].items():
                _, patient_id, lesion_id, point_id, _ = sample_id.split("_")
                sample_id = "{}_{}_{}".format(patient_id, lesion_id, point_id)
                segmentation_features[column][sample_id] = value

        segmentation_features = pd.DataFrame(segmentation_features)
        segmentation_features.index.name = scalar_features.index.name
        segmentation_features.to_csv(export_folder / "scalar_features.csv")

    # endregion


if __name__ == "__main__":
    main()
