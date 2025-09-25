import SimpleITK
import numpy as np
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from enum import Enum
from typing import Any

from mindful_subream.dataset.build.build_steps import SubreamBuildStep
from mindful_subream.dataset.build.build_data import SubreamBuildData, SubreamInputData
from mindful_subream.dataset.build.build_logger import SubreamBuildLogger
from mindful_subream.dataset.build.build_steps.roi_extractor import SubreamROIExtractor
from mindful_subream.dataset.roi_extraction import save_rois, find_patient_images


class IncorrectEntryType(Enum):
    MISSING_ENTRY = 0
    MISSING_OPERATOR = 1
    DUPLICATE = 2
    INVERTED = 3
    MISSING_SIDE = 4
    ROW_COUNT = 5
    SIDE_ID = 6


class IncorrectEntry(object):
    def __init__(self,
                 error_type: IncorrectEntryType,
                 patient_id: str,
                 data: Any | None = None) -> None:
        self.error_type = error_type
        self.patient_id = patient_id
        self.data = data

    @staticmethod
    def to_log(errors: list["IncorrectEntry"]) -> str:
        error_types = set([error.error_type for error in errors])
        errors_by_type = {error_type: [error for error in errors
                                       if (error.error_type == error_type)]
                          for error_type in error_types}

        logs = []

        def add_log(name, _log):
            _log = "{} ({})".format(name, _log)
            logs.append(_log)

        for error_type, errors_of_type in errors_by_type.items():
            if error_type in [IncorrectEntryType.MISSING_ENTRY,
                              IncorrectEntryType.MISSING_OPERATOR,
                              IncorrectEntryType.DUPLICATE,
                              IncorrectEntryType.INVERTED]:
                log = ", ".join(error.patient_id for error in errors_of_type)
                add_log(error_type.name, log)

            elif error_type == IncorrectEntryType.MISSING_SIDE:
                errors_by_side = {side: [error for error in errors_of_type if error.data == side]
                                  for side in (0, 1)}
                for side, side_errors in errors_by_side.items():
                    if len(side_errors) > 0:
                        log = ", ".join(error.patient_id for error in side_errors)
                        name = "{} {}".format(error_type.name, side)
                        add_log(name, log)

            elif error_type == IncorrectEntryType.ROW_COUNT:
                log = ", ".join("{} ({} rows)".format(error.patient_id, error.data) for error in errors_of_type)
                add_log(error_type.name, log)

            elif error_type == IncorrectEntryType.SIDE_ID:
                log = ", ".join("{} (sides {})".format(error.patient_id, error.data) for error in errors_of_type)
                add_log(error_type.name, log)

            else:
                raise RuntimeError(error_type)

        return "; ".join(logs)


class SubreamBreastExtractor(SubreamBuildStep):
    def __init__(self,
                 build_data: SubreamBuildData,
                 output_root: Path,
                 logger: SubreamBuildLogger):
        super().__init__(build_data, output_root, logger)

    def run(self) -> None:
        if self.data.breasts_bounding_boxes_path is None:
            self.logger.warn("No bounding boxes data for extracting breasts.")
            return

        patients_ids = self.data.patients_ids

        breasts_bounding_boxes_table = self.preprocess_operator_ids(self.data.breasts_bounding_boxes_table)
        breasts_bounding_boxes, incorrect_data = self.load_breasts_bounding_boxes(patients_ids, breasts_bounding_boxes_table)
        self.warn_incorrect_data(incorrect_data)
        extracted_images, extracted_masks = self.extract_all_breasts(patients_ids, breasts_bounding_boxes, self.data)
        self.save_breasts()

        self.data.breasts_bounding_boxes_table = breasts_bounding_boxes_table
        self.data.breasts_bounding_boxes = breasts_bounding_boxes
        self.data.extracted_breasts_images = extracted_images
        self.data.extracted_breasts_masks = extracted_masks

    @staticmethod
    def preprocess_operator_ids(breasts_bounding_boxes_table: pd.DataFrame) -> pd.DataFrame:
        operator = breasts_bounding_boxes_table["operator"].apply(lambda x: x.replace("_", ""))
        breasts_bounding_boxes_table["operator"] = operator
        return breasts_bounding_boxes_table

    @staticmethod
    def load_breasts_bounding_boxes(patients_ids: list[str],
                                    breasts_bounding_boxes_table: pd.DataFrame,
                                    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, list[IncorrectEntry]]]:
        base_table = breasts_bounding_boxes_table
        patients_with_data = set(base_table["PatientID"].astype(str))

        operators = base_table["operator"].unique()
        data_by_operator: dict[str, pd.DataFrame] = {operator: base_table[base_table["operator"] == operator]
                                                     for operator in operators}
        patient_ids_by_operator: dict[str, pd.Series] = {operator: operator_data["PatientID"].astype(str)
                                                         for operator, operator_data in data_by_operator.items()}

        breasts_bounding_boxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        incorrect_data: dict[str, list[IncorrectEntry]] = {operator: [] for operator in operators}
        incorrect_data["all"] = []
        for patient_id in patients_ids:
            # region Incorrect entry: Missing entry
            if patient_id not in patients_with_data:
                error = IncorrectEntry(IncorrectEntryType.MISSING_ENTRY, patient_id)
                incorrect_data["all"].append(error)
                continue
            # endregion

            for operator, operator_data in data_by_operator.items():
                patient_rows = patient_ids_by_operator[operator] == patient_id
                patient_data = operator_data[patient_rows]

                # region Incorrect entry: Missing operator
                if len(patient_data) == 0:
                    error = IncorrectEntry(IncorrectEntryType.MISSING_OPERATOR, patient_id)
                    incorrect_data[operator].append(error)
                    continue
                # endregion
                
                # region Incorrect entry: Row count
                if len(patient_data) != 2:
                    error = IncorrectEntry(IncorrectEntryType.ROW_COUNT, patient_id, data=len(patient_data))
                    incorrect_data[operator].append(error)
                    continue
                # endregion

                patient_sides = patient_data["side"]
                # region Incorrect entry: Side ID
                incorrect_sides = patient_sides.apply(lambda x: x not in [0, 1])
                incorrect_sides = patient_sides[incorrect_sides]
                if len(incorrect_sides) > 0:
                    error = IncorrectEntry(IncorrectEntryType.SIDE_ID, patient_id,
                                           data=list(incorrect_sides))
                    incorrect_data[operator].append(error)
                    continue
                # endregion

                patient_side_0 = patient_data[patient_sides == 0]
                patient_side_1 = patient_data[patient_sides == 1]
                # region Incorrect entry: Missing side
                if (len(patient_side_0) == 0) or (len(patient_side_1) == 0):
                    missing_side = 0 if len(patient_side_0) == 0 else 1
                    error = IncorrectEntry(IncorrectEntryType.MISSING_SIDE, patient_id, data=missing_side)
                    incorrect_data[operator].append(error)
                    continue
                # endregion

                (_, side_a_data), (_, side_b_data) = patient_data.iterrows()
                side_0_data, side_1_data = ((side_a_data, side_b_data) if side_a_data["side"] == 0
                                            else (side_b_data, side_a_data))
                bounding_box_0 = SubreamBreastExtractor.convert_breast_bounding_box(side_0_data)
                bounding_box_1 = SubreamBreastExtractor.convert_breast_bounding_box(side_1_data)

                # region Incorrect entry: Duplicate
                if (bounding_box_0 == bounding_box_1).all():
                    error = IncorrectEntry(IncorrectEntryType.DUPLICATE, patient_id)
                    incorrect_data[operator].append(error)
                    continue
                # endregion

                # region Incorrect entry: Inverted
                if bounding_box_0[0][0] < 0.0:
                    error = IncorrectEntry(IncorrectEntryType.INVERTED, patient_id)
                    incorrect_data[operator].append(error)
                    # no need to continue, we can fix
                    bounding_box_0, bounding_box_1 = bounding_box_1, bounding_box_0
                # endregion

                bounding_boxes_id = "{}_{}".format(patient_id, operator)
                breasts_bounding_boxes[bounding_boxes_id] = (bounding_box_0, bounding_box_1)

        return breasts_bounding_boxes, incorrect_data
    
    def warn_incorrect_data(self, incorrect_data: dict[str, list[IncorrectEntry]]) -> None:
        for operator, operator_errors in incorrect_data.items():
            if len(operator_errors) > 0:
                log_start = ("Warning for all operators" if operator == "all" else
                             "Warning for operator `{}`".format(operator))
                operator_log = "BreastExtractor - {}: {}".format(log_start, IncorrectEntry.to_log(operator_errors))
                self.warn(operator_log)

    @staticmethod
    def convert_breast_bounding_box(side_data: pd.Series) -> np.ndarray:
        axes = ("x", "y", "z")
        center = [side_data["center_{}".format(axis)] for axis in axes]
        length = [side_data["length_{}".format(axis)] for axis in axes]
        bounding_box = np.stack([center, length], axis=0)
        return bounding_box

    @staticmethod
    def extract_all_breasts(patients_ids: list[str],
                            breasts_bounding_boxes: dict[str, tuple[np.ndarray, np.ndarray]],
                            build_data: SubreamInputData,
                            subtracted: bool | None = None,
                            ) -> tuple[dict[str, list[SimpleITK.Image]], dict[str, list[SimpleITK.Image]]]:
        if subtracted is None:
            if isinstance(build_data, SubreamBuildData):
                subtracted = build_data.use_sub
            else:
                subtracted = False

        progress_bar = tqdm(patients_ids, desc="Extracting Breasts...")

        extracted_images: dict[str, list[SimpleITK.Image]] = {}
        extracted_masks: dict[str, list[SimpleITK.Image]] = {}

        for bounding_boxes_id, bounding_boxes in breasts_bounding_boxes.items():
            patient_id, operator_id = bounding_boxes_id.split("_")
            description = "Extracting Breasts... (Patient {}, Operator {})".format(patient_id, operator_id)
            progress_bar.set_description(description)

            try:
                patient_images = build_data.read_patient_images(patient_id, subtracted=subtracted)
                patient_mask = build_data.get_aggregated_patient_mask(patient_id)

                for side_id, bounding_box in enumerate(bounding_boxes):
                    breast_id = "{}_{}".format(bounding_boxes_id, side_id)
                    center, length = bounding_box

                    extracted_images[breast_id] = SubreamROIExtractor.extract_roi(patient_images, center, length)
                    if patient_mask is not None:
                        extracted_masks[breast_id] = SubreamROIExtractor.extract_roi([patient_mask], center,
                                                                                     length, use_nearest=True)
            except KeyboardInterrupt:
                # Catching KeyboardInterrupt to stop the process earlier
                break

        return extracted_images, extracted_masks

    def save_breasts(self) -> None:
        extracted_breasts_images_paths = save_rois(output_images_folder=self.get_images_output_path("whole_breast"),
                                                   rois=self.data.extracted_breasts_images,
                                                   image_names="phase")
        self.data.extracted_breasts_images_paths = extracted_breasts_images_paths

        extracted_breasts_masks_paths = save_rois(output_images_folder=self.get_masks_output_path("whole_breast"),
                                                  rois=self.data.extracted_breasts_masks,
                                                  image_names="mask")
        self.data.extracted_breasts_masks_paths = extracted_breasts_masks_paths


def main():
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_image_folder", type=str, required=True)
    arg_parser.add_argument("--aggregated_masks_filepaths", type=str, required=True)
    arg_parser.add_argument("--output_image_folder", type=str, required=True)
    arg_parser.add_argument("--breasts_bounding_boxes_path", type=str, required=True)
    arg_parser.add_argument("--verbose", action="store_true")
    args = arg_parser.parse_args()

    input_image_folder = Path(args.input_image_folder)
    aggregated_masks_filepaths_root = Path(args.aggregated_masks_filepaths)
    output_image_folder = Path(args.output_image_folder)
    breasts_bounding_boxes_path = Path(args.breasts_bounding_boxes_path)
    verbose: bool = args.verbose

    # region Find images
    patient_paths = find_patient_images(input_image_folder)
    patients_ids = list(patient_paths.keys())
    if verbose:
        print("Found {} patients: {}".format(len(patient_paths), list(patient_paths)))

    patient_images_paths = {
        patient_id: list(patient_path.glob("*.mha")) if patient_path.is_dir() else [patient_path]
        for patient_id, patient_path in patient_paths.items()
    }

    aggregated_masks_filepaths = {
        patient_id: aggregated_masks_filepaths_root / "{}.mha".format(patient_id)
        for patient_id in patients_ids
    }
    # endregion

    # region Prepare build data
    build_data = SubreamInputData(patients_ids=patients_ids,
                                  breasts_bounding_boxes_path=breasts_bounding_boxes_path)
    build_data.patients_images_paths = patient_images_paths
    build_data.aggregated_masks_filepaths = aggregated_masks_filepaths
    breasts_bounding_boxes_table = build_data.breasts_bounding_boxes_table
    # endregion

    # region Run extraction
    breasts_bounding_boxes_table = SubreamBreastExtractor.preprocess_operator_ids(breasts_bounding_boxes_table)

    breasts_bounding_boxes, _ = SubreamBreastExtractor.load_breasts_bounding_boxes(build_data.patients_ids, 
                                                                                   breasts_bounding_boxes_table)
    
    extracted_images, extracted_masks = SubreamBreastExtractor.extract_all_breasts(build_data.patients_ids, 
                                                                                   breasts_bounding_boxes, 
                                                                                   build_data)
    # endregion

    # region Save
    output_image_folder.mkdir(parents=True, exist_ok=True)
    images_output_path = output_image_folder / "images"
    save_rois(output_images_folder=images_output_path,
              rois=extracted_images,
              image_names="phase")

    masks_output_path = output_image_folder / "masks"
    save_rois(output_images_folder=masks_output_path,
              rois=extracted_masks,
              image_names="mask")
    # endregion

if __name__ == "__main__":
    main()
