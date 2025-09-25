# ##################################################################### #
# Extraction of Time Intensity Curve (TIC) from 4D ultrafast masks      #
# ##################################################################### #

import numpy as np
import SimpleITK
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Optional

from mindful_subream.dataset.build.build_steps.build_step import SubreamBuildStep
from mindful_subream.dataset.build.build_data import SubreamBuildData
from mindful_subream.dataset.build.build_logger import SubreamBuildLogger


class SubreamTICExtractor(SubreamBuildStep):
    def __init__(self,
                 build_data: SubreamBuildData,
                 output_root: Path,
                 logger: SubreamBuildLogger):
        super(SubreamTICExtractor, self).__init__(
            build_data, output_root, logger)
        self.name = "SubreamTICExtractor"
        self.description = "Extraction of Time Intensity Curve (TIC) from 4D ultrafast masks"
        self.tic_phases_columns: Optional[list[str]] = None

    def run(self):
        self.compute_tic_values()
        self.compute_max_slope()
        self.compute_autic()

        clinical_data_folder = self.get_output_path("clinical_data")
        self.data.tic_data.to_csv(clinical_data_folder / "tic_data.csv")

    @staticmethod
    def read_masks_images(masks_filepaths: dict[str, Path]
                          ) -> dict[str, SimpleITK.Image]:
        return {mask_id: SimpleITK.ReadImage(mask_filepath.as_posix())
                for mask_id, mask_filepath in masks_filepaths.items()}

    def compute_tic_values(self) -> None:
        tic_values: dict[str, list[float]] = {}
        phase_count: Optional[int] = None

        masks_filepaths = self.data.masks_filepaths
        masks_filepaths_by_lesion = self.as_patient_lesion_dict(masks_filepaths)

        for patient_id in tqdm(self.data.patients_ids, desc=self.description + "... "):
            if patient_id not in masks_filepaths_by_lesion:
                continue

            lesion_masks_filepaths = masks_filepaths_by_lesion[patient_id]
            lesion_masks = self.read_masks_images(lesion_masks_filepaths)
            patient_images = self.data.read_patient_images(patient_id, subtracted=True)

            for lesion_index, lesion_mask in lesion_masks.items():
                lesion_id = "{}_{}".format(patient_id, lesion_index)
                lesion_tic = []
                
                lesion_start, lesion_size = SubreamTICExtractor.compute_mask_bounding_box(lesion_mask)
                lesion_mask = SimpleITK.RegionOfInterest(lesion_mask, lesion_size, lesion_start)
                mask_array = SimpleITK.GetArrayFromImage(lesion_mask).astype(np.float32)

                for phase in patient_images:
                    phase = SimpleITK.RegionOfInterest(phase, lesion_size, lesion_start)
                    phase_array = SimpleITK.GetArrayFromImage(phase).astype(np.float32)

                    masked_image = phase_array * mask_array
                    phase_intensity: float = masked_image.sum() / mask_array.sum()

                    lesion_tic.append(phase_intensity)
                tic_values[lesion_id] = lesion_tic

            if phase_count is None:
                phase_count = len(patient_images)

        self.tic_phases_columns = ["phase_{}".format(i) for i in range(phase_count)]
        tic_data = pd.DataFrame.from_dict(tic_values, orient="index", columns=self.tic_phases_columns)
        tic_data.index.name = "LesionID"

        self.data.tic_data = tic_data

    def compute_max_slope(self) -> None:
        """
        ## Summary
            - Computes the max slope of the TIC.

        ### Results
            - Results are stored in `self.data.tic_data` container under "max_slope"
        """

        if len(self.tic_phases_columns) < 2:
            return

        deltas = []
        for i in range(len(self.tic_phases_columns) - 1):
            column_a = self.tic_phases_columns[i]
            column_b = self.tic_phases_columns[i + 1]
            delta = self.data.tic_data[column_b] - self.data.tic_data[column_a]
            delta.name = "delta_{}".format(i)
            deltas.append(delta)

        deltas = pd.concat(deltas, axis=1)
        max_slope = deltas.max(axis=1)
        self.data.tic_data["max_slope"] = max_slope

    def compute_autic(self) -> None:
        """
        ## Summary
            - Computes the Area Under the Time Intensity Curve (AUTIC).
            Assumes equal time between phases.

        ### Results
            - Results are stored in `self.data.tic_data` container under "autic"
        """
        if len(self.tic_phases_columns) < 2:
            return

        first_phase_column, *center_phases_columns, last_phase_column = self.tic_phases_columns
        first_phase_tic = self.data.tic_data[first_phase_column]
        center_phases_tics = self.data.tic_data[center_phases_columns]
        last_phase_tic = self.data.tic_data[last_phase_column]

        tic_auc = 0.5 * (first_phase_tic + last_phase_tic) + center_phases_tics.sum(axis=1)
        self.data.tic_data["autic"] = tic_auc

    @staticmethod
    def compute_mask_bounding_box(mask: SimpleITK.Image) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        mask = mask > 0
        mask = SimpleITK.Cast(mask, pixelID=SimpleITK.sitkUInt8)

        label_shape_statistics_filter = SimpleITK.LabelShapeStatisticsImageFilter()
        label_shape_statistics_filter.Execute(mask)

        idx_x, idx_y, idx_z, size_x, size_y, size_z = label_shape_statistics_filter.GetBoundingBox(label=1)

        return (idx_x, idx_y, idx_z), (size_x, size_y, size_z)
    