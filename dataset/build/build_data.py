import SimpleITK
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from mindful_core.utils.misc import load_table, load_json
from mindful_core.utils.parsing import parse_last_number

from mindful_subream.dataset.build.build_logger import SubreamBuildLogger
from mindful_subream.dataset.roi_extraction import get_patient_folders
from mindful_subream.scripts.check_rois import TYPE_TO_LABEL


def optional_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path)


class SubreamInputData(object):
    def __init__(self,
                 patients_ids: list[str] | None = None,
                 data_breast_table_path: str | Path | None = None,
                 patients_clinical_data_path: str | Path | None = None,
                 masks_folder: str | Path | None = None,
                 masks_roi_points_folder: str | Path | None = None,
                 root_patient_images_folder: str | Path | None = None,
                 masks_data_table_path: str | Path | None = None,
                 breasts_bounding_boxes_path: str | Path | None = None,
                 ):

        self.patients_ids = patients_ids

        # region Tables
        self.data_breast_table_path = optional_path(data_breast_table_path)
        self.data_breast_table: pd.DataFrame = load_table(self.data_breast_table_path)
        if self.data_breast_table is not None:
            valid_lesions = self.data_breast_table["type"].apply(self.has_label)
            self.data_breast_table = self.data_breast_table[valid_lesions]

        self.masks_data_table_path = optional_path(masks_data_table_path)
        self.masks_data_table: pd.DataFrame | None = load_table(masks_data_table_path)

        self.patients_clinical_data_path = optional_path(patients_clinical_data_path)
        self.patients_clinical_data: pd.DataFrame = load_table(self.patients_clinical_data_path,
                                                               index_col="Code", sheet_name="clinical")

        self.breasts_bounding_boxes_path = optional_path(breasts_bounding_boxes_path)
        self.breasts_bounding_boxes_table = load_table(self.breasts_bounding_boxes_path)
        # endregion

        # region Masks
        self.masks_folder = optional_path(masks_folder)
        # - Masks paths by lesion
        self.masks_filepaths: dict[str, Path] | None = None
        # - Aggregated masks paths by patient
        self.aggregated_masks_filepaths: dict[str, Path] | None = None
        # endregion

        # region TIC data
        self.tic_data: pd.DataFrame | None = None
        # endregion

        # region Whole Images
        self.root_patient_images_folder: Path | None = optional_path(root_patient_images_folder)
        self.patients_images_paths: dict[str, list[Path]] | None = None
        self.subtracted_images_paths: dict[str, list[Path]] | None = None
        # endregion

        # region ROIs
        self.extracted_roi_paths: dict[str, list[Path]] | None = None
        self.extracted_rois: dict[str, list[SimpleITK.Image]] | None = None

        # region ROI Points
        self.masks_roi_points_folder: Path | None = optional_path(masks_roi_points_folder)
        self.masks_roi_points_filepaths: dict[str, Path] | None = None
        # - Lesion ROI center per lesion id
        self.masks_roi_points: dict[str, np.ndarray] | None = None
        # endregion

        # region ROI masks
        self.extracted_agg_masks_paths: dict[str, list[Path]] | None = None
        self.extracted_agg_masks: dict[str, list[SimpleITK.Image]] | None = None
        # endregion
        # endregion

        # region MIPs (from ROIs)
        self.extracted_mip_paths: dict[str, Path] | None = None
        self.extracted_mips: dict[str, SimpleITK.Image] | None = None
        # endregion

        # region Whole breasts (images/masks/bounding boxes)
        self.breasts_bounding_boxes: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        self.extracted_breasts_images_paths: dict[str, list[Path]] | None = None
        self.extracted_breasts_images: dict[str, list[SimpleITK.Image]] | None = None
        self.extracted_breasts_masks_paths: dict[str, list[Path]] | None = None
        self.extracted_breasts_masks: dict[str, list[SimpleITK.Image]] | None = None
        # endregion

    def read_patient_images(self,
                            patient_id: str,
                            subtracted: bool,
                            ) -> list[SimpleITK.Image]:
        if subtracted:
            if self.subtracted_images_paths is None:
                raise RuntimeError("Paths to subtracted images are unknown.")
            paths = self.subtracted_images_paths[patient_id]
        else:
            paths = self.patients_images_paths[patient_id]

        return [SimpleITK.ReadImage(path.as_posix(), outputPixelType=SimpleITK.sitkFloat32) for path in paths]

    def get_aggregated_patient_mask(self, patient_id: str) -> SimpleITK.Image | None:
        if patient_id not in self.aggregated_masks_filepaths:
            return None
        return SimpleITK.ReadImage(self.aggregated_masks_filepaths[patient_id].as_posix())

    def discard_patients_ids(self, patients_ids: list[str]) -> None:
        self.patients_ids = list(filter(lambda _id: _id not in patients_ids, self.patients_ids))

    @staticmethod
    def has_label(lesion_type: Any) -> bool:
        return isinstance(lesion_type, str) and (len(lesion_type) > 0)


class SubreamBuildData(SubreamInputData):
    def __init__(self,
                 config: dict[str, Any],
                 data_breast_table_path: str | Path,
                 patients_clinical_data_path: str | Path | None,
                 masks_folder: Path,
                 masks_roi_points_folder: str | Path,
                 root_patient_images_folder: str | Path,
                 logger: SubreamBuildLogger,
                 datestamp: str,
                 masks_data_table_path: str | Path | None = None,
                 breasts_bounding_boxes_path: str | Path | None = None,
                 normalize_subtracted: bool = True,
                 resume_from: str | Path | None = None,
                 reference_folds: str | Path = None
                 ):
        super().__init__(data_breast_table_path=data_breast_table_path,
                         patients_clinical_data_path=patients_clinical_data_path,
                         masks_folder=masks_folder,
                         masks_roi_points_folder=masks_roi_points_folder,
                         root_patient_images_folder=root_patient_images_folder,
                         masks_data_table_path=masks_data_table_path,
                         breasts_bounding_boxes_path=breasts_bounding_boxes_path)

        self.config = config
        self.logger = logger
        self.datestamp = datestamp
        self.normalize_subtracted = normalize_subtracted

        if self.masks_data_table is not None:
            self.check_labels_references(self.data_breast_table, self.masks_data_table)

        self.folds: dict[str, dict[str, dict[str, list[Path]]]] = {}

        self.resume_from_path = resume_from
        self.resume_from = load_json(resume_from) if resume_from is not None else {}

        self.reference_folds = Path(reference_folds) if reference_folds is not None else None

    def check_labels_references(self, *tables: pd.DataFrame) -> None:
        reference_labels: pd.Series | None = None
        no_problem_found = True
        for table in tables:
            lesion_labels = self.get_lesion_labels_by_id(table)

            if reference_labels is None:
                reference_labels = lesion_labels
            else:
                in_reference: pd.Series = lesion_labels.index.map(lambda lesion_id: lesion_id in reference_labels)
                in_new: pd.Series = reference_labels.index.map(lambda lesion_id: lesion_id in lesion_labels)
                if not in_reference.all():
                    missing_ids = ", ".join(lesion_labels.index[~in_reference])
                    self.logger.warn("Missing IDs from reference: [{}]".format(missing_ids))
                    no_problem_found = False

                if not in_new.all():
                    missing_ids = ", ".join(reference_labels.index[~in_new])
                    self.logger.warn("Missing IDs in reference: [{}]".format(missing_ids))
                    no_problem_found = False

                lesion_labels = lesion_labels[reference_labels.index]
                matching_labels = lesion_labels.eq(reference_labels)
                if not matching_labels.all():
                    mismatching_lesions = ", ".join(lesion_labels[~matching_labels].index)
                    self.logger.warn(
                        "Mismatching labels found between tables for lesions [{}].".format(mismatching_lesions))
                    no_problem_found = False

        if no_problem_found:
            self.logger.log("SUCCESS: No problem found when checking labels in the provided tables.")

    # region Get lesion data (ids, labels, ...)
    # region Get lesion ids from table

    def get_lesion_ids_from_table(self, table: pd.DataFrame = None) -> pd.Series:
        if table is None:
            table = self.data_breast_table

        # noinspection PyPackages
        if ("Code" in table) and ("n° lésion" in table):
            keep_rows = table.apply(SubreamBuildData.get_keep_data_breast_row, axis=1)
            table = table[keep_rows]
            lesion_ids = table.apply(SubreamBuildData.get_data_breast_lesion_id, axis=1)
        elif "mask_filename" in table:
            lesion_ids = table["mask_filename"].apply(SubreamBuildData.get_mask_lesion_id)
        else:
            raise RuntimeError("Unknown data table format. Columns in table: {}".format(table.columns))

        if not lesion_ids.is_unique:
            duplicates = ", ".join(lesion_ids[lesion_ids.duplicated(keep="first")])
            error_message = "Found duplicated lesion id(s) in table: [{}] (with columns {})." \
                .format(duplicates, list(table.columns))
            self.logger.error(error_message, RuntimeError)
        return lesion_ids

    @staticmethod
    def get_data_breast_lesion_id(row: pd.Series) -> str:
        patient_code = int(row["Code"])
        lesion_index = int(row["n° lésion"])
        return "{}_{}".format(patient_code, lesion_index)

    @staticmethod
    def get_keep_data_breast_row(row: pd.Series) -> bool:
        code = row["Code"]
        lesion_id = row["n° lésion"]
        return not (np.isnan(code) or np.isnan(lesion_id))

    @staticmethod
    def get_mask_lesion_id(mask_filepath: str | Path) -> str:
        mask_filename = Path(mask_filepath).stem
        patient_id, *_, lesion_id, _ = mask_filename.split("_")
        patient_id = SubreamBuildData.remove_patient_prefix(patient_id)
        return "{}_{}".format(patient_id, lesion_id)

    @staticmethod
    def remove_patient_prefix(patient_id: str | int) -> int:
        if not isinstance(patient_id, str):
            return patient_id

        return parse_last_number(patient_id)

    # endregion

    # region Get lesion labels

    def get_lesion_labels_from_table(self, table: pd.DataFrame = None) -> pd.Series:
        if table is None:
            table = self.data_breast_table

        # noinspection PyPackages
        if ("Code" in table) and ("n° lésion" in table):
            return table["type"].apply(self.get_lesion_label)

        elif "mask_filename" in table:
            from_type: pd.Series = table["type"].apply(self.get_lesion_label)
            from_mask_filename = table["mask_filename"].apply(self.get_lesion_label)
            non_matching_labels: pd.Series = ~from_type.eq(from_mask_filename)
            if non_matching_labels.any():
                error_table = table[non_matching_labels][["mask_filename", "type"]]
                error_table.to_csv(self.logger.log_path / "mask_label_table_mismatches.csv", index=False)
                self.logger.warn("Mismatch between labels in mask-labels type table.")
            else:
                self.logger.log("SUCCESS: Mask filenames and types match.")
            return from_type
        else:
            raise RuntimeError("Unknown data table format.")

    @staticmethod
    def get_lesion_label(lesion_type: str | Path | int) -> str:
        reference = ["B", "M", "G"]
        if isinstance(lesion_type, int) and (2 >= lesion_type >= 0):
            return reference[lesion_type]

        elif isinstance(lesion_type, Path) or lesion_type.endswith(".mha"):
            mask_filename = Path(lesion_type).stem
            return mask_filename.split("_")[-1]

        elif lesion_type in reference:
            return lesion_type

        elif lesion_type in TYPE_TO_LABEL:
            return TYPE_TO_LABEL[lesion_type]

        else:
            raise ValueError("Could not parse lesion type ``".format(lesion_type))

    @staticmethod
    def get_lesion_label_value(lesion_label: str) -> int:
        # Background : 0
        # Benign : 1
        # Lymphnodes : 2
        # Malignant : 3
        return ["background", "b", "g", "m"].index(lesion_label.lower())

    # endregion

    def get_lesion_labels_by_id(self, table: pd.DataFrame = None) -> pd.Series:
        lesion_ids = self.get_lesion_ids_from_table(table)
        lesion_labels = self.get_lesion_labels_from_table(table)
        series = pd.Series(list(lesion_labels), index=list(lesion_ids))
        return series

    # endregion

    def get_patient_folders(self) -> dict[str, Path]:
        return get_patient_folders(self.root_patient_images_folder)

    def has_patient_images(self,
                           patient_id: str,
                           subtracted: bool = None
                           ) -> bool:
        if subtracted is None:
            subtracted = self.use_sub

        if subtracted:
            if self.subtracted_images_paths is None:
                return False
            return patient_id in self.subtracted_images_paths
        else:
            return patient_id in self.patients_images_paths

    def read_patient_images(self,
                            patient_id: str,
                            subtracted: bool = None,
                            ) -> list[SimpleITK.Image]:
        if subtracted is None:
            subtracted = self.use_sub

        return super().read_patient_images(patient_id, subtracted)

        # region

    # endregion

    # region Properties (from config)
    @property
    def use_sub(self) -> bool:
        return self.config["use_sub"]

    @property
    def roi_size(self) -> float:
        return self.config["roi_size"]

    @property
    def mip_dim(self) -> int:
        return self.config["mip_dim"]

    @property
    def mip_augmentation_count(self) -> int:
        return self.config["mip_augmentation_count"]

    # region Folds
    @property
    def folds_count(self) -> int:
        return self.config["folds_count"]

    @property
    def stochastic_folds_iterations(self) -> int:
        return self.config["stochastic_folds_iterations"]

    @property
    def test_folds_count(self) -> int:
        return self.config["test_folds_count"]

    @property
    def validation_folds_count(self) -> int:
        return self.config["validation_folds_count"]

    # endregion

    @property
    def readme_templates_folder(self) -> Path:
        if "readme_templates" in self.config:
            str_path = self.config["readme_templates_folder"]
        else:
            str_path = "../dataset/build/readme_templates"
        return Path(str_path)

    @property
    def max_patients(self) -> int | None:
        if "max_patients" not in self.config:
            return None

        max_patients = self.config["max_patients"]
        if (not isinstance(max_patients, int)) or (max_patients <= 0):
            return None

        return max_patients
    # endregion
