# ##################################################################### #
# This script runs all pre-processing steps for the subream dataset     #
# ##################################################################### #
from monai.utils import set_determinism
from pytorch_lightning import seed_everything
from pathlib import Path
import argparse
from datetime import datetime
import os
import shutil

from mindful_core.utils.misc import load_json

from mindful_subream.dataset.build.build_data import SubreamBuildData
from mindful_subream.dataset.build.build_logger import SubreamBuildLogger
from mindful_subream.dataset.build.build_steps import (
    SubreamBuildStep,
    SubreamPatientLoader,
    SubreamGeometryExtractor,
    SubreamTICExtractor,
    SubreamROIExtractor,
    SubreamBreastExtractor,
    # SubreamMIPExtractor,
    SubreamFoldsBuilder,
    SubreamClinicalDataBuilder,
    SubreamReadmeMaker
)
from mindful_subream.dataset.roi_extraction import get_patient_folders

"""
Expected data in JSON config file:
    - data_breast
        Path to the "DataBreast.xlsx" file
    - patients_clinical_data
        Path to the "tab_clinical_data_cleanversion.xlsx" file
    - masks_data
        Path to the lesion attribute file ("all_lesions_attributes_march_2023.csv")
    - masks
        Path to the root of mask files (*.mha)
    - masks_roi_points
        Path to the root of mask coordinates files (.*txt)
    - breasts_bbox
        Path to the breasts bounding boxes coordinates ("bbox_stats_octobre2024_v2.csv")
    - patients_images
        Path to the root of patient images (*.mha)

    - output_path
        Output root for built data
    - roi_size
        Size of extracted ROIs (in mm)
    - use_sub
        If true, subtracts all phases with the first phase
    - mip_dim
        The dimension to reduce for computing MIP images
    - mip_augmentation_count
        The number of augmentations to perform when building MIPs (prior to reduction)

    - folds_count
        The number of partitions (and folds) to split data into for cross-validation
    - test_folds_count
        The number of partitions per fold used for the test set
    - validation_folds_count
        The number of partitions per fold used for the validation set
    - stochastic_folds_iterations
        The number of iterations to perform when making folds/partitions in order to balance prevalence 
            across partitions
"""


class SubreamMasterBuilder(object):
    def __init__(self, config: str | Path | dict) -> None:
        if isinstance(config, dict):
            self.config_path = None
            self.config = config
        else:
            self.config_path = Path(config)
            self.config = load_json(config)
        self.datestamp = datetime.now().date()

        self.seed = self.config.get("seed", 3930787959)
        set_determinism(self.seed)
        seed_everything(self.seed)

        self._output_path = None
        self._output_points_folder = None
        self._output_images_folder = None
        self._output_mips_folder = None
        self._output_folds_folder = None
        self._log_path = None
        self._main_log_file = None

        self.logger = SubreamBuildLogger(self.log_path)
        self.data = SubreamBuildData(self.config,
                                     data_breast_table_path=self.config["data_breast"],
                                     patients_clinical_data_path=self.config.get("patients_clinical_data", None),
                                     masks_folder=self.config["masks"],
                                     masks_roi_points_folder=self.config["masks_roi_points"],
                                     root_patient_images_folder=self.config["patients_images"],

                                     logger=self.logger,
                                     datestamp=str(self.datestamp),

                                     masks_data_table_path=self.config.get("masks_data", None),
                                     breasts_bounding_boxes_path=self.config.get("breasts_bbox", None),
                                     normalize_subtracted=self.config.get("normalize_subtracted", True),
                                     resume_from=self.config.get("resume_from", None),
                                     reference_folds=self.config.get("reference_folds", None)
                                     )

        self.builders: list[type[SubreamBuildStep]] = [
            SubreamPatientLoader,

            SubreamROIExtractor,
            SubreamGeometryExtractor,
            SubreamBreastExtractor,
            # SubreamMIPExtractor,
            SubreamTICExtractor,

            SubreamFoldsBuilder,
            SubreamClinicalDataBuilder,
            SubreamReadmeMaker
        ]

    def run(self) -> None:
        base_message = "Running Subream dataset builder with the following steps:"
        builders_names = [builder_class.__name__ for builder_class in self.builders]
        steps_message = "{}\t- ".format(os.linesep).join([base_message] + builders_names)
        self.logger.log(steps_message)
        self.logger.log("Seed is: {}".format(self.seed))

        builders: list[SubreamBuildStep] = []
        for builder_class in self.builders:
            builder = builder_class(self.data, self.output_path, self.logger)
            builders.append(builder)

            self.logger.new_line()
            self.logger.log("===== Running builder `{}` =====".format(builder_class.__name__))
            builder.run()

        if self.config.get("clean_up", False):
            for builder in builders:
                builder.clean_up()

    # region Properties
    # region Output folders

    @property
    def output_path(self) -> Path:
        if self._output_path is None:
            self._output_path = self.make_builder_folder("output_path", "subream_build_outputs")
        return self._output_path

    @property
    def output_points_folder(self) -> Path:
        if self._output_points_folder is None:
            self._output_points_folder = self.output_path / "points"
            self._output_points_folder.mkdir()
        return self._output_points_folder

    @property
    def output_images_folder(self) -> Path:
        if self._output_images_folder is None:
            self._output_images_folder = self.output_path / "images"
            self._output_images_folder.mkdir()
        return self._output_images_folder

    @property
    def output_mips_folder(self) -> Path:
        if self._output_mips_folder is None:
            self._output_mips_folder = self.output_path / "mips"
            self._output_mips_folder.mkdir()
        return self._output_mips_folder

    @property
    def output_folds_folder(self) -> Path:
        if self._output_folds_folder is None:
            self._output_folds_folder = self.output_path / "folds"
            self._output_folds_folder.mkdir()
        return self._output_folds_folder

    # endregion

    @property
    def log_path(self) -> Path:
        if self._log_path is None:
            self._log_path = self.make_builder_folder("log_path", "subream_build_logs")
        return self._log_path

    # endregion

    # region Misc
    def make_builder_folder(self, key: str, fallback_name: str) -> Path:
        if key in self.config:
            path = self.config[key]
        else:
            path = self.config_path.parent / fallback_name
        path = Path(path) / str(self.datestamp)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        return path

    def get_patient_folders(self) -> dict[str, Path]:
        return get_patient_folders(Path(self.config["patients_images"]))
    # endregion
