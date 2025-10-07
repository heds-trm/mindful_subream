import pandas as pd
from pathlib import Path

from mindful_core.utils.parsing import parse_last_number
from mindful_core.utils.data_constants import SCAN_ID

from mindful_subream.dataset.build.build_steps.build_step import SubreamBuildStep
from mindful_subream.dataset.build.build_data import SubreamBuildData
from mindful_subream.dataset.build.build_logger import SubreamBuildLogger


class SubreamClinicalDataBuilder(SubreamBuildStep):
    def __init__(self,
                 build_data: SubreamBuildData,
                 output_root: Path,
                 logger: SubreamBuildLogger):
        super(SubreamClinicalDataBuilder, self).__init__(build_data, output_root, logger)
        self.known_scalars = ["age"]

    def run(self) -> None:
        if self.data.extracted_roi_paths is None:
            return

        base_categorical_data, base_scalar_data = self.get_clinical_data()
        lesion_attributes = self.get_lesion_attributes()
        categorical_data, scalar_data = self.join_data(base_categorical_data,
                                                       base_scalar_data,
                                                       lesion_attributes,
                                                       self.data.tic_data)
        self.save_data(categorical_data, scalar_data)

    def get_clinical_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits clinical data based on known scalars entries and return two data frame, the first one containing 
            categorial data and the second one containing scalar data.
        ### Returns
            - tuple[pd.DataFrame, pd.DataFrame]
                - Two pandas dataframe (categorial data, scalar data).
        """
        if self.data.patients_clinical_data is None:
            clinical_data = pd.DataFrame()
        else:
            clinical_data = self.data.patients_clinical_data.copy()
            clinical_data.index = clinical_data.index.map(str)

        categorical_data = clinical_data[[column for column in clinical_data.columns
                                          if column.lower() not in self.known_scalars]]

        scalar_data = clinical_data[[column for column in clinical_data.columns
                                     if column.lower() in self.known_scalars]]

        return categorical_data, scalar_data

    def get_lesion_attributes(self) -> pd.DataFrame:
        def get_lesion_id(mask_filename: str) -> str:
            mask_filepath = Path(mask_filename)
            patient_id, *_, lesion_index, _ = mask_filepath.name.split("_")
            patient_id = parse_last_number(patient_id)
            return "{}_{}".format(patient_id, lesion_index)

        lesion_attributes = self.data.masks_data_table.copy()
        lesion_attributes_index = lesion_attributes["mask_filename"].apply(get_lesion_id)
        lesion_attributes = lesion_attributes.drop(["mask_filename", "type"], axis="columns")
        lesion_attributes.index = lesion_attributes_index

        return lesion_attributes

    def join_data(self,
                  base_categorical_data: pd.DataFrame,
                  base_scalar_data: pd.DataFrame,
                  lesion_attributes: pd.DataFrame,
                  tic_data: pd.DataFrame | None,
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parameters:
            - base_categorial_data:
                - (Categorial) data taken for each patient
            - base_scalar_data:
                - (Scalar) data taken for each patient (eg. age)
            - lesion_attributes:
                - (Scalar) data taken for each lesion (eg. volume)
            - tic_data:
                - Time Intensity Curve (TIC) from 4D ultrafast roi/masks (for each lesion).
        """
        categorical_data_rows = {}
        scalar_data_rows = {}
        lesion_attributes_rows = {}
        tic_data_rows = {}

        # region Remap to shared index
        for point_id in self.data.extracted_roi_paths:
            patient_id, lesion_index, _ = point_id.split("_")
            lesion_id = "{}_{}".format(patient_id, lesion_index)

            if patient_id in base_categorical_data.index:
                categorical_data_rows[point_id] = base_categorical_data.loc[patient_id]
                scalar_data_rows[point_id] = base_scalar_data.loc[patient_id]

                if lesion_id in lesion_attributes.index:
                    lesion_attributes_rows[point_id] = lesion_attributes.loc[lesion_id]

                if (tic_data is not None) and (lesion_id in tic_data.index):
                    tic_data_rows[point_id] = tic_data.loc[lesion_id]
        # endregion

        categorical_data = pd.DataFrame.from_dict(categorical_data_rows, orient="index")

        scalar_data = pd.DataFrame.from_dict(scalar_data_rows, orient="index")
        lesion_attributes_data = pd.DataFrame.from_dict(lesion_attributes_rows, orient="index")
        scalar_frames = [scalar_data, lesion_attributes_data]

        if tic_data is not None:
            tic_data = pd.DataFrame.from_dict(tic_data_rows, orient="index")
            scalar_frames.append(tic_data)

        scalar_data = pd.concat(scalar_frames, axis=1)

        return categorical_data, scalar_data

    def save_data(self, categorical_data: pd.DataFrame, scalar_data: pd.DataFrame) -> None:
        categorical_data.index.name = SCAN_ID
        scalar_data.index.name = SCAN_ID

        clinical_data_folder = self.get_output_path("clinical_data")
        categorical_filepath = clinical_data_folder / "categorical_features.csv"
        scalar_filepath = clinical_data_folder / "scalar_features.csv"

        categorical_data.to_csv(categorical_filepath)
        scalar_data.to_csv(scalar_filepath)
