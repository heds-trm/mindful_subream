from pathlib import Path
from abc import ABC, abstractmethod
from typing import Iterator, TypeVar

from mindful_subream.dataset.build.build_data import SubreamBuildData
from mindful_subream.dataset.build.build_logger import SubreamBuildLogger

_T = TypeVar("_T")


class SubreamBuildStep(ABC):
    def __init__(self,
                 build_data: SubreamBuildData,
                 output_root: Path,
                 logger: SubreamBuildLogger):
        self.data = build_data
        self.output_root = output_root
        self.logger = logger

    # region Hooks
    def prepare(self) -> None:
        pass

    def run_for_patient(self, patient_id: str) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError("`run` must be implemented in sub-classes.")

    def clean_up(self) -> None:
        pass

    # endregion

    # region Helpers / Utils
    @staticmethod
    def split_id(data_dict: dict[str, _T]) -> Iterator[tuple[list[str], _T]]:
        for key, value in data_dict.items():
            yield key.split("_"), value

    @staticmethod
    def as_patient_lesion_dict(data_dict: dict[str, _T]) -> dict[str, dict[str, _T]]:
        result = {}
        for (patient_id, lesion_index), value in SubreamBuildStep.split_id(data_dict):
            if patient_id not in result:
                result[patient_id] = {}
            result[patient_id][lesion_index] = value
        return result

    @staticmethod
    def as_patient_lesion_point_dict(data_dict: dict[str, _T]) -> dict[str, dict[str, dict[str, _T]]]:
        result = {}
        for (patient_id, lesion_index, point_index), value in SubreamBuildStep.split_id(data_dict):
            if patient_id not in result:
                result[patient_id] = {}

            if lesion_index not in result[patient_id]:
                result[patient_id][lesion_index] = {}

            result[patient_id][lesion_index][point_index] = value
        return result

    # endregion

    # region Logger
    def log(self, message: str) -> None:
        self.logger.log(message)

    def warn(self, message: str) -> None:
        self.logger.warn(message)

    def error(self, message: str, exception_type: type[Exception]) -> None:
        self.logger.error(message, exception_type)

    # endregion

    # region Get output paths
    def get_output_path(self, folder_name: str) -> Path:
        path = self.output_root / folder_name
        if not path.exists():
            path.mkdir()
        return path

    def get_images_output_path(self, folder_name: str) -> Path:
        return self._get_output_path("images", folder_name)

    def get_masks_output_path(self, folder_name: str) -> Path:
        return self._get_output_path("masks", folder_name)

    def get_folds_output_path(self, folder_name: str) -> Path:
        return self._get_output_path("folds", folder_name)
    
    def _get_output_path(self, parent_name: str, folder_name: str):
        path = self.get_output_path(parent_name) / folder_name
        if not path.exists():
            path.mkdir()
        return path

    # endregion
