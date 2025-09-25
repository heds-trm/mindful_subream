import pandas as pd
from pathlib import Path

from mindful_subream.dataset.build.build_steps.build_step import SubreamBuildStep
from mindful_subream.dataset.build.build_data import SubreamBuildData
from mindful_subream.dataset.build.build_logger import SubreamBuildLogger
from mindful_subream.dataset.lesion_attributes_extraction import extract_objects_attributes, filter_by_attribute


class SubreamGeometryExtractor(SubreamBuildStep):
    def __init__(self,
                 build_data: SubreamBuildData,
                 output_root: Path,
                 logger: SubreamBuildLogger):
        super(SubreamGeometryExtractor, self).__init__(build_data, output_root, logger)

    def run(self) -> None:
        if self.data.masks_data_table is not None:
            return

        if (self.data.masks_filepaths is None) or (len(self.data.masks_filepaths) == 0):
            raise RuntimeError("Masks are required to compute geometric characteristics of lesions, "
                               "but no masks are available.")

        missing_paths = [path.absolute().as_posix() for path in self.data.masks_filepaths.values() if not path.exists()]
        if len(missing_paths) != 0:
            raise FileNotFoundError("{} masks could not be found: {}".format(len(missing_paths), missing_paths))

        all_roi_attributes: list[pd.DataFrame] = []
        for roi_id, roi_path in self.data.masks_filepaths.items():
            objects_attributes, _ = extract_objects_attributes(roi_path, binarize=False, min_pixel_count=50)

            if objects_attributes.empty:
                raise RuntimeError("Could not detect any object in `{}`".format(roi_path))

            objects_attributes = filter_by_attribute(objects_attributes,
                                                     attributes_filter="volume_and_distance_to_center")

            objects_attributes.index = objects_attributes.index.map(lambda x: "{}_{}".format(roi_id, x))
            all_roi_attributes.append(objects_attributes)

        masks_data_table = pd.concat(all_roi_attributes)
        masks_data_table.to_csv(self.get_masks_output_path("roi") / "masks_data_table.csv")
        self.data.masks_data_table = masks_data_table
