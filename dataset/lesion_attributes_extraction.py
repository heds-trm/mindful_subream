# noinspection PyPep8Naming
import SimpleITK as sitk
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal


def transform_index(image: sitk.Image, point: tuple[int, int, int]) -> tuple[float, float, float]:
    raise NotImplementedError


def extract_objects_attributes(image_path: Path | str,
                               binarize: bool = False,
                               min_pixel_count: int = 200
                               ) -> tuple[pd.DataFrame, sitk.Image]:
    image_path = Path(image_path).as_posix()
    image = sitk.ReadImage(image_path, sitk.sitkInt8)
    center_index = tuple([x // 2 for x in image.GetSize()])
    center = np.asarray(image.TransformIndexToPhysicalPoint(center_index), dtype=np.float32)
    if binarize:
        image = image > 0

    connected_component_filter = sitk.ConnectedComponentImageFilter()
    label_shape_stats_filter = sitk.LabelShapeStatisticsImageFilter()

    objects: sitk.Image = connected_component_filter.Execute(image)
    object_count = connected_component_filter.GetObjectCount()

    label_shape_stats_filter.Execute(objects)

    # region Generic column names
    spatial_dims = ("x", "y", "z")
    bounding_box_center_columns = ["bounding_box_center_{}".format(spatial_dim)
                                   for spatial_dim in spatial_dims]
    bounding_box_size_columns = ["bounding_box_size_{}".format(spatial_dim)
                                 for spatial_dim in spatial_dims]
    center_of_gravity_columns = ["center_of_gravity_{}".format(spatial_dim)
                                 for spatial_dim in spatial_dims]
    ellipsoid_diameter_columns = ["equivalent_ellipsoid_diameter_{}".format(i)
                                  for i in range(1, 4)]
    # endregion

    spacing = objects.GetSpacing()

    objects_attributes = {}
    for i in range(1, object_count + 1):
        pixel_count = label_shape_stats_filter.GetNumberOfPixels(i)
        if pixel_count < min_pixel_count:
            continue

        object_attributes = {}

        # region Bounding box
        bounding_box = label_shape_stats_filter.GetBoundingBox(i)
        bounding_box_start, bounding_box_size = bounding_box[:3], bounding_box[3:]
        bounding_box_start = image.TransformIndexToPhysicalPoint(bounding_box_start)
        bounding_box_center = tuple([(start + size * dim_spacing * 0.5)
                                     for start, size, dim_spacing
                                     in zip(bounding_box_start, bounding_box_size, spacing)])
        for column, value in zip(bounding_box_center_columns, bounding_box_center):
            object_attributes[column] = value

        bounding_box_size = [value * _spacing for value, _spacing in zip(bounding_box_size, spacing)]
        for column, value in zip(bounding_box_size_columns, bounding_box_size):
            object_attributes[column] = value
        # endregion

        object_attributes["volume"] = label_shape_stats_filter.GetPhysicalSize(i)

        # region Centroid / Center of gravity
        centroid = label_shape_stats_filter.GetCentroid(i)
        for column, value in zip(center_of_gravity_columns, centroid):
            object_attributes[column] = value
        # endregion

        object_attributes["elongation"] = label_shape_stats_filter.GetElongation(i)

        # region Ellipsoid diameter
        ellipsoid_diameter = label_shape_stats_filter.GetEquivalentEllipsoidDiameter(i)
        for column, value in zip(ellipsoid_diameter_columns, ellipsoid_diameter):
            object_attributes[column] = value
        # endregion

        object_attributes["flatness"] = label_shape_stats_filter.GetFlatness(i)
        object_attributes["pixel_count"] = pixel_count

        distance_to_center = np.sqrt(np.square(np.asarray(centroid, np.float32) - center).sum())
        object_attributes["distance_to_center"] = distance_to_center

        objects_attributes[i] = object_attributes

    objects_attributes = pd.DataFrame(objects_attributes).transpose()
    objects_attributes.index.name = "centroid_id"
    return objects_attributes, objects


def filter_by_attribute(objects_attributes: pd.DataFrame,
                        attributes_filter: str | Literal["volume_and_distance_to_center"]
                        ) -> pd.DataFrame:
    if attributes_filter.startswith("max_"):
        column = attributes_filter[4:]
        index = objects_attributes[column].idxmax()
        objects_attributes = objects_attributes.loc[[index]]

    elif attributes_filter.startswith("min_"):
        column = attributes_filter[4:]
        index = objects_attributes[column].idxmin()
        objects_attributes = objects_attributes.loc[[index]]

    elif attributes_filter == "volume_and_distance_to_center":
        top_volume = objects_attributes.nlargest(n=3, columns="volume")
        index = top_volume["distance_to_center"].idxmin()
        objects_attributes = objects_attributes.loc[[index]]

    else:
        raise ValueError("{} is an unknown filter".format(attributes_filter))

    return objects_attributes


def filter_objects_image(objects: sitk.Image, object_ids: list[int]) -> sitk.Image:
    if len(object_ids) == 1:
        objects = objects == object_ids[0]

    else:
        filtered_objects = None
        for object_id in object_ids:
            centroid_object = (objects == object_id) * object_id
            if filtered_objects is None:
                filtered_objects = centroid_object
            else:
                filtered_objects = filtered_objects + centroid_object

        objects = filtered_objects

    return objects


def process_folder(folder: Path,
                   filename_filter: str | None = None,
                   binarize: bool = False,
                   min_pixel_count: int = 200,
                   attributes_filter: str | None = None,
                   save_objects: bool = False,
                   objects_folder: str | Path | None = None
                   ) -> pd.DataFrame:
    if save_objects:
        if objects_folder is None:
            objects_folder = folder / "objects"
        elif isinstance(objects_folder, str):
            objects_folder = Path(objects_folder)
        objects_folder.mkdir(exist_ok=True)

    all_objects_attributes: list[pd.DataFrame] = []
    for filepath in folder.iterdir():
        if (filename_filter is not None) and not filepath.match(filename_filter):
            continue
        objects_attributes, objects = extract_objects_attributes(filepath, binarize, min_pixel_count)

        if objects_attributes.empty:
            sitk.WriteImage(objects, objects_folder / "{}_objects.mha".format(filepath.stem))
            continue

        if attributes_filter is not None:
            objects_attributes = filter_by_attribute(objects_attributes, attributes_filter)
            objects = filter_objects_image(objects, list(objects_attributes.index))

        if save_objects:
            sitk.WriteImage(objects, objects_folder / "{}_objects.mha".format(filepath.stem))

        objects_attributes.index = objects_attributes.index.map(lambda x: "{}_{}".format(filepath.stem, x))
        all_objects_attributes.append(objects_attributes)

    return pd.concat(all_objects_attributes)
