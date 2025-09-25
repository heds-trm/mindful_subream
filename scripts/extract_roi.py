import argparse

from mindful_subream.dataset.roi_extraction import (
    get_points_filepaths,
    points_filepaths_to_short_labels,
    extract_all_rois
)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_image_folder", type=str, required=True)
    arg_parser.add_argument("--points_folder", type=str, required=True)
    arg_parser.add_argument("--output_image_folder", type=str, required=True)
    arg_parser.add_argument("--roi_size", type=float, default=50.0)
    arg_parser.add_argument("--use_sub", default=False)
    args = arg_parser.parse_args()

    input_image_folder = Path(args.input_image_folder)
    points_folder = Path(args.points_folder)
    output_image_folder = Path(args.output_image_folder)
    roi_size = float(args.roi_size)
    use_sub = args.use_sub in ["Y", "y", "Yes", "yes", "on", "True", True]

    points_filepaths = get_points_filepaths(points_folder)
    patients_points = load_all_patients_points(points_filepaths)
    patients_lesion_short_labels = points_filepaths_to_short_labels(points_filepaths)

    patients_images = find_and_load_patient_images(input_image_folder)
    extracted_rois = extract_all_rois(patients_images, patients_points,
                                      use_sub=use_sub, roi_size=roi_size)

    result_paths = save_rois(output_image_folder, extracted_rois, patients_lesion_short_labels)
    print(result_paths)


if __name__ == "__main__":
    main()
