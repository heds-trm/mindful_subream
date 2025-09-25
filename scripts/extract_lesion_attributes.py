import argparse


from mindful_subream.dataset.lesion_attributes_extraction import process_folder


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("folder")
    arg_parser.add_argument("--filename_filter", default=None)
    arg_parser.add_argument("--binarize", action="store_true")
    arg_parser.add_argument("--min_pixel_count", default=200)
    arg_parser.add_argument("--attributes_filter", default=None)
    arg_parser.add_argument("--save_objects", action="store_true")

    args = arg_parser.parse_args()
    root_folder: Path = Path(args.folder)
    filename_filter: str | None = args.filename_filter
    binarize: bool = args.binarize
    save_objects: bool = args.save_objects
    min_pixel_count: int = int(args.min_pixel_count)
    attributes_filter: str | None = args.attributes_filter

    if (root_folder / "lightning_logs").exists():
        root_folder = root_folder / "lightning_logs"

    is_lightning_logs_folder = root_folder.name == "lightning_logs"

    if is_lightning_logs_folder:
        folders = [folder / "segmentations" for folder in root_folder.glob("version_*")]
        folders = [folder for folder in folders if folder.exists()]
    else:
        folders = [root_folder]

    objects_attributes = []
    for folder in folders:
        print("Extracting lesion attributes from segmentations in {}".format(folder))
        fold_objects_attributes = process_folder(folder, filename_filter, binarize, min_pixel_count, attributes_filter,
                                                 save_objects)
        fold_objects_attributes.to_csv(folder / "all_objects_attributes.csv")
        objects_attributes.append(fold_objects_attributes)

    if is_lightning_logs_folder:
        objects_attributes = pd.concat(objects_attributes, axis="index")
        objects_attributes.to_csv(root_folder.parent / "all_objects_attributes.csv")


if __name__ == "__main__":
    main()
