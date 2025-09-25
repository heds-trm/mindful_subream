import pandas as pd
from pathlib import Path
import argparse

from mindful_core.utils.misc import load_json

from mindful_subream.dataset.dataset_master_builder import SubreamMasterBuilder
from mindful_subream.scripts.organize_raw_data_augmentations import organize_raw_data_augmentations


def get_augmentations(images_root: Path) -> list[str]:
    return [path.name for path in images_root.glob(pattern="recon_*") if path.is_dir()]


def build_augmentation_dataset(augmentation: str,
                               template: dict,
                               patients_images_root: Path,
                               output_root: Path,
                               log_root: Path,
                               ) -> SubreamMasterBuilder:
    template["patients_images"] = (patients_images_root / augmentation).as_posix()
    template["output_path"] = (output_root / augmentation).as_posix()
    template["log_path"] = (log_root / augmentation).as_posix()

    builder = SubreamMasterBuilder(template)
    builder.run()

    return builder


def aggregate_folds_subset(subsets_folds: list[list[Path]],
                           augmentations: list[str]
                           ) -> list[pd.DataFrame]:
    if len(subsets_folds) != len(augmentations):
        raise ValueError("You must have the same number of subsets "
                         "as the number of augmentations (for a 1:1 mapping).")

    subsets_folds = [list(ith_folds) for ith_folds in zip(*subsets_folds)]
    aggregated_folds = []
    for ith_subsets_folds in subsets_folds:
        updated_folds = []
        reference_fold = pd.read_csv(ith_subsets_folds[0], index_col="ScanID")
        for augmentation, fold_path in zip(augmentations, ith_subsets_folds):
            fold = pd.read_csv(fold_path, index_col="ScanID")
            fold["SubsetID"] = reference_fold["SubsetID"]
            fold.index = fold.index.map(lambda x: "{}_{}".format(x, augmentation))
            updated_folds.append(fold)
        aggregated_fold = pd.concat(updated_folds, axis="index")
        aggregated_fold = aggregated_fold.sort_index()
        aggregated_folds.append(aggregated_fold)

    return aggregated_folds


def aggregate_all_folds(all_folds: list[dict[str, dict[str, dict[str, list[Path]]]]],
                        augmentations: list[str],
                        folds_output_dir: Path):
    if len(all_folds) != len(augmentations):
        raise ValueError("You must have the same number of subsets "
                         "as the number of augmentations (for a 1:1 mapping). "
                         "Got {} subsets and {} augmentations.".format(len(all_folds), len(augmentations)))

    reference = all_folds[0]

    for modality in reference:
        for fold_version in reference[modality]:
            for mode in reference[modality][fold_version]:
                subset_folds = [folds[modality][fold_version][mode] for folds in all_folds]
                aggregated_folds: list[pd.DataFrame] = aggregate_folds_subset(subset_folds, augmentations)

                output_dir = folds_output_dir / modality / fold_version / mode
                output_dir.mkdir(parents=True, exist_ok=True)
                for i, aggregated_fold in enumerate(aggregated_folds):
                    aggregated_fold.to_csv(output_dir / "fold_{:02d}.csv".format(i))


def build_raw_data_aug_dataset(template_path: Path,
                               resume_from: str | None,
                               verbose=True) -> None:
    template = load_json(template_path)

    patients_images_root = Path(template["patients_images"])
    output_root = Path(template["output_path"])
    log_root = template_path.parent / "logs"

    organize_raw_data_augmentations(patients_images_root, verbose=verbose)

    augmentations = get_augmentations(patients_images_root)
    folds: list[dict[str, dict[str, dict[str, list[Path]]]]] = []
    skip = resume_from is not None
    failed = {}
    failed_log = None
    for augmentation in augmentations:
        if skip:
            if augmentation == resume_from:
                skip = False
            else:
                if verbose:
                    print("Skipping dataset for augmentation {}".format(augmentation))
                continue

        if verbose:
            print("Building dataset for augmentation {}".format(augmentation))

        try:
            builder = build_augmentation_dataset(augmentation, template, patients_images_root, output_root, log_root)
            folds.append(builder.data.folds)
        except Exception as exception:
            failed[augmentation] = exception
            if failed_log is None:
                failed_log = open(log_root / "failed.txt", "w")
                augmentations.remove(augmentation)
            failed_log.write("{}: {}\n".format(augmentation, exception))

    if len(failed) > 0:
        print("Failed building for the following augmentations: {}".format(list(failed.keys())))

    if failed_log is not None:
        failed_log.close()

    folds_output_dir = output_root / "folds"
    aggregate_all_folds(folds, augmentations, folds_output_dir)


def list_and_aggregate_folds(root: Path, date: str):
    subsets = list(root.iterdir())
    augmentations = []
    all_folds = []
    for subset in subsets:
        subset_folds = {}
        subset_root = subset / date / "folds"

        if not subset_root.exists():
            continue

        augmentations.append(subset.name)

        for modality_dir in subset_root.iterdir():
            subset_folds[modality_dir.name] = {}
            if not modality_dir.is_dir():
                continue
            for fold_version_dir in modality_dir.iterdir():
                if not fold_version_dir.is_dir():
                    continue
                subset_folds[modality_dir.name][fold_version_dir.name] = {}
                for mode_dir in fold_version_dir.iterdir():
                    if not mode_dir.is_dir():
                        continue
                    subset_folds[modality_dir.name][fold_version_dir.name][mode_dir.name] = list(mode_dir.iterdir())
        all_folds.append(subset_folds)

    aggregate_all_folds(all_folds=all_folds,
                        augmentations=augmentations,
                        folds_output_dir=root / "folds")


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("template")
    arg_parser.add_argument("--resume_from", default=None)
    args = arg_parser.parse_args()

    template_path = Path(args.template)
    resume_from: str | None = args.resume_from

    # build_raw_data_aug_dataset(template_path, resume_from)

    tmp = Path("/data/projects/subream/data/processed/classification/raw_data_augmented/")
    list_and_aggregate_folds(tmp, "2025-01-08")


if __name__ == "__main__":
    main()
