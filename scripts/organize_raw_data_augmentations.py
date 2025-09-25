from pathlib import Path
from tqdm import tqdm
import shutil
import os
import argparse


def list_patient_folders(root: Path) -> list[Path]:
    return list(root.glob(pattern="patient_*"))


def list_all_augmentations(patient_folders: list[Path]) -> list[str]:
    all_augmentations = []
    for patient_folder in patient_folders:
        mha_folder = patient_folder / "mha"
        patient_first_phases = mha_folder.glob("*_t0.mha")
        for image_path in patient_first_phases:
            name_parts = image_path.stem.split("_")
            augmentation_id = "recon_{}_{}".format(name_parts[2], name_parts[3])
            if augmentation_id not in all_augmentations:
                all_augmentations.append(augmentation_id)
    return all_augmentations


def remove_folder_if_empty(folder_path: Path) -> bool:
    folder_content = list(folder_path.iterdir())
    remove_folder = len(folder_content) == 0
    if remove_folder:
        try:
            os.rmdir(folder_path)
        except OSError:
            remove_folder = False

    return remove_folder


def organize_raw_data_augmentations(root: Path, verbose=True) -> list[str]:
    patient_folders = list_patient_folders(root)
    all_augmentations = list_all_augmentations(patient_folders)

    if verbose:
        print("Found {} augmentations to organize: {}".format(len(all_augmentations), all_augmentations))

    for augmentation in tqdm(all_augmentations, disable=not verbose):
        augmentations_root = root / augmentation
        augmentations_root.mkdir(exist_ok=True)

        for patient_folder in patient_folders:
            patient_id = patient_folder.name.split("_")[-1]

            target_dir = augmentations_root / patient_folder.name
            for phase_id in ["t0", "t13"]:
                source_filename = "{}_{}_{}.mha".format(patient_id, augmentation, phase_id)
                source_path = patient_folder / "mha" / source_filename
                if not source_path.exists():
                    continue

                target_dir.mkdir(exist_ok=True)
                target_filename = "Patient_{}_{}.mha".format(patient_id, phase_id)
                target_path = target_dir / target_filename

                if not target_path.exists():
                    shutil.copy2(source_path, target_path)

    return all_augmentations


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("root")
    args = arg_parser.parse_args()

    root = Path(args.root)
    
    organize_raw_data_augmentations(root)


if __name__ == "__main__":
    main()
