import os
import shutil


PARENT_FOLDER = r"/home/belinda.lokaj/programming/subream/mindful/mindful_subream/dataset/TIC/2026-02-10/clinical_data/non_imaging_tic_only_TrueFolds/b_m"

FILES_TO_COPY = [
    r"/home/belinda.lokaj/programming/subream/mindful/mindful_subream/dataset/TIC/2026-02-10/clinical_data/non_imaging_tic_only_TrueFolds/b_m/subream_fold_04.csv",
    r"/home/belinda.lokaj/programming/subream/mindful/mindful_subream/dataset/TIC/2026-02-10/clinical_data/non_imaging_tic_only_TrueFolds/b_m/subream_fold_03.csv",
    r"/home/belinda.lokaj/programming/subream/mindful/mindful_subream/dataset/TIC/2026-02-10/clinical_data/non_imaging_tic_only_TrueFolds/b_m/subream_fold_02.csv",
    r"/home/belinda.lokaj/programming/subream/mindful/mindful_subream/dataset/TIC/2026-02-10/clinical_data/non_imaging_tic_only_TrueFolds/b_m/subream_fold_01.csv",
    r"/home/belinda.lokaj/programming/subream/mindful/mindful_subream/dataset/TIC/2026-02-10/clinical_data/non_imaging_tic_only_TrueFolds/b_m/subream_fold_00.csv"
]


def copy_files_to_all_folders(parent_folder: str, files_to_copy: list[str]):
    for item in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, item)
        if not os.path.isdir(folder_path):
            continue

        for file_path in files_to_copy:
            destination = os.path.join(folder_path, os.path.basename(file_path))
            shutil.copy2(file_path, destination)


if __name__ == "__main__":
    copy_files_to_all_folders(PARENT_FOLDER, FILES_TO_COPY)
