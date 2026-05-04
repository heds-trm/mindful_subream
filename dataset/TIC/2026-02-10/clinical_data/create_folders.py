import os
import pandas as pd

# === Configuration à modifier ici ===
BASE_PATH = r"/home/belinda.lokaj/programming/subream/mindful/mindful_subream/dataset/TIC/2026-02-10/clinical_data/non_imaging_tic_only_TrueFolds/b_g_m"
CSV_PATH = r"/home/belinda.lokaj/programming/subream/mindful/mindful_subream/dataset/TIC/2026-02-10/clinical_data/folders_name.csv"
COLUMN_NAME = "folder_name"


def create_folders_from_csv(base_path: str, csv_path: str, column_name: str):
    """Crée des dossiers dans base_path à partir des valeurs d'une colonne CSV."""

    if not os.path.isdir(base_path):
        raise ValueError(f"Le chemin du dossier parent n'existe pas ou n'est pas un dossier : {base_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Erreur lors de la lecture du CSV : {e}")

    if column_name not in df.columns:
        raise ValueError(
            f"La colonne '{column_name}' est introuvable dans le CSV. Colonnes disponibles : {list(df.columns)}"
        )

    for folder_name in df[column_name].dropna():
        folder_name = str(folder_name).strip()
        if not folder_name:
            continue

        folder_path = os.path.join(base_path, folder_name)

        try:
            os.makedirs(folder_path, exist_ok=True)
            print(f"Créé (ou déjà existant) : {folder_path}")
        except Exception as e:
            print(f"Impossible de créer {folder_path} : {e}")


if __name__ == "__main__":
    create_folders_from_csv(BASE_PATH, CSV_PATH, COLUMN_NAME)
