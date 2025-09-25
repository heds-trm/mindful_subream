from pathlib import Path
from string import Formatter

from mindful_subream.dataset.build.build_steps.build_step import SubreamBuildStep
from mindful_subream.dataset.build.build_data import SubreamBuildData
from mindful_subream.dataset.build.build_logger import SubreamBuildLogger


class SubreamReadmeMaker(SubreamBuildStep):
    def __init__(self,
                 build_data: SubreamBuildData,
                 output_root: Path,
                 logger: SubreamBuildLogger):
        super(SubreamReadmeMaker, self).__init__(build_data, output_root, logger)

    def run(self) -> None:
        self.scan_folder(self.output_root)

    def scan_folder(self, folder: Path):
        if (not folder.exists()) or (not folder.is_dir()):
            return

        if self.folder_requires_readme(folder):
            self.add_readme(folder)
            if (folder / "README.txt").exists():
                self.log("ReadmeMaker - Folder `{}` contains a `README.txt`.".format(folder))
            else:
                self.warn("ReadmeMaker - Warning: Folder `{}` does not contain a `README.txt` !".format(folder))

        for filepath in folder.iterdir():
            if filepath.is_dir():
                self.scan_folder(filepath)

    @staticmethod
    def folder_requires_readme(folder: Path) -> bool:
        if (not folder.exists()) or (not folder.is_dir()):
            return False

        if folder.name in ["lymphnode", "positive", "negative"]:
            return False

        for filepath in folder.iterdir():
            if filepath.is_file() \
                    and (filepath.suffix in [".mha", ".csv", ".txt"]) \
                    and (filepath.name != "README.txt"):
                return False

        return True

    def add_readme(self, folder: Path):
        rel_path = folder.relative_to(self.output_root)
        expected_readme_template_path = self.data.readme_templates_folder / rel_path / "README.txt"
        if not expected_readme_template_path.exists():
            return

        with open(expected_readme_template_path, "r") as file:
            text = file.read()

        text = self.parse_readme(text)
        with open(folder / "README.txt", "w") as file:
            file.write(text)

    def parse_readme(self, text: str) -> str:
        text_keys: list[str] = [field[1] for field in Formatter().parse(text) if field[1] is not None]
        text_format_dict = {}
        for text_key in text_keys:
            if hasattr(self.data, text_key):
                value = getattr(self.data, text_key)
                text_format_dict[text_key] = value
            else:
                self.warn("ReadmeMaker - Value `{}` is unregistered, skipping...".format(text_key))
                text_format_dict[text_key] = text_key.upper()
        text = text.format(**text_format_dict)
        return text

    @staticmethod
    def is_a_date(text: str) -> bool:
        if "-" not in text:
            return False

        fields = text.split("-")
        if len(fields) != 3:
            return False

        if not all([field.isdigit() for field in fields]):
            return False

        fields_lengths = [len(field) for field in fields]
        expected_lengths = (4, 2, 2)
        if not all([(field_length == expected_length)
                    for field_length, expected_length
                    in zip(fields_lengths, expected_lengths)]):
            return False

        return True
