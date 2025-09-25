from pathlib import Path
from datetime import datetime
from io import TextIOWrapper
import os
import warnings


class SubreamBuildLogger(object):
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._main_log_file = None

    @property
    def main_log_file(self) -> TextIOWrapper:
        if self._main_log_file is None:
            self._main_log_file = open(self.log_path / "builder.log", "w")
        return self._main_log_file

    def log(self, message: str, print_message=True) -> None:
        message = "{} - {}".format(datetime.now().isoformat(), message)
        self.main_log_file.write(message + os.linesep)
        self.main_log_file.flush()
        if print_message:
            print(message)

    def warn(self, message: str, print_message=False) -> None:
        warnings.warn(message)
        self.log(message, print_message)

    def error(self, message: str, exception_type: type[Exception], print_message=False) -> None:
        self.log(message, print_message)
        raise exception_type(message)

    def new_line(self) -> None:
        self.main_log_file.write(os.linesep)
