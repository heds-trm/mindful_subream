from typing import Union
import warnings


class WarningsContextManager(object):
    """
        Used to differentiate warnings created by this script for users and other warnings.
    """
    manager: Union["WarningsContextManager", None] = None

    def __init__(self):
        self.warnings: list[str] = []

    def __enter__(self):
        if self.manager is None:
            self.manager = self

        return self.manager

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None:
        self.manager = None
        return None

    @classmethod
    def warn(cls, message: str) -> None:
        if cls.manager is None:
            warnings.warn(message)
        cls.manager.warnings.append(message)

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
    