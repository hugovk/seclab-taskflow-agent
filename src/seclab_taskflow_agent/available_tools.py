# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""YAML resource loader for taskflow grammar files.

Loads and caches personality, taskflow, toolbox, model_config, and prompt
YAML files, validating them against Pydantic grammar models at parse time.
"""

from __future__ import annotations

__all__ = ["AvailableTools"]

import importlib.resources
from enum import Enum
from typing import Union

import yaml
from pydantic import ValidationError

from .models import (
    DOCUMENT_MODELS,
    ModelConfigDocument,
    PersonalityDocument,
    PromptDocument,
    TaskflowDocument,
    ToolboxDocument,
)


class BadToolNameError(Exception):
    pass


class VersionException(Exception):
    pass


class FileTypeException(Exception):
    pass


class InvalidToolFormatError(BadToolNameError):
    def __init__(self, toolname: str) -> None:
        super().__init__(
            f'Not a valid toolname: "{toolname}". '
            f'Expected format: "packagename.filename"'
        )


class ToolDirNotFoundError(BadToolNameError):
    def __init__(self, toolname: str, pkg_dir: object) -> None:
        super().__init__(f"Cannot load {toolname} because {pkg_dir} is not a valid directory.")


class FiletypeMismatchError(FileTypeException):
    def __init__(self, filepath: object, expected: str, got: str) -> None:
        super().__init__(f"Error in {filepath}: expected filetype {expected!r}, got {got!r}.")


class UnknownFiletypeError(BadToolNameError):
    def __init__(self, filetype: str, toolname: str) -> None:
        super().__init__(f"Unknown filetype {filetype!r} in {toolname}")


class ToolValidationError(BadToolNameError):
    def __init__(self, toolname: str, exc: Exception) -> None:
        super().__init__(f"Validation error loading {toolname}: {exc}")


class ToolLoadError(BadToolNameError):
    def __init__(self, toolname: str, exc: Exception) -> None:
        super().__init__(f"Cannot load {toolname}: {exc}")


class ToolFileNotFoundError(BadToolNameError):
    def __init__(self, toolname: str, filepath: object) -> None:
        super().__init__(f"Cannot load {toolname} because {filepath} is not a valid file.")


class AvailableToolType(Enum):
    Personality = "personality"
    Taskflow = "taskflow"
    Prompt = "prompt"
    Toolbox = "toolbox"
    ModelConfig = "model_config"


# Union of all document model types returned by AvailableTools
DocumentModel = Union[
    TaskflowDocument, PersonalityDocument, ToolboxDocument,
    ModelConfigDocument, PromptDocument,
]


class AvailableTools:
    """Loads, validates, and caches YAML grammar files as Pydantic models."""

    def __init__(self) -> None:
        self._cache: dict[AvailableToolType, dict[str, DocumentModel]] = {}

    def get_personality(self, name: str) -> PersonalityDocument:
        """Load a personality YAML and return a validated PersonalityDocument."""
        return self._load(AvailableToolType.Personality, name)

    def get_taskflow(self, name: str) -> TaskflowDocument:
        """Load a taskflow YAML and return a validated TaskflowDocument."""
        return self._load(AvailableToolType.Taskflow, name)

    def get_prompt(self, name: str) -> PromptDocument:
        """Load a prompt YAML and return a validated PromptDocument."""
        return self._load(AvailableToolType.Prompt, name)

    def get_toolbox(self, name: str) -> ToolboxDocument:
        """Load a toolbox YAML and return a validated ToolboxDocument."""
        return self._load(AvailableToolType.Toolbox, name)

    def get_model_config(self, name: str) -> ModelConfigDocument:
        """Load a model_config YAML and return a validated ModelConfigDocument."""
        return self._load(AvailableToolType.ModelConfig, name)

    # Keep legacy alias for code that uses the generic accessor
    def get_tool(self, tooltype: AvailableToolType, toolname: str) -> DocumentModel:
        """Generic loader — prefer the typed ``get_*()`` methods."""
        return self._load(tooltype, toolname)

    def _load(self, tooltype: AvailableToolType, toolname: str) -> DocumentModel:
        """Load, validate, and cache a YAML grammar file.

        Args:
            tooltype: Expected file type (personality, taskflow, etc.).
            toolname: Dotted module path, e.g. ``"examples.taskflows.echo"``.

        Returns:
            A validated Pydantic document model instance.

        Raises:
            BadToolNameError: If the tool cannot be found or loaded.
            VersionException: If the grammar version is unsupported.
            FileTypeException: If the filetype doesn't match expectations.
        """
        # Check cache first
        if tooltype in self._cache and toolname in self._cache[tooltype]:
            return self._cache[tooltype][toolname]

        # Resolve package and filename from dotted path
        components = toolname.rsplit(".", 1)
        if len(components) != 2:
            raise InvalidToolFormatError(toolname)
        package, filename = components

        try:
            pkg_dir = importlib.resources.files(package)
            if not pkg_dir.is_dir():
                raise ToolDirNotFoundError(toolname, pkg_dir)
            filepath = pkg_dir.joinpath(filename + ".yaml")
            with filepath.open() as fh:
                raw = yaml.safe_load(fh)

            # Validate header before full parse
            header = raw.get("seclab-taskflow-agent", {})
            filetype = header.get("filetype", "")
            if filetype != tooltype.value:
                raise FiletypeMismatchError(filepath, tooltype.value, filetype)

            # Parse into the appropriate Pydantic model
            model_cls = DOCUMENT_MODELS.get(filetype)
            if model_cls is None:
                raise UnknownFiletypeError(filetype, toolname)

            try:
                doc = model_cls(**raw)
            except ValidationError as exc:
                # Surface version errors as VersionException for compat
                for err in exc.errors():
                    if "Unsupported version" in str(err.get("msg", "")):
                        raise VersionException(str(err["msg"])) from exc
                raise ToolValidationError(toolname, exc) from exc

            # Cache and return
            if tooltype not in self._cache:
                self._cache[tooltype] = {}
            self._cache[tooltype][toolname] = doc
            return doc

        except ModuleNotFoundError as exc:
            raise ToolLoadError(toolname, exc) from exc
        except FileNotFoundError:
            raise ToolFileNotFoundError(toolname, filepath)
        except ValueError as exc:
            raise ToolLoadError(toolname, exc) from exc
