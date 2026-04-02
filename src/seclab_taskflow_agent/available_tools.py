# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""YAML resource loader for taskflow grammar files.

Loads and caches personality, taskflow, toolbox, model_config, and prompt
YAML files, validating them against Pydantic grammar models at parse time.
"""

from __future__ import annotations

__all__ = ["AvailableTools"]

import importlib.resources
import os
import sys
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


class AvailableToolType(Enum):
    Personality = "personality"
    Taskflow = "taskflow"
    Prompt = "prompt"
    Toolbox = "toolbox"
    ModelConfig = "model_config"


# Maps each AvailableToolType to the conventional subdirectory name used in packages
_SUBDIR_MAP: dict[AvailableToolType, str] = {
    AvailableToolType.Taskflow: "taskflows",
    AvailableToolType.Personality: "personalities",
    AvailableToolType.Toolbox: "toolboxes",
    AvailableToolType.Prompt: "prompts",
    AvailableToolType.ModelConfig: "model_configs",
}


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

    def list_resources(
        self, tooltype: AvailableToolType | None = None
    ) -> dict[AvailableToolType, list[str]]:
        """Discover all available YAML resources across the Python path.

        Scans every directory in ``sys.path`` (including the current working
        directory when the empty-string entry is present) for Python packages
        that contain a conventional resource subdirectory (e.g. ``taskflows/``,
        ``personalities/``, …).  Each ``.yaml`` file found in such a
        subdirectory is returned as a fully-qualified dotted resource name of
        the form ``<package>.<subdir>.<stem>``.

        Args:
            tooltype: When provided, only resources of that type are returned.
                      When ``None`` (default) all types are returned.

        Returns:
            A mapping from :class:`AvailableToolType` to a sorted list of
            dotted resource names.
        """
        types_to_scan = [tooltype] if tooltype is not None else list(AvailableToolType)
        result: dict[AvailableToolType, list[str]] = {t: [] for t in types_to_scan}

        seen_dirs: set[str] = set()
        for path_entry in sys.path:
            actual_path = path_entry if path_entry else os.getcwd()
            actual_path = os.path.abspath(actual_path)
            if actual_path in seen_dirs or not os.path.isdir(actual_path):
                continue
            seen_dirs.add(actual_path)

            try:
                top_level_entries = os.listdir(actual_path)
            except PermissionError:
                continue

            for pkg_name in top_level_entries:
                pkg_path = os.path.join(actual_path, pkg_name)
                if not os.path.isdir(pkg_path) or pkg_name.startswith("."):
                    continue

                for tt in types_to_scan:
                    subdir = _SUBDIR_MAP[tt]
                    subdir_path = os.path.join(pkg_path, subdir)
                    if not os.path.isdir(subdir_path):
                        continue
                    try:
                        yaml_files = sorted(
                            f[:-5]
                            for f in os.listdir(subdir_path)
                            if f.endswith(".yaml")
                        )
                    except PermissionError:
                        continue
                    for stem in yaml_files:
                        resource_name = f"{pkg_name}.{subdir}.{stem}"
                        if resource_name not in result[tt]:
                            result[tt].append(resource_name)

        for tt in types_to_scan:
            result[tt].sort()

        return result

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
            raise BadToolNameError(
                f'Not a valid toolname: "{toolname}". '
                f'Expected format: "packagename.filename"'
            )
        package, filename = components

        try:
            pkg_dir = importlib.resources.files(package)
            if not pkg_dir.is_dir():
                raise BadToolNameError(
                    f"Cannot load {toolname} because {pkg_dir} is not a valid directory."
                )
            filepath = pkg_dir.joinpath(filename + ".yaml")
            with filepath.open() as fh:
                raw = yaml.safe_load(fh)

            # Validate header before full parse
            header = raw.get("seclab-taskflow-agent", {})
            filetype = header.get("filetype", "")
            if filetype != tooltype.value:
                raise FileTypeException(
                    f"Error in {filepath}: expected filetype {tooltype.value!r}, "
                    f"got {filetype!r}."
                )

            # Parse into the appropriate Pydantic model
            model_cls = DOCUMENT_MODELS.get(filetype)
            if model_cls is None:
                raise BadToolNameError(
                    f"Unknown filetype {filetype!r} in {toolname}"
                )

            try:
                doc = model_cls(**raw)
            except ValidationError as exc:
                # Surface version errors as VersionException for compat
                for err in exc.errors():
                    if "Unsupported version" in str(err.get("msg", "")):
                        raise VersionException(str(err["msg"])) from exc
                raise BadToolNameError(
                    f"Validation error loading {toolname}: {exc}"
                ) from exc

            # Cache and return
            if tooltype not in self._cache:
                self._cache[tooltype] = {}
            self._cache[tooltype][toolname] = doc
            return doc

        except ModuleNotFoundError as exc:
            raise BadToolNameError(
                f"Cannot load {toolname}: {exc}"
                + self._available_hint(tooltype)
            ) from exc
        except FileNotFoundError:
            raise BadToolNameError(
                f"Cannot load {toolname} because {filepath} is not a valid file."
                + self._available_hint(tooltype)
            )
        except ValueError as exc:
            raise BadToolNameError(f"Cannot load {toolname}: {exc}") from exc

    def _available_hint(self, tooltype: AvailableToolType) -> str:
        """Return a human-readable hint listing available resources of *tooltype*."""
        available = self.list_resources(tooltype).get(tooltype, [])
        if not available:
            return ""
        items = "\n  ".join(available)
        label = _SUBDIR_MAP[tooltype]
        return f"\n\nAvailable {label}:\n  {items}"
