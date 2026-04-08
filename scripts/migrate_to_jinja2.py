#!/usr/bin/env python3
# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""
Automated migration script for converting taskflow YAML files
from custom template syntax to Jinja2 syntax.

Usage:
    python scripts/migrate_to_jinja2.py /path/to/taskflows
    python scripts/migrate_to_jinja2.py --dry-run taskflow.yaml
"""

import re
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


class TemplateMigrator:
    """Migrates custom template syntax to Jinja2."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.transformations: List[Tuple[str, str]] = []

    def migrate_content(self, content: str) -> str:
        """Apply all template transformations to content."""
        original = content

        # 0. Version number: version: 1 or version: 2 -> version: "1.0"
        content = re.sub(
            r'^(\s*version:\s*)(?:1|2)\s*$',
            r'\1"1.0"',
            content,
            flags=re.MULTILINE
        )

        # 1. Environment variables: {{ env VAR }} -> {{ env('VAR') }}
        content = re.sub(
            r'\{\{\s*env\s+([A-Z0-9_]+)\s*\}\}',
            r"{{ env('\1') }}",
            content
        )

        # 2. Global variables: {{ GLOBALS_key }} -> {{ globals.key }}
        content = re.sub(
            r'\{\{\s*GLOBALS_([a-zA-Z0-9_\.]+)\s*\}\}',
            r'{{ globals.\1 }}',
            content
        )

        # 3. Input variables: {{ INPUTS_key }} -> {{ inputs.key }}
        content = re.sub(
            r'\{\{\s*INPUTS_([a-zA-Z0-9_\.]+)\s*\}\}',
            r'{{ inputs.\1 }}',
            content
        )

        # 4. Result dict keys: {{ RESULT_key }} -> {{ result.key }}
        content = re.sub(
            r'\{\{\s*RESULT_([a-zA-Z0-9_\.]+)\s*\}\}',
            r'{{ result.\1 }}',
            content
        )

        # 5. Result primitive: {{ RESULT }} -> {{ result }}
        content = re.sub(
            r'\{\{\s*RESULT\s*\}\}',
            r'{{ result }}',
            content
        )

        # 6. Reusable prompts: {{ PROMPTS_path }} -> {% include 'path' %}
        content = re.sub(
            r'\{\{\s*PROMPTS_([a-zA-Z0-9_\.]+)\s*\}\}',
            r"{% include '\1' %}",
            content
        )

        if content != original:
            self.transformations.append((original, content))

        return content

    def migrate_file(self, file_path: Path) -> bool:
        """Migrate a single YAML file.

        Returns:
            True if file was modified, False otherwise
        """
        if file_path.suffix != '.yaml':
            logger.info(f"Skipping non-YAML file: {file_path}")
            return False

        try:
            with open(file_path, 'r') as f:
                original_content = f.read()

            migrated_content = self.migrate_content(original_content)

            if migrated_content == original_content:
                logger.info(f"No changes needed: {file_path}")
                return False

            if self.dry_run:
                logger.info(f"{'='*60}")
                logger.info(f"Would modify: {file_path}")
                logger.info(f"{'='*60}")
                self._show_diff(original_content, migrated_content)
                return True

            # Write migrated content
            with open(file_path, 'w') as f:
                f.write(migrated_content)

            logger.info(f"Migrated: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error migrating {file_path}: {e}")
            return False

    def _show_diff(self, original: str, migrated: str):
        """Show simplified diff between original and migrated."""
        orig_lines = original.splitlines()
        mig_lines = migrated.splitlines()

        for i, (orig, mig) in enumerate(zip(orig_lines, mig_lines, strict=False), 1):
            if orig != mig:
                logger.info(f"Line {i}:")
                logger.info(f"  - {orig}")
                logger.info(f"  + {mig}")

    def migrate_directory(self, directory: Path, recursive: bool = True) -> int:
        """Migrate all YAML files in directory.

        Returns:
            Number of files modified
        """
        pattern = '**/*.yaml' if recursive else '*.yaml'
        yaml_files = list(directory.glob(pattern))

        if not yaml_files:
            logger.info(f"No YAML files found in {directory}")
            return 0

        logger.info(f"Found {len(yaml_files)} YAML files")

        modified_count = 0
        for yaml_file in yaml_files:
            if self.migrate_file(yaml_file):
                modified_count += 1

        return modified_count


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description='Migrate taskflow YAML files to Jinja2 syntax'
    )
    parser.add_argument(
        'paths',
        nargs='+',
        type=Path,
        help='YAML files or directories to migrate'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show changes without modifying files'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not recurse into subdirectories'
    )

    args = parser.parse_args()

    migrator = TemplateMigrator(dry_run=args.dry_run)

    total_modified = 0
    for path in args.paths:
        if not path.exists():
            logger.error(f"Path not found: {path}")
            continue

        if path.is_file():
            if migrator.migrate_file(path):
                total_modified += 1
        elif path.is_dir():
            modified = migrator.migrate_directory(
                path,
                recursive=not args.no_recursive
            )
            total_modified += modified
        else:
            logger.error(f"Invalid path: {path}")

    logger.info(f"{'='*60}")
    if args.dry_run:
        logger.info(f"Dry run complete. {total_modified} files would be modified.")
    else:
        logger.info(f"Migration complete. {total_modified} files modified.")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
