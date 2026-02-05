# SPDX-FileCopyrightText: 2025 GitHub
# SPDX-License-Identifier: MIT

import os
import shutil
import subprocess
import sys


def read_file_list(list_path):
    """
    Reads a file containing file paths, ignoring empty lines and lines starting with '#'.
    Returns a list of relative file paths.
    """
    with open(list_path) as f:
        lines = [line.strip() for line in f]
    return [line for line in lines if line and not line.startswith("#")]


def copy_files(file_list, dest_dir):
    """
    Copy files listed in file_list to dest_dir, preserving their relative paths.

    :param file_list: List of file paths (relative to current working directory)
    :param dest_dir: Destination directory where files will be copied
    """
    for rel_path in file_list:
        abs_src = os.path.abspath(rel_path)
        abs_dest = os.path.abspath(os.path.join(dest_dir, rel_path))
        os.makedirs(os.path.dirname(abs_dest), exist_ok=True)
        shutil.copy2(abs_src, abs_dest)
        print(f"Copied {abs_src} -> {abs_dest}")


def ensure_git_repo(dest_dir):
    """
    Initializes a git repository in dest_dir if it's not already a git repo.
    Sets the main branch to 'main'.
    """
    git_dir = os.path.join(dest_dir, ".git")
    if not os.path.isdir(git_dir):
        try:
            subprocess.run(["git", "init", "-b", "main"], cwd=dest_dir, check=True)
            print(f"Initialized new git repository in {dest_dir} with 'main' as the default branch")
        except subprocess.CalledProcessError as e:
            print(f"Failed to initialize git repository in {dest_dir}: {e}")
            sys.exit(1)
    else:
        # Ensure main branch exists and is checked out
        try:
            branches = subprocess.check_output(["git", "branch"], cwd=dest_dir, text=True)
            if "main" not in branches:
                subprocess.run(["git", "checkout", "-b", "main"], cwd=dest_dir, check=True)
                print("Created and switched to 'main' branch.")
            else:
                subprocess.run(["git", "checkout", "main"], cwd=dest_dir, check=True)
                print("Switched to 'main' branch.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to ensure 'main' branch in {dest_dir}: {e}")
            sys.exit(1)


def git_add_files(file_list, dest_dir):
    """
    Runs 'git add' on each file in file_list within dest_dir.
    """
    cwd = os.getcwd()
    os.chdir(dest_dir)
    try:
        for rel_path in file_list:
            try:
                subprocess.run(["git", "add", "-f", rel_path], check=True)
                print(f"git add {rel_path}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to git add {rel_path}: {e}")
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python copy_files.py <file_list.txt> <dest_dir>")
        sys.exit(1)
    file_list_path = sys.argv[1]
    dest_dir = sys.argv[2]
    file_list = read_file_list(file_list_path)
    copy_files(file_list, dest_dir)
    ensure_git_repo(dest_dir)
    git_add_files(file_list, dest_dir)
