# SPDX-FileCopyrightText: 2025 GitHub
# SPDX-License-Identifier: MIT

import subprocess
import sys


def get_image_digest(image_name, tag):
    result = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", f"{image_name}:{tag}"],
        stdout=subprocess.PIPE,
        check=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.strip().startswith("Digest:"):
            return line.strip().split(":", 1)[1].strip()
    return None


def build_and_push_image(dest_dir, image_name, tag):
    # Build
    subprocess.run(
        ["docker", "buildx", "build", "--platform", "linux/amd64", "-t", f"{image_name}:{tag}", dest_dir], check=True
    )
    # Push
    subprocess.run(["docker", "push", f"{image_name}:{tag}"], check=True)
    print(f"Pushed {image_name}:{tag}")
    digest = get_image_digest(image_name, tag)
    print(f"Image digest: {digest}")
    with open("/tmp/digest.txt", "w") as f:
        f.write(digest)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python build_and_publish_docker.py <ghcr_username/repo> <tag>")
        print("Example: python build_and_publish_docker.py ghcr.io/anticomputer/my-python-app latest")
        sys.exit(1)

    image_name = sys.argv[1]
    tag = sys.argv[2]

    # Build and push image
    build_and_push_image("docker", image_name, tag)
