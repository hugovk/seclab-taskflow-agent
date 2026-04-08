# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


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
    logger.info(f"Pushed {image_name}:{tag}")
    digest = get_image_digest(image_name, tag)
    if digest is None:
        raise RuntimeError(f"Failed to determine image digest for {image_name}:{tag}")
    logger.info(f"Image digest: {digest}")
    with open("/tmp/digest.txt", "w") as f:
        f.write(digest)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) != 3:
        logger.error("Usage: python build_and_publish_docker.py <ghcr_username/repo> <tag>")
        logger.error("Example: python build_and_publish_docker.py ghcr.io/anticomputer/my-python-app latest")
        sys.exit(1)

    image_name = sys.argv[1]
    tag = sys.argv[2]

    # Build and push image
    build_and_push_image("docker", image_name, tag)
