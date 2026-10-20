#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository from this script, regardless of the caller's directory.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python - "$PWD" "$@" <<'PY'
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile

parser = argparse.ArgumentParser(description="Download and extract SARE evaluation data.")
parser.add_argument("root", help=argparse.SUPPRESS)
parser.add_argument("dataset", choices=["Chameleon", "Genimage", "CommunityForensics", "Forensynths"])
args = parser.parse_args()

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

root = Path(args.root).resolve()
destination = root / "downloads" / "SARE-evaluation"
repo = "SARE-DB/SARE-evaluation"
# Pin metadata and archives to one revision while new datasets are being uploaded.
revision = HfApi().dataset_info(repo).sha
manifest_path = hf_hub_download(
    repo, "manifest.json", repo_type="dataset", revision=revision
)
manifest = json.loads(Path(manifest_path).read_text())
if args.dataset not in manifest.get("completed_datasets", []):
    raise SystemExit(f"{args.dataset} is not fully uploaded yet. Please try again later.")

prefix = f"archives/{args.dataset}/"
entries = [entry for entry in manifest["archives"] if entry["path"].startswith(prefix)]
expected = manifest["datasets"][args.dataset]["parts"]
if not entries or len(entries) != expected:
    raise SystemExit("Incomplete archive manifest; no data was extracted.")
if len({entry["path"] for entry in entries}) != len(entries):
    raise SystemExit("Duplicate archive paths in manifest.")
for entry in entries:
    path = PurePosixPath(entry["path"])
    if path.is_absolute() or ".." in path.parts or path.suffix != ".tar":
        raise SystemExit(f"Invalid archive path: {path}")

snapshot_download(
    repo, repo_type="dataset", revision=revision,
    allow_patterns=[entry["path"] for entry in entries] + ["manifest.json", "SHA256SUMS"],
    local_dir=str(destination),
)

# Check every required part before extracting anything, including missing files.
for entry in entries:
    path = destination / entry["path"]
    if not path.is_file() or path.stat().st_size != entry["bytes"]:
        raise SystemExit(f"Missing or incomplete archive: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != entry["sha256"]:
        raise SystemExit(f"Checksum mismatch: {path}. Remove this archive and retry.")
    print(f"Verified {path.name}", flush=True)

# Extract only regular files/directories within this dataset, never server symlinks.
target = root / "dataset" / args.dataset
for entry in entries:
    with tarfile.open(destination / entry["path"], "r:") as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if (relative.is_absolute() or ".." in relative.parts
                    or relative.parts[:2] != ("dataset", args.dataset)
                    or not (member.isfile() or member.isdir())):
                raise SystemExit(f"Unexpected archive member: {member.name}")
            output = root.joinpath(*relative.parts)
            if not output.resolve().is_relative_to(target):
                raise SystemExit(f"Unsafe extraction path or existing symlink: {output}")
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, output.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
    print(f"Extracted {entry['path']}", flush=True)

print(f"Ready: {target}")
PY
