#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
base="https://github.com/jh721/SARE/releases/download/v1.0.0"
dest="downloads/sare-v1.0.0"
mkdir -p "$dest"
# Download release parts without changing their bytes.
curl --fail --location --retry 5 --continue-at - "$base/sare_clip_sd14.pth" --output "$dest/sare_clip_sd14.pth"
curl --fail --location --retry 5 --continue-at - "$base/sare_dino_sd14.pth.part00" --output "$dest/sare_dino_sd14.pth.part00"
curl --fail --location --retry 5 --continue-at - "$base/sare_dino_sd14.pth.part01" --output "$dest/sare_dino_sd14.pth.part01"
curl --fail --location --retry 5 "$base/SHA256SUMS" --output "$dest/SHA256SUMS"
curl --fail --location --retry 5 "$base/CHECKPOINT_SHA256SUMS" --output "$dest/CHECKPOINT_SHA256SUMS"
curl --fail --location --retry 5 "$base/DINOv3-LICENSE.md" --output "$dest/DINOv3-LICENSE.md"
(cd "$dest" && sha256sum -c SHA256SUMS)
mkdir -p "output/Genimage/1/SARE_CLIP_SD14/weights"
cat "$dest/sare_clip_sd14.pth" > "output/Genimage/1/SARE_CLIP_SD14/weights/sare_clip_sd14.pth"
mkdir -p "output/Genimage/1/SARE_DINO_SD14/weights"
cat "$dest/sare_dino_sd14.pth.part00" "$dest/sare_dino_sd14.pth.part01" > "output/Genimage/1/SARE_DINO_SD14/weights/sare_dino_sd14.pth"
sha256sum -c "$dest/CHECKPOINT_SHA256SUMS"
echo "All checkpoints are ready at the paths expected by SARE."
