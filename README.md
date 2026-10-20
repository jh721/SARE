# SARE

## Installation

Use Linux, an NVIDIA GPU with a CUDA-compatible driver, and Python 3.10. The inference code requires CUDA. The dependency versions in `requirements.txt` were exported from a Python 3.10.20 environment with PyTorch 2.5.1 and CUDA 12.1 wheels.

Download or clone this repository, then run the following commands from its root directory:

```bash
conda create -n sare python=3.10 -y
conda activate sare
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

For SARE-DINO, once the checkpoint release is published, the download script in Inference will also provide the DINOv3 ViT-H+/16 pretrained weights. Alternatively, obtain them from the [official DINOv3 repository](https://github.com/facebookresearch/dinov3), following its access and license requirements. Place the file at:

```text
dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
```

Set this environment variable **from the repository root**, before entering `scripts/`:

```bash
export DINOV3_WEIGHTS="$(pwd)/dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
```

## Inference

### 1. Download the SARE checkpoints

Download the checkpoint for the model you want to evaluate and place it at the exact path below. These files are distributed separately from the source code.

| Model | `--model_name` | Checkpoint path relative to the repository root |
| --- | --- | --- |
| SARE-CLIP | `clip-ViT-L-14` | `output/GenImage/1/SARE_CLIP_SD14/weights/sare_clip_sd14.pth` |
| SARE-DINO | `dinov3-vith16plus` | `output/GenImage/1/SARE_DINO_SD14/weights/sare_dino_sd14.pth` |

**Checkpoint release: upload pending.** The download command below will work after [v1.0.0](https://github.com/jh721/SARE/releases) is published.
Once published, run from the repository root to download both SARE models and the DINOv3 backbone:

```bash
bash scripts/download_checkpoints.sh
```

### 2. Download and place the evaluation data

Download the prepared evaluation data from [SARE-DB/SARE-evaluation on Hugging Face](https://huggingface.co/datasets/SARE-DB/SARE-evaluation). You only need the datasets you intend to evaluate.

**The data release is being uploaded progressively.** Before downloading a dataset, check that its name appears in `completed_datasets` in the repository's `manifest.json`. A partially uploaded dataset is not ready for evaluation. The manifest and `SHA256SUMS` are updated after each dataset is fully uploaded and verified.

Run the following commands from the **SARE repository root**, with the environment from Installation activated. This example downloads Chameleon:

```bash
hf download SARE-DB/SARE-evaluation --repo-type dataset \
  --include "archives/Chameleon/*" --include "manifest.json" \
  --include "SHA256SUMS" --local-dir downloads/SARE-evaluation

# Verify the downloaded archive files.
(cd downloads/SARE-evaluation && sha256sum --ignore-missing -c SHA256SUMS)

# Run only after checksum verification succeeds.
for archive in downloads/SARE-evaluation/archives/Chameleon/*.tar; do
  tar -xf "$archive"
done
```

Replace `Chameleon` in both commands with `Genimage`, `CommunityForensics`, or `Forensynths` to download another completed dataset. Download **all** archive parts for that dataset. Each `.tar` is an independent archive; extract every part separately, without concatenating them.

Every archive already contains paths beginning with `dataset/`. **Extract from the repository root, not from inside `dataset/`.** Extraction creates the exact layout below, ready for `scripts/test.sh`: no renaming, moving folders, or creating symbolic links is required. The archives contain actual image and reconstruction files. Keep the existing `test.sh` dataset paths.

Use a fresh local checkout or empty destination for extraction. On the original server, the existing `dataset/` entries are symlinks; those already provide the data, so do not extract the release over them.

```text
dataset/
├── Genimage/
│   └── Genimage_recon/
│       ├── wo_prompt/
│       └── w_prompt_blip/
├── CommunityForensics/
│   └── CommunityForensics_recon/
│       ├── wo_prompt/
│       └── w_prompt_blip/
├── Forensynths/
│   └── Forensynths_recon/
│       ├── wo_prompt/
│       └── w_prompt_blip/
└── Chameleon/
    └── Chameleon_recon/
        ├── wo_prompt/
        └── w_prompt_blip/
```

Keep the capitalization exactly as shown: the dataset directory is `Genimage`, while the command-line dataset name and checkpoint directory use `GenImage`. Avoid introducing an extra directory level when extracting archives.

SARE uses both an image input and a reconstruction-error input. For each evaluated sample, the loader needs all three files:

1. The image under `wo_prompt/.../crop/...`.
2. Its corresponding image under `w_prompt_blip/.../crop/...`.
3. Its corresponding reconstruction under `w_prompt_blip/.../inpainting/...`.

Raw benchmark images alone are insufficient. This inference workflow uses precomputed reconstructions and does not generate them automatically. The existing data layout may also contain `wo_prompt/.../inpainting/...`; the commands below do not read that branch.

For example, Chameleon must contain:

```text
dataset/Chameleon/Chameleon_recon/
├── wo_prompt/
│   └── crop/test/
│       ├── 0_real/<image files>
│       └── 1_fake/<image files>
└── w_prompt_blip/
    ├── crop/test/
    │   ├── 0_real/<matching image files>
    │   └── 1_fake/<matching image files>
    └── inpainting/test/
        ├── 0_real/<matching reconstruction files>
        └── 1_fake/<matching reconstruction files>
```

The other datasets follow these paths beneath **each** of `wo_prompt/` and `w_prompt_blip/`:

| Dataset | Image path beneath the branch | Real / fake folders |
| --- | --- | --- |
| GenImage | `<generator>/<subset>/crop/val/<class>/` | `nature` / `ai` |
| CommunityForensics | `<generator>/crop/test/<class>/` | `nature` / `ai` |
| Forensynths | `<generator>/crop/<class>/` | `0_real` / `1_fake` |
| Forensynths: cyclegan, progan, stylegan, stylegan2 | `<generator>/<subcategory>/crop/<class>/` | `0_real` / `1_fake` |
| Chameleon | `crop/test/<class>/` | `0_real` / `1_fake` |

Under `w_prompt_blip/`, place matching reconstructions at the same relative path with `crop/` replaced by `inpainting/`. Preserve sample basenames. For CommunityForensics, Forensynths, and Chameleon, the loader maps `.jpg` image names to `.png` in `w_prompt_blip/`; GenImage preserves the filename. The generator/subset names and their ordering are defined in `GenImage_LIST`, `CommunityForensics_LIST`, and `Forensynths_LIST` in `data/dataset.py`.

### 3. Run inference

Configure `scripts/test.sh`, then run it from the `scripts/` directory. The script calls the evaluation entry point with `--is_test` and evaluates each selected generator in turn.

**Select a model.** In `test.sh`, uncomment all three lines of the model block you want to use and comment out the other model block. Keep exactly one model block active.

For SARE-CLIP (the default):

```bash
### SARE_CLIP ###
MODEL_NAME=clip-ViT-L-14
MODEL_SUBPATH=SARE_CLIP_SD14
MODEL_PATH=../output/GenImage/1/${MODEL_SUBPATH}/weights/sare_clip_sd14.pth
```

For SARE-DINO:

```bash
### SARE_DINO ###
MODEL_NAME=dinov3-vith16plus
MODEL_SUBPATH=SARE_DINO_SD14
MODEL_PATH=../output/GenImage/1/${MODEL_SUBPATH}/weights/sare_dino_sd14.pth
```

For SARE-DINO, also set `DINOV3_WEIGHTS` as described in Installation before running the script. Change the complete model block, including `MODEL_SUBPATH`, so that results are saved under the correct model directory.

**Select a dataset.** Under `TEST_DATASET`, uncomment the `ROOT_PATH`, `DATASET_NAME`, and `FAKE_INDEXES` lines for the desired dataset and comment out the other dataset blocks. Keep exactly one dataset block active. Chameleon is selected by default:

```bash
ROOT_PATH=../dataset/Chameleon/Chameleon_recon/wo_prompt
DATASET_NAME=Chameleon
FAKE_INDEXES=(1)
```

The script already contains blocks for all four datasets:

| `DATASET_NAME` | `ROOT_PATH` (relative to `scripts/`) | `FAKE_INDEXES` |
| --- | --- | --- |
| `GenImage` | `../dataset/Genimage/Genimage_recon/wo_prompt` | `(1 2 3 4 5 6 7 8)` |
| `CommunityForensics` | `../dataset/CommunityForensics/CommunityForensics_recon/wo_prompt` | `(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21)` |
| `Forensynths` | `../dataset/Forensynths/Forensynths_recon/wo_prompt` | `(1 2 3 4 5 6 7 8 9 10 11 12)` |
| `Chameleon` | `../dataset/Chameleon/Chameleon_recon/wo_prompt` | `(1)` |

Keep the full index array to evaluate every generator, or select a subset, such as `FAKE_INDEXES=(1 2)`. The script runs each index separately.

**Set the GPU and batch size.** Set `DEVICE_ID=0` in `test.sh` to use the first GPU (the current script defaults to `1`). Reduce `BATCH_SIZE` from its default of `24` if GPU memory is insufficient. Keep `EMBEDDING_SIZE=1024` and `INPUT_SIZE=224` for the released checkpoints.

**Run the script.** From the repository root, with the `sare` environment activated:

```bash
cd scripts
bash test.sh
```

The script uses Bash arrays, so run it with `bash`. Run it from `scripts/` so that its relative paths resolve correctly.

Results are appended to the following path relative to the repository root:

```text
output/GenImage/1/<MODEL_SUBPATH>/<DATASET_NAME>.txt
```

For example, the default model and dataset save results to `output/GenImage/1/SARE_CLIP_SD14/Chameleon.txt`. Each evaluation records accuracy, AUC, recall, precision, F1, FNR, and EER. Labels are `0` for real and `1` for generated images; the default binary decision threshold is `0.5`. Both classes must be present to compute AUC. Results are aggregate metrics, not an exported list of per-image predictions. Repeated runs append to the same result file.

If the loader reports zero images or image-read errors, check the extraction hierarchy, filename matching, and the presence of both `wo_prompt` and `w_prompt_blip` data before interpreting the metrics.
