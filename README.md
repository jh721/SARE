# SARE

## Installation

Download or clone this repository, then run the following commands from its root directory:

```bash
conda create -n sare python=3.10 -y
conda activate sare
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

For SARE-DINO, download the DINOv3 ViT-H+/16 pretrained weights from the [official DINOv3 repository](https://github.com/facebookresearch/dinov3) and place the file at:

```text
dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
```

Set this environment variable **from the repository root**, before entering `scripts/`:

```bash
export DINOV3_WEIGHTS="$(pwd)/dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
```

## Inference

### 1. Download the SARE checkpoints

Download the checkpoint for the model you want to evaluate and place it at the exact path below.

| Model | `--model_name` | Checkpoint path relative to the repository root |
| --- | --- | --- |
| SARE-CLIP | `clip-ViT-L-14` | `output/Genimage/1/SARE_CLIP_SD14/weights/sare_clip_sd14.pth` |
| SARE-DINO | `dinov3-vith16plus` | `output/Genimage/1/SARE_DINO_SD14/weights/sare_dino_sd14.pth` |

Download both SARE models and the DINOv3 backbone from [GitHub Releases (v1.0.0)](https://github.com/jh721/SARE/releases/tag/v1.0.0) by running the following command from the repository root:

```bash
bash scripts/download_checkpoints.sh
```

### 2. Download and place the evaluation data

Download the prepared evaluation data from [SARE-DB/SARE-evaluation on Hugging Face](https://huggingface.co/datasets/SARE-DB/SARE-evaluation).

**The data release is being uploaded progressively.** Before downloading a dataset, check that its name appears in `completed_datasets` in the repository's `manifest.json`. A partially uploaded dataset is not ready for evaluation. The manifest and `SHA256SUMS` are updated after each dataset is fully uploaded and verified.

Run the download script with a dataset name:

```bash
bash scripts/download_dataset.sh Chameleon
bash scripts/download_dataset.sh Genimage
bash scripts/download_dataset.sh CommunityForensics
bash scripts/download_dataset.sh Forensynths
```

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

### 3. Run inference

Configure `scripts/test.sh`, then run it from the `scripts/` directory.

**Select a model.**

For SARE-CLIP:

```bash
### SARE_CLIP ###
MODEL_NAME=clip-ViT-L-14
MODEL_SUBPATH=SARE_CLIP_SD14
MODEL_PATH=../output/Genimage/1/${MODEL_SUBPATH}/weights/sare_clip_sd14.pth
```

For SARE-DINO:

```bash
### SARE_DINO ###
MODEL_NAME=dinov3-vith16plus
MODEL_SUBPATH=SARE_DINO_SD14
MODEL_PATH=../output/Genimage/1/${MODEL_SUBPATH}/weights/sare_dino_sd14.pth
```

**Select a dataset.**

```bash
ROOT_PATH=../dataset/Chameleon/Chameleon_recon/wo_prompt
DATASET_NAME=Chameleon
FAKE_INDEXES=(1)
```

| `DATASET_NAME` | `ROOT_PATH` | `FAKE_INDEXES` |
| --- | --- | --- |
| `Genimage` | `../dataset/Genimage/Genimage_recon/wo_prompt` | `(1 2 3 4 5 6 7 8)` |
| `CommunityForensics` | `../dataset/CommunityForensics/CommunityForensics_recon/wo_prompt` | `(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21)` |
| `Forensynths` | `../dataset/Forensynths/Forensynths_recon/wo_prompt` | `(1 2 3 4 5 6 7 8 9 10 11 12)` |
| `Chameleon` | `../dataset/Chameleon/Chameleon_recon/wo_prompt` | `(1)` |

**Run the script.**

```bash
cd scripts
bash test.sh
```
