# default

### SARE_CLIP ###
MODEL_NAME=clip-ViT-L-14
MODEL_SUBPATH=SARE_CLIP_SD14
MODEL_PATH=../output/GenImage/1/${MODEL_SUBPATH}/weights/sare_clip_sd14.pth

# ### SARE_DINO ###
# MODEL_NAME=dinov3-vith16plus
# MODEL_SUBPATH=SARE_DINO_SD14
# MODEL_PATH=../output/GenImage/1/${MODEL_SUBPATH}/weights/sare_dino_sd14.pth


### SETTINGS ###
DEVICE_ID=1
EMBEDDING_SIZE=1024
MODEL_NAME=${1:-$MODEL_NAME}
MODEL_PATH=${2:-$MODEL_PATH}
DEVICE_ID=${3:-$DEVICE_ID}
EMBEDDING_SIZE=${4:-$EMBEDDING_SIZE}


### TEST_DATASET ###

# ROOT_PATH=../dataset/Genimage/Genimage_recon/wo_prompt
# DATASET_NAME=GenImage
# FAKE_INDEXES=(1 2 3 4 5 6 7 8)

# ROOT_PATH=../dataset/CommunityForensics/CommunityForensics_recon/wo_prompt
# DATASET_NAME=CommunityForensics
# FAKE_INDEXES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21)

# ROOT_PATH=../dataset/Forensynths/Forensynths_recon/wo_prompt
# DATASET_NAME=Forensynths
# FAKE_INDEXES=(1 2 3 4 5 6 7 8 9 10 11 12)

ROOT_PATH=../dataset/Chameleon/Chameleon_recon/wo_prompt
DATASET_NAME=Chameleon
FAKE_INDEXES=(1)


####################
FAKE_ROOT_PATH=""
SAVE_TXT=../output/GenImage/1/${MODEL_SUBPATH}/${DATASET_NAME}.txt
INPUT_SIZE=224
BATCH_SIZE=24
####################
for FAKE_INDEX in ${FAKE_INDEXES[@]}
do
  echo FAKE_INDEX:${FAKE_INDEX},MODEL_NAME:${MODEL_NAME},MODEL_PATH:${MODEL_PATH}
  python train.py --root_path ${ROOT_PATH} --fake_root_path '' --model_name ${MODEL_NAME} \
                  --input_size ${INPUT_SIZE} --batch_size ${BATCH_SIZE} --device_id ${DEVICE_ID} --is_test \
                  --model_path ${MODEL_PATH} --is_crop --fake_indexes ${FAKE_INDEX} \
                  --dataset_name ${DATASET_NAME} --save_txt ${SAVE_TXT} --embedding_size ${EMBEDDING_SIZE}
done