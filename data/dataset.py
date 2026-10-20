import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import os
import torch
import glob
import json
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage.filters import gaussian_filter
try:
    from transform import create_train_transforms, create_val_transforms, create_sdie_transforms
except:
    from .transform import create_train_transforms, create_val_transforms, create_sdie_transforms

GenImage_LIST = ['stable_diffusion_v_1_4/imagenet_ai_0419_sdv4', 'stable_diffusion_v_1_5/imagenet_ai_0424_sdv5',
                 'Midjourney/imagenet_midjourney', 'ADM/imagenet_ai_0508_adm', 'wukong/imagenet_ai_0424_wukong',
                 'glide/imagenet_glide', 'VQDM/imagenet_ai_0419_vqdm', 'BigGAN/imagenet_ai_0419_biggan']
CommunityForensics_LIST = [
    'Dalle2',
    'Dalle3',
    'DeciDiffusionV2',
    'DFGAN',
    'Firefly_Image2',
    'Firefly_Image3',
    'FLUX-dev',
    'FLUX-schnell',
    'GALIP',
    'Hourglass',
    'IdeogramV1',
    'IdeogramV2',
    'Imagen3',
    'kandinsky_2_2',
    'kvikontent_midjourney_v6',
    'LCM_lora_sdv15',
    'LCM_lora_sdxl',
    'LCM_lora_ssd1b',
    'MidjourneyV5_2',
    'MidjourneyV6_1',
    'stable_cascade',
]

Forensynths_LIST = [
    'biggan',
    'crn',
    'cyclegan',
    'deepfake',
    'gaugan',
    'imle',
    'progan',
    'san',
    'seeingdark',
    'stargan',
    'stylegan',
    'stylegan2',
]


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def cv2_scale(img, scale):
    h, w = img.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h))

    return resized_img


def resize_long_size(img, long_size=512):
    scale_percent = long_size / max(img.shape[0], img.shape[1])

    new_width = int(img.shape[1] * scale_percent)
    new_height = int(img.shape[0] * scale_percent)

    img_resized = cv2.resize(img, (new_width, new_height))

    return img_resized


def read_image(image_path, resize_size=None):
    try:
        if os.path.isfile(image_path):
            image = cv2.imread(image_path)
        else:
            ext = image_path.split(".")[-1]
            new_image_path = glob.glob(f"{image_path.replace(ext, '')}*")[0]
            image = cv2.imread(new_image_path)
        if resize_size is not None:
            image = resize_long_size(image, long_size=resize_size)
        # Revert from BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, True
    except:
        print(f'{image_path} read error!!!')
        return np.zeros(shape=(512, 512, 3), dtype=np.uint8), False


def shuffle_two_array(a, b, seed=None):
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(a)
    np.random.set_state(state)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(b)
    return a, b


def one_hot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


def split_data(image_paths, labels, val_split=0.1, test_split=0.0, phase='train', seed=2022):
    image_paths, labels = shuffle_two_array(image_paths, labels, seed=seed)
    total_len = len(image_paths)
    if test_split > 0:
        if phase == 'train':
            start_index, end_index = 0, int(total_len * (1 - val_split - test_split))
        elif phase == 'val':
            start_index, end_index = int(total_len * (1 - val_split - test_split)), int(total_len * (1 - test_split))
        else:
            start_index, end_index = int(total_len * (1 - test_split)), total_len
    else:
        if phase == 'train':
            start_index, end_index = 0, int(total_len * (1 - val_split))
        else:
            start_index, end_index = int(total_len * (1 - val_split)), total_len
    image_paths, labels = image_paths[start_index:end_index], labels[start_index:end_index]

    return image_paths, labels


def split_dir(image_dirs, val_split=0.1, phase='train', seed=2022):
    if phase == 'all':
        return image_dirs
    image_dirs, _ = shuffle_two_array(image_dirs, image_dirs, seed=seed)
    total_len = len(image_dirs)
    if phase == 'train':
        start_index, end_index = 0, int(total_len * (1 - val_split * 2))
    elif phase == 'val':
        start_index, end_index = int(total_len * (1 - val_split * 2)), int(total_len * (1 - val_split))
    else:
        start_index, end_index = int(total_len * (1 - val_split)), total_len
    image_dirs = image_dirs[start_index:end_index]

    return image_dirs


def find_images(dir_path, extensions=['.jpg', '.png', '.jpeg', '.bmp']):
    image_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if os.path.basename(file).startswith("._"):  # skip files that start with "._"
                continue
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))

    return image_files


# Calculate the DIRE
def calculate_dire(img, sdir_img, is_success=True, input_size=224, phase='train'):
    if not is_success:
        return torch.zeros(size=(3, input_size, input_size), dtype=torch.float32)
    sdie_transforms = create_sdie_transforms(size=input_size, phase=phase)

    data = sdie_transforms(image=img, rec_image=sdir_img)
    img, sdir_img = data['image'], data['rec_image']

    # norm [0,255] -> [-1, 1]
    img = img / 127.5 - 1
    sdir_img = sdir_img / 127.5 - 1
    # absolute error
    sdie = np.abs(img - sdir_img)
    sdie_tensor = torch.from_numpy(np.transpose(np.array(sdie, dtype=np.float32), [2, 0, 1]))

    return sdie_tensor


def load_CommunityForensics(root_path='', phase='train', seed=2023,
                  indexes='1,2,3,4,5,6,7,8', val_split=0.1):
    indexes = [int(i) - 1 for i in indexes.split(',')]
    dir_list = CommunityForensics_LIST
    selected_dir_list = [dir_list[i] for i in indexes]
    real_images, real_labels, fake_images, fake_labels = [], [], [], []
    dir_phase = phase
    for i, selected_dir in enumerate(selected_dir_list):
        real_root = os.path.join(root_path, selected_dir, dir_phase, 'nature')
        fake_root = os.path.join(root_path, selected_dir, dir_phase, 'ai')

        real_root = real_root.replace(dir_phase, f'crop/{dir_phase}')
        fake_root = fake_root.replace(dir_phase, f'crop/{dir_phase}')

        real_images_t = sorted(glob.glob(f'{real_root}/*.*'))
        fake_images_t = sorted(glob.glob(f'{fake_root}/*.*'))
        if phase != 'test':
            real_images_t, _ = split_data(real_images_t, real_images_t, val_split, phase=phase, seed=seed)
            fake_images_t, _ = split_data(fake_images_t, fake_images_t, val_split, phase=phase, seed=seed)
        real_images += real_images_t
        real_labels += [0 for _ in real_images_t]
        fake_images += fake_images_t
        fake_labels += [i + 1 for _ in fake_images_t]
        print(f'phase:{phase}, real:{len(real_images_t)}, fake-{i+1}:{len(fake_images_t)}, selected_dir:{selected_dir}')
    total_images = real_images + fake_images
    labels = real_labels + fake_labels
    print(f'phase:{phase}, real:{len(real_images)}, fake:{len(fake_images)}')

    image_paths_drct = total_images

    image_paths_dire = []
    for img_path in total_images:
        path_org = img_path.replace('/wo_prompt/', '/w_prompt_blip/').replace("jpg", "png")
        path_recon = path_org.replace('crop', 'inpainting')
        image_paths_dire.append((path_org, path_recon))

    return image_paths_drct, image_paths_dire, labels


def load_Forensynths(root_path='', phase='train', seed=2023,
                  indexes='1,2,3,4,5,6,7,8', val_split=0.1):
    indexes = [int(i) - 1 for i in indexes.split(',')]
    dir_list = Forensynths_LIST
    selected_dir_list = [dir_list[i] for i in indexes]
    if selected_dir_list[0] in ['cyclegan', 'progan', 'stylegan', 'stylegan2']:
        selected_dir_list = glob.glob(f'{root_path}/{selected_dir_list[0]}/*')
        selected_dir_list = [selected_dir.replace(f'{root_path}/', '') for selected_dir in selected_dir_list]
    real_images, real_labels, fake_images, fake_labels = [], [], [], []
    dir_phase = phase
    for i, selected_dir in enumerate(selected_dir_list):
        real_root = os.path.join(root_path, selected_dir, '0_real')
        fake_root = os.path.join(root_path, selected_dir, '1_fake')

        real_root = real_root.replace(selected_dir, f'{selected_dir}/crop')
        fake_root = fake_root.replace(selected_dir, f'{selected_dir}/crop')

        real_images_t = sorted(glob.glob(f'{real_root}/*.*'))
        fake_images_t = sorted(glob.glob(f'{fake_root}/*.*'))
        if phase != 'test':
            real_images_t, _ = split_data(real_images_t, real_images_t, val_split, phase=phase, seed=seed)
            fake_images_t, _ = split_data(fake_images_t, fake_images_t, val_split, phase=phase, seed=seed)
        real_images += real_images_t
        real_labels += [0 for _ in real_images_t]
        fake_images += fake_images_t
        fake_labels += [i + 1 for _ in fake_images_t]
        print(f'phase:{phase}, real:{len(real_images_t)}, fake-{i+1}:{len(fake_images_t)}, selected_dir:{selected_dir}')
    total_images = real_images + fake_images
    labels = real_labels + fake_labels
    print(f'phase:{phase}, real:{len(real_images)}, fake:{len(fake_images)}')

    image_paths_drct = total_images

    image_paths_dire = []
    for img_path in total_images:
        path_org = img_path.replace('wo_prompt', 'w_prompt_blip').replace("jpg", "png")
        path_recon = path_org.replace('crop', 'inpainting')
        image_paths_dire.append((path_org, path_recon))

    return image_paths_drct, image_paths_dire, labels


def load_Chameleon(root_path='/disk1/chenby/dataset/AIGC_data/GenImage', phase='train', seed=2023,
                  indexes='1,2,3,4,5,6,7,8', val_split=0.1):
    indexes = [int(i) - 1 for i in indexes.split(',')]
    real_images, real_labels, fake_images, fake_labels = [], [], [], []
    dir_phase = 'test'
    real_root = os.path.join(root_path, 'crop', dir_phase, '0_real')
    fake_root = os.path.join(root_path, 'crop', dir_phase, '1_fake')

    real_images_t = sorted(glob.glob(f'{real_root}/*.*'))
    fake_images_t = sorted(glob.glob(f'{fake_root}/*.*'))
    if phase != 'test':
        real_images_t, _ = split_data(real_images_t, real_images_t, val_split, phase=phase, seed=seed)
        fake_images_t, _ = split_data(fake_images_t, fake_images_t, val_split, phase=phase, seed=seed)
    real_images += real_images_t
    real_labels += [0 for _ in real_images_t]
    fake_images += fake_images_t
    fake_labels += [1 for _ in fake_images_t]
    print(f'phase:{phase}, real:{len(real_images_t)}, fake:{len(fake_images_t)}')
    total_images = real_images + fake_images
    labels = real_labels + fake_labels
    print(f'phase:{phase}, real:{len(real_images)}, fake:{len(fake_images)}')

    image_paths_drct = total_images

    image_paths_dire = []
    for img_path in total_images:
        path_org = img_path.replace('/wo_prompt/', '/w_prompt_blip/').replace('.jpg', '.png')
        path_recon = path_org.replace('crop', 'inpainting')
        image_paths_dire.append((path_org, path_recon))

    return image_paths_drct, image_paths_dire, labels


def load_normal_data(root_path, val_split, seed, phase='train', regex='*.*', test_all=False):
    images_t = sorted(glob.glob(f'{root_path}/{regex}'))

    # with open('/data/jykang/research_models/Deepfake/invalid_images.json', 'r') as f:
    #     exclude_dict = json.load(f)
    # exclude_paths = set()
    # for path_list in exclude_dict.values():
    #     exclude_paths.update(path_list)

    # images_t = [img for img in images_t if img not in exclude_paths]

    if not test_all:
        images_t, _ = split_data(images_t, images_t, val_split=val_split, phase=phase, seed=seed)

    captions_t = [' ' for _ in images_t]
    print(f'{root_path}: {len(images_t)}')
    return images_t, captions_t


def load_GenImage(root_path='/disk1/chenby/dataset/AIGC_data/GenImage', phase='train', seed=2023,
                  indexes='1,2,3,4,5,6,7,8', val_split=0.1):

    indexes = [int(i) - 1 for i in indexes.split(',')]
    dir_list = GenImage_LIST
    selected_dir_list = [dir_list[i] for i in indexes]
    real_images, real_labels, fake_images, fake_labels = [], [], [], []
    dir_phase = 'train' if phase != 'test' else 'val'
    for i, selected_dir in enumerate(selected_dir_list):
        real_root = os.path.join(root_path, selected_dir, dir_phase, 'nature')
        fake_root = os.path.join(root_path, selected_dir, dir_phase, 'ai')

        real_root = real_root.replace('val', 'crop/val')
        fake_root = fake_root.replace('val', 'crop/val')

        real_images_t = sorted(glob.glob(f'{real_root}/*.*'))
        fake_images_t = sorted(glob.glob(f'{fake_root}/*.*'))
        if phase != 'test':
            real_images_t, _ = split_data(real_images_t, real_images_t, val_split, phase=phase, seed=seed)
            fake_images_t, _ = split_data(fake_images_t, fake_images_t, val_split, phase=phase, seed=seed)
        real_images += real_images_t
        real_labels += [0 for _ in real_images_t]
        fake_images += fake_images_t
        fake_labels += [i + 1 for _ in fake_images_t]
        print(f'phase:{phase}, real:{len(real_images_t)}, fake-{i+1}:{len(fake_images_t)}, selected_dir:{selected_dir}')
    total_images = real_images + fake_images
    labels = real_labels + fake_labels
    print(f'phase:{phase}, real:{len(real_images)}, fake:{len(fake_images)}')

    image_paths_drct = total_images

    image_paths_dire = []
    for img_path in total_images:
        # path_org = img_path.replace('Genimage_recon/wo_prompt/10_50_over0', 'Genimage_recon2/w_BLIP/75_5_50_fix512')
        path_org = img_path.replace('/wo_prompt/', '/w_prompt_blip/')
        path_recon = path_org.replace('crop', 'inpainting')
        image_paths_dire.append((path_org, path_recon))

    return image_paths_drct, image_paths_dire, labels


def load_data(real_root_path, fake_root_path,
              phase='train', val_split=0.1, seed=2022, ):
    # load real images
    total_real_images, total_real_captions = [], []
    for real_root in real_root_path.split(','):
        real_images_t, real_captions_t = load_normal_data(real_root, val_split, seed, phase)
        total_real_images += list(real_images_t)
        total_real_captions += list(real_captions_t)
    # load fake images
    total_fake_images, total_fake_captions = [], []
    for fake_root in fake_root_path.split(','):
        fake_images_t, fake_captions_t = load_normal_data(fake_root, val_split, seed, phase)
        total_fake_images += list(fake_images_t)
        total_fake_captions += list(fake_captions_t)
    image_paths = total_real_images + total_fake_images
    labels = [0 for _ in total_real_images] + [1 for _ in total_fake_images]
    print(f'{phase}-total:{len(image_paths)}, real:{len(total_real_images)},fake:{len(total_fake_images)}')

    # Image paths for drct (w/o text)
    image_paths_drct = image_paths

    # Image paths for dire (w/ text)
    image_paths_dire = []
    if phase == 'train':
        for img_path in image_paths:
            path_org = img_path.replace('Genimage_recon/wo_prompt/10_50_over0', 'Genimage_reconXL/w_BLIP/50_3_50_over0').replace('inpainting', 'crop')
            path_recon = path_org.replace('crop', 'inpainting')
            image_paths_dire.append((path_org, path_recon))
    else:
        for img_path in image_paths:
            path_org = img_path.replace('Genimage_recon/wo_prompt/10_50_over0', 'Genimage_reconXL/w_BLIP/50_3_50_fix512').replace('inpainting', 'crop')
            path_recon = path_org.replace('crop', 'inpainting')
            image_paths_dire.append((path_org, path_recon))

    return image_paths_drct, image_paths_dire, labels


def load_pair_data(root_path, fake_root_path=None, phase='train', seed=2023, fake_indexes='1',
                   inpainting_dir='full_inpainting'):
    if fake_root_path is None:
        assert len(root_path.split(',')) == 2
        root_path, rec_root_path = root_path.split(',')[:2]
        image_paths = sorted(glob.glob(f"{root_path}/*.*"))
        rec_image_paths = sorted(glob.glob(f"{rec_root_path}/*.*"))
        assert len(image_paths) == len(rec_image_paths)
        total_paths = []
        for image_path, rec_image_path in zip(image_paths, rec_image_paths):
            total_paths.append((image_path, rec_image_path))
        print(f'Pair data-{phase}:{len(total_paths)}.')
        return total_paths
    assert (len(root_path.split(',')) == 2 and len(fake_root_path.split(',')) == 2) or \
           (root_path == fake_root_path and 'GenImage' in root_path)
    real_root, real_rec_root = root_path.split(',')[:2]
    fake_root, fake_rec_root = fake_root_path.split(',')[:2]

    image_paths, labels = [], []
    # load real images
    real_image_paths = sorted(glob.glob(f"{real_root}/*.*"))
    real_image_rec_paths = sorted(glob.glob(f"{real_rec_root}/*.*"))
    assert len(real_image_paths) == len(real_image_rec_paths) and len(real_image_paths) > 0
    total_real = len(real_image_paths)
    if phase != 'test':
        real_image_paths, real_image_rec_paths = split_data(real_image_paths, real_image_rec_paths, phase=phase, seed=seed)
    for real_image_path, real_image_rec_path in zip(real_image_paths, real_image_rec_paths):
        image_paths.append((real_image_path, real_image_rec_path))
    # load fake images
    fake_image_paths = sorted(glob.glob(f"{fake_root}/*.*"))
    fake_image_rec_paths = sorted(glob.glob(f"{fake_rec_root}/*.*"))
    assert len(fake_image_paths) == len(fake_image_rec_paths) and len(fake_image_paths) > 0
    total_fake = len(fake_image_paths)
    if phase != 'test':
        fake_image_paths, fake_image_rec_paths  = split_data(fake_image_paths, fake_image_rec_paths, phase=phase, seed=seed)
    for fake_image_path, fake_image_rec_path in zip(fake_image_paths, fake_image_rec_paths):
        image_paths.append((fake_image_path, fake_image_rec_path))
    labels = [0 for _ in range(len(real_image_paths))] + [1 for _ in range(len(fake_image_paths))]
    print(f'Phase:{phase}, real:{len(real_image_paths)}, fake:{len(fake_image_paths)},'
          f'Total real:{total_real}, fake:{total_fake}')

    return image_paths, labels


class AIGCDetectionDataset(Dataset):
    def __init__(self, root_path='/disk4/chenby/dataset/MSCOCO', fake_root_path='/disk4/chenby/dataset/DRCT-2M',
                 fake_indexes='1,2,3,4,5,6', phase='train', is_one_hot=False, seed=2021,
                 transform=None, use_label=True, num_classes=None, regex='*.*',
                 is_dire=False, inpainting_dir='full_inpainting', post_aug_mode=None):
        self.root_path = root_path
        self.phase = phase
        self.is_one_hot = is_one_hot
        self.num_classes = len(fake_indexes.split(',')) + 1 if num_classes is None else num_classes
        self.transform = transform
        self.use_label = use_label
        self.regex = regex
        self.is_dire = is_dire
        self.post_aug_mode = post_aug_mode
        self.seed = seed

        if use_label:
            if self.is_dire:
                # load DR data
                self.image_paths, self.labels = load_pair_data(root_path, fake_root_path, phase,
                                                               fake_indexes=fake_indexes,
                                                               inpainting_dir=inpainting_dir)
            elif 'Genimage' in root_path and fake_root_path == '':
                self.image_paths, self.image_paths_dire, self.labels = load_GenImage(root_path=root_path, phase=phase, seed=seed,
                                                              indexes=fake_indexes)
            elif 'CommunityForensics' in root_path:
                self.image_paths, self.image_paths_dire, self.labels = load_CommunityForensics(root_path=root_path, phase=phase, seed=seed, indexes=fake_indexes)
            elif 'Forensynths' in root_path:
                self.image_paths, self.image_paths_dire, self.labels = load_Forensynths(root_path=root_path, phase=phase, seed=seed, indexes=fake_indexes)
            elif 'Chameleon' in root_path:
                self.image_paths, self.image_paths_dire, self.labels = load_Chameleon(root_path=root_path, phase=phase, seed=seed, indexes=fake_indexes)
            else:
                root_dir = root_path
                root_path = f'{root_dir}/wo_prompt/10_50_over0/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/crop/train/nature'
                fake_root_path = f'{root_dir}/wo_prompt/10_50_over0/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/inpainting/train/nature,' \
                                 f'{root_dir}/wo_prompt/10_50_over0/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/crop/train/ai,' \
                                 f'{root_dir}/wo_prompt/10_50_over0/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/inpainting/train/ai'

                self.image_paths, self.image_paths_dire, self.labels = load_data(real_root_path=root_path, fake_root_path=fake_root_path,
                                                          phase=phase, seed=seed)

            self.labels = [int(label > 0)for label in self.labels] if self.num_classes == 2 else self.labels
        else:
            if len(root_path.split(',')) == 2 and 'DR' in root_path:
                self.is_dire = True
                self.image_paths = load_pair_data(root_path, phase=phase, fake_indexes=fake_indexes,
                                                  inpainting_dir=inpainting_dir)
            else:
                if self.regex == 'all':
                    self.image_paths = sorted(find_images(dir_path=root_path, extensions=['.jpg', '.png', '.jpeg', '.bmp']))
                else:
                    self.image_paths = sorted(glob.glob(f'{root_path}/{self.regex}'))[:]
            print(f'Total predict images:{len(self.image_paths)}, regex:{self.regex}')
        if self.phase == 'test' and self.post_aug_mode is not None:
            print(f"post_aug_mode:{self.post_aug_mode}, {self.post_aug_mode.split('_')[1]}")

    def __len__(self):
        return len(self.image_paths)

    def get_labels(self):
        return list(self.labels)

    def __getitem__(self, index):
        if not self.is_dire:
            image_path = self.image_paths[index]
            image, is_success = read_image(image_path)
            org_image_path, rec_image_path = self.image_paths_dire[index]
            org_image, is_success = read_image(org_image_path)
            rec_image, is_success = read_image(rec_image_path)
        else:
            image_path, rec_image_path = self.image_paths[index]
            image, is_success = read_image(image_path)
            rec_image, rec_is_success = read_image(rec_image_path)
            is_success = is_success and rec_is_success
            image = calculate_dire(image, rec_image, phase=self.phase)

        if self.phase == 'test' and self.post_aug_mode is not None:
            if 'jpeg' in self.post_aug_mode:
                compress_val = int(self.post_aug_mode.split('_')[1])
                image = cv2_jpg(image, compress_val)
            elif 'scale' in self.post_aug_mode:
                scale = float(self.post_aug_mode.split('_')[1])
                image = cv2_scale(image, scale)

        label = 0  # default label
        if self.use_label:
            label = self.labels[index] if is_success else 0

        if self.transform is not None and not self.is_dire:
            try:
                if isinstance(self.transform, torchvision.transforms.transforms.Compose):
                    image = self.transform(Image.fromarray(image))
                else:
                    if self.phase == "train":
                        data = self.transform(image=image, rec_image=rec_image, org_image=org_image)
                        image = data["image"]
                        rec_image = data["rec_image"]
                        org_image = data["org_image"]
                    else:
                        image = self.transform(image=image)["image"]
                        rec_image = self.transform(image=rec_image)["image"]
                        org_image = self.transform(image=org_image)["image"]
            except:
                print("transform error!!!")
                image = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
                if isinstance(self.transform, torchvision.transforms.transforms.Compose):
                    image = self.transform(Image.fromarray(image))
                else:
                    data = self.transform(image=image)
                    image = data["image"]
                label = 0

        dire = torch.abs(org_image - rec_image)

        if not self.use_label:
            return image, image_path.replace(f"{self.root_path}", '')  # os.path.basename(image_path)

        if self.is_one_hot:
            label = one_hot(self.num_classes, label)

        return image, dire, label




