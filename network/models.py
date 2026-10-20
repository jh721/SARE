import os
import sys
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import timm
from transformers import CLIPModel
import clip



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_imagenet_weight(_conv_stem_weight, input_channel=3):
    for i in range(input_channel // 3):
        if i == 0:
            _conv_stem_weight_new = _conv_stem_weight
        else:
            _conv_stem_weight_new = torch.cat([_conv_stem_weight_new, _conv_stem_weight], axis=1)

    return torch.nn.Parameter(_conv_stem_weight_new)


class CLIPVisual(nn.Module):
    def __init__(self, model_name, num_classes=2, freeze_extractor=False):
        super(CLIPVisual, self).__init__()
        model = CLIPModel.from_pretrained(model_name)
        self.visual_model = model.vision_model
        if freeze_extractor:
            self.freeze(self.visual_model)
        self.fc = nn.Linear(in_features=model.vision_embed_dim, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.visual_model(x)
        x = self.fc(x[1])

        return x

    @staticmethod
    def freeze(model):
        for param in model.parameters():
            param.requires_grad = False


class CLIPModelV2(nn.Module):
    CHANNELS = {
        "RN50": 1024,
        "ViT-B/32": 512,
        "ViT-L/14": 768
    }

    def __init__(self, name='clip-RN50', num_classes=2, freeze_extractor=False):
        super(CLIPModelV2, self).__init__()
        name = name.replace('clip-', '').replace('L-', 'L/').replace('B-', 'B/')
        self.model, self.preprocess = clip.load(name, device="cpu")
        if freeze_extractor:
            self.freeze(self.model)
            print(f'Freezing the feature extractors!')

        self.fc = nn.Linear(self.CHANNELS[name], num_classes)

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x)
        if return_feature:
            return features
        return self.fc(features)

    @staticmethod
    def freeze(model):
        for param in model.parameters():
            param.requires_grad = False


class CrossAttentionModule(nn.Module):
    def __init__(self, dim=1024, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query_feat, cond_feat):
        # [B, D] → [B, 1, D]
        q = query_feat.unsqueeze(1)
        k = cond_feat.unsqueeze(1)
        v = cond_feat.unsqueeze(1)
        attn_out, _ = self.attn(q, k, v)
        attn_out = self.dropout(attn_out)
        return self.norm(attn_out + q).squeeze(1)  # residual + squeeze


class ContrastiveModels(nn.Module):
    def __init__(self, model_name, num_classes=2, pretrained=True, embedding_size=1024,
                 freeze_extractor=False):
        super(ContrastiveModels, self).__init__()
        self.model_name = model_name
        self.embedding_size = embedding_size
        self.model = get_models(model_name=model_name, pretrained=pretrained, num_classes=embedding_size,
                                freeze_extractor=freeze_extractor)
        self.model_dire = get_models(model_name='resnet50', pretrained=pretrained, num_classes=embedding_size)

        self.cross_attention = CrossAttentionModule(dim=embedding_size)
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x, dire, return_feature=False):
        feature = self.model(x)
        dire_feature = self.model_dire(dire)
        fused_feature = self.cross_attention(feature, dire_feature)

        y_pred = self.fc(fused_feature)
        if return_feature:
            return y_pred, feature

        return y_pred

    def extract_feature(self, x):
        feature = self.model(x)

        return feature

class DINOv3Visual(nn.Module):
    HUB_ALIASES = {
        'dinov3-vith+': 'dinov3_vith16plus',
        'dinov3-vith16plus': 'dinov3_vith16plus',
        'dinov3_vith16plus': 'dinov3_vith16plus',
    }

    FEATURE_DIMS = {
        'dinov3_vits16': 384,
        'dinov3_vits16plus': 384,
        'dinov3_vitb16': 768,
        'dinov3_vitl16': 1024,
        'dinov3_vith16plus': 1280,
    }

    def __init__(self, model_name='dinov3-vith16plus', num_classes=1024,
                 pretrained=True, freeze_extractor=False):
        super(DINOv3Visual, self).__init__()
        self.model_name = model_name
        self.backbone, feature_dim = self.load_backbone(model_name, pretrained=pretrained)

        if freeze_extractor:
            self.freeze(self.backbone)
            print('Freezing the DINOv3 feature extractor!')

        self.fc = nn.Linear(feature_dim, num_classes)

    @classmethod
    def load_backbone(cls, model_name, pretrained=True):
        model_key = model_name.lower()
        hub_name = cls.HUB_ALIASES.get(model_key)
        weights = os.environ.get('DINOV3_WEIGHTS')
        hub_kwargs = {'pretrained': pretrained}
        if weights:
            hub_kwargs['weights'] = weights

        if hub_name is not None:
            cls.require_python310_for_torch_hub(model_name)
            backbone = torch.hub.load('facebookresearch/dinov3', hub_name, **hub_kwargs)
            return backbone, cls.FEATURE_DIMS[hub_name]

        if model_key in cls.FEATURE_DIMS:
            cls.require_python310_for_torch_hub(model_name)
            backbone = torch.hub.load('facebookresearch/dinov3', model_key, **hub_kwargs)
            return backbone, cls.FEATURE_DIMS[model_key]

        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError('Install transformers to load DINOv3 from a Hugging Face model id or local path.') from exc

        backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        config = getattr(backbone, 'config', None)
        feature_dim = getattr(config, 'hidden_size', None)
        if feature_dim is None:
            raise ValueError(f'Could not infer DINOv3 feature dimension from {model_name}.')

        return backbone, feature_dim

    @staticmethod
    def require_python310_for_torch_hub(model_name):
        if sys.version_info < (3, 10):
            version = f'{sys.version_info.major}.{sys.version_info.minor}'
            raise RuntimeError(
                f'{model_name} uses the official facebookresearch/dinov3 torch.hub code, '
                f'which requires Python >= 3.10. Current Python is {version}. '
                'Create a Python 3.10+ conda environment or pass a local/Hugging Face model path '
                'that can be loaded by transformers.AutoModel.'
            )


    def forward(self, x):
        features = self.forward_features(x)
        return self.fc(features)

    def forward_features(self, x):
        outputs = self.backbone(x)

        if isinstance(outputs, torch.Tensor):
            if outputs.ndim == 3:
                return outputs[:, 0]
            return outputs

        if isinstance(outputs, dict):
            if 'pooler_output' in outputs and outputs['pooler_output'] is not None:
                return outputs['pooler_output']
            if 'last_hidden_state' in outputs:
                return outputs['last_hidden_state'][:, 0]
            if 'x_norm_clstoken' in outputs:
                return outputs['x_norm_clstoken']

        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state[:, 0]
        if hasattr(outputs, 'x_norm_clstoken'):
            return outputs.x_norm_clstoken

        raise TypeError(f'Unsupported DINOv3 output type: {type(outputs)}')

    @staticmethod
    def freeze(model):
        for param in model.parameters():
            param.requires_grad = False


def get_efficientnet_ns(model_name='tf_efficientnet_b3_ns', pretrained=True, num_classes=2, start_down=True):
    """
     # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    :param model_name:
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    if not start_down:
        net.conv_stem.stride = (1, 1)
    n_features = net.classifier.in_features
    net.classifier = nn.Linear(n_features, num_classes)

    return net


def get_swin_transformers(model_name='swin_base_patch4_window7_224', pretrained=True, num_classes=2):
    """
    :param model_name: swin_base_patch4_window12_384   swin_base_patch4_window7_224 swin_base_patch4_window7_224_in22k
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.head.in_features
    net.head = nn.Linear(n_features, num_classes)

    return net


def get_convnext(model_name='convnext_base_in22k', pretrained=True, num_classes=2, in_channel=3):
    """
    :param model_name: convnext_base_384_in22ft1k, convnext_base_in22k
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.head.fc.in_features
    net.head.fc = nn.Linear(n_features, num_classes)

    if in_channel != 3:
        first_conv_weight = net.stem[0].weight
        first_out_channels = net.stem[0].out_channels
        first_conv = nn.Conv2d(in_channel, first_out_channels, kernel_size=4, stride=4)
        first_conv.weight = init_imagenet_weight(first_conv_weight, input_channel=in_channel)
        net.stem[0] = first_conv

    return net


def get_resnet(model_name='resnet200d', pretrained=True, num_classes=2):
    """
    :param model_name: resnet200d, input_size=512, resnet50
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.fc.in_features
    net.fc = nn.Linear(n_features, num_classes)

    return net


def get_clip_visual_model(model_name="openai/clip-vit-base-patch32", num_classes=2, pretrained=True,
                          freeze_extractor=False):
    if 'openai/clip' in model_name:
        model = CLIPVisual(model_name=model_name, num_classes=num_classes)
    else:
        # 'clip-' + 'name', clip-RN50, clip-ViT-L/14
        model = CLIPModelV2(name=model_name, num_classes=num_classes, freeze_extractor=freeze_extractor)

    return model


def get_dinov3_model(model_name='dinov3-vith16plus', num_classes=1024, pretrained=True,
                     freeze_extractor=False):
    return DINOv3Visual(model_name=model_name, num_classes=num_classes, pretrained=pretrained,
                        freeze_extractor=freeze_extractor)


def get_models(model_name='tf_efficientnet_b3_ns', pretrained=True, num_classes=2,
               in_channel=3, freeze_extractor=False, embedding_size=None):
    if embedding_size is not None and isinstance(embedding_size, int) and embedding_size > 0:
        model = ContrastiveModels(model_name, num_classes, pretrained, embedding_size, freeze_extractor)
    elif 'efficientnet' in model_name:
        model = get_efficientnet_ns(model_name, pretrained, num_classes)
    elif 'convnext' in model_name:
        model = get_convnext(model_name, pretrained, num_classes, in_channel=in_channel)
    elif 'swin' in model_name:
        model = get_swin_transformers(model_name, pretrained, num_classes)
    elif 'clip' in model_name:
        model = get_clip_visual_model(model_name, num_classes, freeze_extractor=freeze_extractor)
    elif 'dinov3' in model_name.lower():
        model = get_dinov3_model(model_name, num_classes, pretrained, freeze_extractor=freeze_extractor)
    elif 'swin' in model_name:
        model = get_swin_transformers(model_name, pretrained=pretrained, num_classes=num_classes)
    elif 'resnet' in model_name:
        model = get_resnet(model_name, pretrained, num_classes)
    else:
        raise NotImplementedError(model_name)

    return model


if __name__ == '__main__':
    import time
    image_size = 224
    model = get_models(model_name='clip-ViT-L-14', num_classes=2, pretrained=False,
                       embedding_size=512)  # clip-ViT-L-14
    print(model)
    # print(model.default_cfg)
    model = model.to(torch.device('cpu'))
    img = torch.randn(1, 3, image_size, image_size)  # your high resolution picture
    start = time.time()
    times = 1
    for _ in range(times):
        out = model(img)
        if isinstance(out, tuple):
            print([o.shape for o in out])
        else:
            print(out.shape)
    print((time.time()-start)/times)

    # from torchsummary import summary
    # input_s = (3, image_size, image_size)
    # print(summary(model, input_s, device='cpu'))
    pass
