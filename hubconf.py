import os.path
from typing import Literal

from transformers import CLIPVisionModelWithProjection, CLIPTextModel, AutoConfig

from src.models.ConvNet_TPS import ConvNet_TPS
from src.models.UNet import UNetVanilla
from src.models.emasc import EMASC

dependencies = ['torch', 'diffusers', 'transformers']
import torch
from diffusers import UNet2DConditionModel
from src.models.inversion_adapter import InversionAdapter


def inversion_adapter(dataset: Literal['dresscode', 'vitonhd']):
    config = AutoConfig.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    text_encoder_config =  UNet2DConditionModel.load_config("stabilityai/stable-diffusion-2-inpainting", subfolder="text_encoder")
    inversion_adapter = InversionAdapter(input_dim=config.vision_config.hidden_size,
                                         hidden_dim=config.vision_config.hidden_size * 4,
                                         output_dim=text_encoder_config['hidden_size'] * 16,
                                         num_encoder_layers=1,
                                         config=config.vision_config)

    checkpoint_url = f"https://github.com/miccunifi/ladi-vton/releases/download/weights/inversion_adapter_{dataset}.pth"
    inversion_adapter.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu'))
    return inversion_adapter


def extended_unet(dataset: Literal['dresscode', 'vitonhd']):
    config = UNet2DConditionModel.load_config("stabilityai/stable-diffusion-2-inpainting", subfolder="unet")
    config['in_channels'] = 31
    unet = UNet2DConditionModel.from_config(config)

    checkpoint_url = f"https://github.com/miccunifi/ladi-vton/releases/download/weights/unet_{dataset}.pth"
    unet.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu'))
    return unet


def emasc(dataset: Literal['dresscode', 'vitonhd']):
    in_feature_channels = [128, 128, 128, 256, 512]
    out_feature_channels = [128, 256, 512, 512, 512]

    emasc = EMASC(in_feature_channels,
                  out_feature_channels,
                  kernel_size=3,
                  padding=1,
                  stride=1,
                  type='nonlinear')

    checkpoint_url = f"https://github.com/miccunifi/ladi-vton/releases/download/weights/emasc_{dataset}.pth"
    emasc.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu'))
    return emasc


def warping_module(dataset: Literal['dresscode', 'vitonhd']):
    tps = ConvNet_TPS(256, 192, 21, 3)
    refinement = UNetVanilla(n_channels=24, n_classes=3, bilinear=True)

    checkpoint_url = f"https://github.com/miccunifi/ladi-vton/releases/download/weights/warping_{dataset}.pth"
    tps.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu')['tps'])
    refinement.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu')['refinement'])

    return tps, refinement
