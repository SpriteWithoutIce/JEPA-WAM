"""
vjepa_vit.py

Vision backbone using V-JEPA 2.1 encoder for spatial-temporal feature extraction.
Replaces DinoSigLIP in the Prismatic VLA framework.
"""

from functools import partial
from typing import Callable, Tuple

import torch
import torch.nn as nn
from PIL.Image import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from prismatic.models.backbones.vision.base_vision import ImageTransform, VisionBackbone

# Lazy import to avoid hard dependency
_VJEPA_ENCODER_CLS = None


def _get_vjepa_encoder_cls():
    global _VJEPA_ENCODER_CLS
    if _VJEPA_ENCODER_CLS is None:
        from vjepa21.extractor import VJEPA21Encoder
        _VJEPA_ENCODER_CLS = VJEPA21Encoder
    return _VJEPA_ENCODER_CLS


class VJEPAImageTransform:
    """ImageNet normalization transform for V-JEPA."""

    def __init__(self, image_size: int = 224) -> None:
        self.image_size = image_size
        self.transform = Compose([
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img: Image, **kwargs) -> torch.Tensor:
        return self.transform(img.convert("RGB"))


class VJEPAVisionBackbone(VisionBackbone):
    """V-JEPA 2.1 Vision Backbone.

    Forward input:  [B, V, 3, H, W]  (current frame, multi-view)
    Forward output: [B, V*196, 1024]

    Future encoding: [B, V, 8, 3, H, W] -> [B, V, 4, 14, 14, 1024]
    """

    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        image_sequence_len: int = 1,
        checkpoint_path: str = "pretrained_models/vjepa2_1_vitl.pt",
        model_name: str = "vit_large",
    ) -> None:
        super().__init__(
            vision_backbone_id,
            image_resize_strategy,
            default_image_size=default_image_size,
            image_sequence_len=image_sequence_len,
        )
        self.checkpoint_path = checkpoint_path
        self.vjepa_model_name = model_name
        self.dtype = torch.bfloat16

        # Initialize V-JEPA encoder with num_frames=8 (RoPE handles variable length)
        VJEPA21Encoder = _get_vjepa_encoder_cls()
        self.encoder = VJEPA21Encoder(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            img_size=default_image_size,
            num_frames=8,
            device="cpu",  # will be moved dynamically
        )
        # Freeze by default (configurable via freeze_backbones stage)
        for p in self.encoder.model.parameters():
            p.requires_grad = False
        self.encoder.model.eval()

        self._embed_dim = self.encoder.embed_dim  # 1024 for ViT-L

        # Image transform
        self.image_transform = VJEPAImageTransform(image_size=default_image_size)

    def _ensure_device(self, target_device: torch.device) -> None:
        model_device = next(self.encoder.model.parameters()).device
        if model_device != target_device:
            self.encoder.model = self.encoder.model.to(target_device)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode current-frame images.

        Args:
            pixel_values: [B, V, 3, H, W]
        Returns:
            [B, V*196, D_jepa]
        """
        self._ensure_device(pixel_values.device)
        B, V, C, H, W = pixel_values.shape

        # Duplicate frame to satisfy T>=2 requirement
        x = pixel_values.unsqueeze(2).repeat(1, 1, 2, 1, 1, 1)  # [B, V, 2, C, H, W]
        x = x.reshape(B * V, 2, C, H, W)

        with torch.no_grad():
            emb = self.encoder.extract(x, return_type="all_tokens")  # [B*V, 196, D]

        emb = emb.reshape(B, V, 196, -1)  # [B, V, 196, D]
        emb = emb.reshape(B, V * 196, -1)  # [B, V*196, D]
        return emb

    def encode_future(self, future_pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode future-frame images (for aux head supervision).

        Args:
            future_pixel_values: [B, V, 8, 3, H, W]
        Returns:
            [B, V, 4, 14, 14, D_jepa]
        """
        self._ensure_device(future_pixel_values.device)
        B, V, T, C, H, W = future_pixel_values.shape

        x = future_pixel_values.reshape(B * V, T, C, H, W)  # [B*V, 8, 3, H, W]

        with torch.no_grad():
            emb = self.encoder.extract(x, return_type="all_tokens")  # [B*V, 784, D]

        # 784 = 4 * 14 * 14  (T/2=4 temporal tokens, 14*14=196 spatial per timestep)
        emb = emb.reshape(B, V, 4, 14, 14, -1)
        return emb.detach()

    def get_image_transform(self) -> ImageTransform:
        return self.image_transform

    def get_fsdp_wrapping_policy(self) -> Callable:
        # V-JEPA encoder is frozen; simple policy wrapping the whole encoder
        return partial(_module_wrap_policy, module_classes={type(self.encoder.model)})

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return (3, self.default_image_size, self.default_image_size)

    @property
    def num_patches(self) -> int:
        # 196 patches per view (14x14)
        return 196 * self.image_sequence_len

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.dtype
