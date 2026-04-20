"""
V-JEPA 2.1 Standalone Embedding Extractor

High-level API for loading pretrained V-JEPA 2.1 weights and extracting embeddings
from image sequences, without depending on the original vjepa2 codebase.
"""

import warnings
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn

from .models import vit_base, vit_large, vit_giant_xformers, vit_gigantic_xformers, VIT_EMBED_DIMS


IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225])


def _clean_state_dict_key(state_dict):
    """Remove 'module.' and 'backbone.' prefixes from checkpoint keys."""
    new_state_dict = {}
    for key, val in state_dict.items():
        new_key = key.replace("module.", "").replace("backbone.", "")
        new_state_dict[new_key] = val
    return new_state_dict


def _infer_ckpt_key(checkpoint):
    """Auto-detect encoder state dict key in checkpoint."""
    for key in ("ema_encoder", "target_encoder", "encoder"):
        if key in checkpoint:
            return key
    # Fallback: if checkpoint is already a flat state dict for encoder
    if "predictor" in checkpoint:
        raise ValueError(
            "Cannot find encoder weights in checkpoint. "
            "Expected one of ['ema_encoder', 'target_encoder', 'encoder'], but got keys: "
            f"{list(checkpoint.keys())}"
        )
    warnings.warn(
        "Checkpoint does not contain 'ema_encoder'/'target_encoder'/'encoder' key. "
        "Assuming the checkpoint itself is the encoder state dict."
    )
    return None


class VJEPA21Encoder:
    """
    Standalone V-JEPA 2.1 encoder for embedding extraction.

    Supports loading checkpoints from:
      - V-JEPA 2.1 ViT-B/L/g/G pretrained models
      - Both .pt and .pth formats

    Example:
        >>> encoder = VJEPA21Encoder(
        ...     checkpoint_path="vjepa2_1_vitl_dist_vitG_384.pt",
        ...     model_name="vit_large",
        ...     img_size=224,
        ...     num_frames=8,
        ... )
        >>> images = torch.randn(8, 3, 224, 224)  # 8 frames
        >>> emb = encoder.extract(images, return_type="all_tokens")
        >>> print(emb.shape)  # [1, 784, 1024]
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "vit_large",
        img_size: int = 224,
        num_frames: int = 8,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            checkpoint_path: Path to the pretrained .pt checkpoint.
            model_name: One of {"vit_base", "vit_large", "vit_giant_xformers", "vit_gigantic_xformers"}.
            img_size: Spatial resolution (both H and W). Default 224.
            num_frames: Number of input frames. Must be >= 1 and even (tubelet_size=2).
                        Common choices: 2 (single-image-ish) or 8 (short clip).
            device: Device to load the model on.
        """
        self.model_name = model_name
        self.img_size = img_size
        self.num_frames = num_frames
        self.device = device

        if num_frames % 2 != 0:
            raise ValueError(f"num_frames must be even (tubelet_size=2), got {num_frames}")

        # Build model
        self.model = self._build_model(model_name, img_size, num_frames)
        self.model.to(device)
        self.model.eval()

        # Load checkpoint
        self._load_checkpoint(checkpoint_path)

        # Infer embed_dim from model
        self.embed_dim = self.model.embed_dim

    def _build_model(self, model_name, img_size, num_frames):
        """Construct the Vision Transformer encoder."""
        model_kwargs = dict(
            img_size=(img_size, img_size),
            patch_size=16,
            num_frames=num_frames,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )

        model_fn = {
            "vit_base": vit_base,
            "vit_large": vit_large,
            "vit_giant_xformers": vit_giant_xformers,
            "vit_gigantic_xformers": vit_gigantic_xformers,
        }.get(model_name)

        if model_fn is None:
            raise ValueError(
                f"Unknown model_name '{model_name}'. "
                f"Choose from: vit_base, vit_large, vit_giant_xformers, vit_gigantic_xformers"
            )

        model = model_fn(**model_kwargs)
        return model

    def _load_checkpoint(self, checkpoint_path):
        """Load pretrained weights into the encoder."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        ckpt_key = _infer_ckpt_key(checkpoint)
        if ckpt_key is not None:
            state_dict = checkpoint[ckpt_key]
        else:
            state_dict = checkpoint

        state_dict = _clean_state_dict_key(state_dict)
        msg = self.model.load_state_dict(state_dict, strict=False)

        if msg.missing_keys:
            warnings.warn(f"Missing keys when loading checkpoint: {msg.missing_keys}")
        if msg.unexpected_keys:
            warnings.warn(f"Unexpected keys when loading checkpoint: {msg.unexpected_keys}")

    def preprocess(self, images: Union[torch.Tensor, np.ndarray, List]) -> torch.Tensor:
        """
        Preprocess a sequence of images into the format expected by V-JEPA 2.1.

        Supported input formats:
          - torch.Tensor:
              - [T, 3, H, W]  → single sample
              - [B, T, 3, H, W]
              - [B, 3, T, H, W]
          - np.ndarray:
              - [T, H, W, 3]  → single sample
              - [B, T, H, W, 3]
          - List[PIL.Image.Image] of length T

        Returns:
            torch.Tensor of shape [B, 3, T, H, W], dtype float32, normalized with
            ImageNet mean/std.
        """
        # Handle PIL list
        if isinstance(images, list):
            try:
                from PIL import Image
                if isinstance(images[0], Image.Image):
                    tensors = []
                    for img in images:
                        img = img.convert("RGB")
                        arr = np.array(img).astype(np.float32) / 255.0  # [H, W, 3]
                        tensors.append(torch.from_numpy(arr))
                    images = torch.stack(tensors, dim=0)  # [T, H, W, 3]
            except ImportError:
                pass

        # Convert numpy to torch
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
            if images.max() > 1.0:
                images = images / 255.0

        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Unsupported input type: {type(images)}")

        # Ensure float32 and [0,1]
        images = images.float()
        if images.max() > 1.0:
            images = images / 255.0

        # Normalize dimensions to [B, 3, T, H, W]
        if images.dim() == 4:
            # [T, 3, H, W] or [T, H, W, 3]
            if images.shape[-1] == 3:
                # [T, H, W, 3] → [T, 3, H, W]
                images = images.permute(0, 3, 1, 2)
            # Now [T, 3, H, W] → [1, 3, T, H, W]
            images = images.unsqueeze(0).permute(0, 2, 1, 3, 4)
        elif images.dim() == 5:
            if images.shape[-1] == 3:
                # [B, T, H, W, 3] → [B, T, 3, H, W]
                images = images.permute(0, 1, 4, 2, 3)
            if images.shape[2] == 3:
                # [B, T, 3, H, W] → [B, 3, T, H, W]
                images = images.permute(0, 2, 1, 3, 4)
            # If already [B, 3, T, H, W], do nothing
        else:
            raise ValueError(
                f"Expected input with 4 or 5 dimensions, got {images.dim()} with shape {images.shape}"
            )

        # Validate shape
        B, C, T, H, W = images.shape
        if C != 3:
            raise ValueError(f"Expected 3 channels, got {C}")
        if H != self.img_size or W != self.img_size:
            warnings.warn(
                f"Input spatial size ({H}x{W}) differs from model img_size ({self.img_size}x{self.img_size}). "
                f"RoPE can handle this, but verify this is intentional."
            )
        if T != self.num_frames:
            warnings.warn(
                f"Input temporal length ({T}) differs from num_frames ({self.num_frames}). "
                f"The model will process the actual input length, but initialization was for {self.num_frames}."
            )

        # ImageNet normalization
        mean = IMAGENET_DEFAULT_MEAN.view(1, 3, 1, 1, 1).to(images.device)
        std = IMAGENET_DEFAULT_STD.view(1, 3, 1, 1, 1).to(images.device)
        images = (images - mean) / std

        return images

    @torch.no_grad()
    def extract(
        self,
        images: Union[torch.Tensor, np.ndarray, List],
        return_type: str = "all_tokens",
    ) -> torch.Tensor:
        """
        Extract embeddings from a sequence of images.

        Args:
            images: See `preprocess()` for supported formats.
            return_type: How to aggregate the output tokens.
                - "all_tokens":    Return all patch tokens. Shape [B, N, D].
                                   N = (T/2) * (H/16) * (W/16).
                - "spatial_mean":  Average over spatial dims per timestep.
                                   Shape [B, T/2, D].
                - "global_mean":   Global average over all tokens.
                                   Shape [B, D].

        Returns:
            torch.Tensor of the requested shape.
        """
        x = self.preprocess(images).to(self.device)

        with torch.cuda.amp.autocast(enabled=False):
            out = self.model(x)  # [B, N, D]

        B, N, D = out.shape
        T_tok = x.shape[2] // 2  # temporal tokens after tubelet
        S = N // T_tok           # spatial tokens per timestep

        if return_type == "all_tokens":
            return out
        elif return_type == "spatial_mean":
            out = out.view(B, T_tok, S, D)
            return out.mean(dim=2)  # [B, T_tok, D]
        elif return_type == "global_mean":
            return out.mean(dim=1)  # [B, D]
        else:
            raise ValueError(f"Unknown return_type: {return_type}")
