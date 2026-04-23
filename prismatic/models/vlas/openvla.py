"""
openvla.py

PyTorch Module defining OpenVLA as a lightweight wrapper around a PrismaticVLM; defines custom logic around
discretizing actions with the ActionTokenizer.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL.Image import Image as Img
from transformers import LlamaTokenizerFast
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class OpenVLA(PrismaticVLM):
    def __init__(
        self,
        *args,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
        action_tokenizer: ActionTokenizer,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.norm_stats = norm_stats
        self.action_tokenizer = action_tokenizer

    @torch.inference_mode()
    def predict_action(
        self,
        image: Union[Img, List[Img]],
        instruction: str,
        unnorm_key: Optional[str] = None,
        proprio: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action (de-tokenizes).

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vision_backbone.get_image_transform(), self.llm_backbone.tokenizer

        # Build VLA Prompt
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        # prompt_builder.add_turn(role="gpt", message="")
        prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs
        tokenized = tokenizer(prompt_text, truncation=True, return_tensors="pt")
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # If the special empty token ('') does not already appear after the colon (':') token in the prompt
            # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
        elif isinstance(tokenizer, Qwen2TokenizerFast):
            # do nothing here. I think...
            pass
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image(s)
        if isinstance(image, list):
            img_tensors = [image_transform(img) for img in image]
            if not all(isinstance(t, torch.Tensor) for t in img_tensors):
                raise ValueError("List image transform must return Tensor for each image.")
            pixel_values = torch.stack(img_tensors, dim=0)[None, ...].to(self.device)  # [1, V, 3, H, W]
        else:
            pixel_values = image_transform(image)
            if isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values[None, ...].to(self.device)
            elif isinstance(pixel_values, dict):
                pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
            else:
                raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        use_continuous_head = (
            self.action_head is not None
            and proprio is not None
            and hasattr(self.action_head, "sample_action")
        )
        if use_continuous_head:
            action_placeholder_tokens = getattr(self, "action_placeholder_tokens", 0)
            if action_placeholder_tokens > 0:
                placeholder_ids = torch.full(
                    (input_ids.shape[0], action_placeholder_tokens),
                    fill_value=ACTION_TOKEN_BEGIN_IDX,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                input_ids = torch.cat([input_ids, placeholder_ids], dim=1)
                placeholder_mask = torch.ones(
                    (attention_mask.shape[0], action_placeholder_tokens),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, placeholder_mask], dim=1)

            if isinstance(proprio, torch.Tensor):
                proprio_t = proprio.to(self.device, dtype=torch.float32)
            else:
                proprio_t = torch.tensor(np.asarray(proprio), device=self.device, dtype=torch.float32)
            if proprio_t.dim() == 1:
                proprio_t = proprio_t.unsqueeze(0)

            if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
                key = self._check_unnorm_key(self.norm_stats, unnorm_key)
                proprio_norm_stats = self.norm_stats[key]["proprio"]
                proprio_high = torch.tensor(np.array(proprio_norm_stats["q99"]), device=self.device, dtype=torch.float32)
                proprio_low = torch.tensor(np.array(proprio_norm_stats["q01"]), device=self.device, dtype=torch.float32)
                mask = torch.tensor(
                    np.array(proprio_norm_stats.get("mask", np.ones_like(proprio_norm_stats["q01"], dtype=bool))),
                    device=self.device,
                    dtype=torch.bool,
                )
                normalized_proprio = torch.where(
                    mask,
                    2 * (proprio_t - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
                    proprio_t,
                )
            else:
                normalized_proprio = proprio_t

            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()
            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=None,
                output_hidden_states=True,
                return_dict=True,
            )
            llm_hidden = outputs["llm_hidden"]
            hidden_states = outputs.get("llm_hidden_states", None)
            if hidden_states is None:
                # Fallback to only final hidden state repeated once.
                hidden_states = (llm_hidden,)
            task_token_count = outputs.get("task_token_count", self.vision_backbone.num_patches)
            action_token_count = outputs.get("action_token_count", getattr(self, "action_placeholder_tokens", 0))
            z_action = llm_hidden[:, -1, :]

            if hasattr(self.action_head, "predict_action"):
                normalized_actions = self.action_head.predict_action(
                    z_action,
                    normalized_proprio,
                    hidden_states=hidden_states,
                    task_token_count=task_token_count,
                    action_token_count=action_token_count,
                    phase="Inference",
                )
            else:
                normalized_actions = self.action_head.sample_action(
                    z_action,
                    normalized_proprio,
                )
            normalized_actions = normalized_actions.detach().float().cpu().numpy()[0]

            action_norm_stats = self.get_action_stats(unnorm_key)
            if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
                mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
                action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
            elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
                mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
                action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
            else:
                raise ValueError("Unsupported action/proprio normalization type detected!")
            actions = np.where(
                mask,
                0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
                normalized_actions,
            )
            return actions

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super(PrismaticVLM, self).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, (opt T,) 3, res, res] or Dict[str, ...]
                max_new_tokens=self.get_action_dim(unnorm_key),
                **kwargs
            )
            # fmt: on

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, please pass a `unnorm_key` from the following "
                f"options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        # Error Handling
        assert (
            unnorm_key in norm_stats
        ), f"The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}"

        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return self.norm_stats[unnorm_key]["action"]
