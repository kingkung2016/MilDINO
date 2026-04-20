# ------------------------------------------------------------------------
# Note: This file is an original wrapper that *loads* DINOv3 models.
# Using DINOv3 code/weights is subject to that license’s restrictions.
# DINOv3 itself is licensed under the DINOv3 License; see https://github.com/facebookresearch/dinov3/tree/main?tab=License-1-ov-file
# ------------------------------------------------------------------------

from typing import Sequence, Optional
import os
import torch
import torch.nn as nn

from rfdetr.models.backbone.dinov3_configs.Dinov3_backbones import dinov3_vits16, dinov3_vitb16, dinov3_vitl16, dinov3_vith16plus, dinov3_vit7b16


# channels per size (for projector wiring)
_SIZE2WIDTH = {"small": 384, "base": 768, "large": 1024, "huge": 1280, "7b": 4096}

# Map model names to their corresponding sizes
_MODEL_MAP = {
    "small": dinov3_vits16,
    "base": dinov3_vitb16,
    "large": dinov3_vitl16,
    "huge": dinov3_vith16plus,
    "7b": dinov3_vit7b16
}

class DinoV3(nn.Module):
    """
    RF-DETR-facing DINOv3 wrapper:
      - forward(x) -> List[B, C, H/16, W/16] (one per selected layer)
      - _out_feature_channels: List[int]
      - export(): no-op (kept for parity)
    """

    def __init__(
        self,
        size: str = "large",
        out_feature_indexes: Sequence[int] = None,
        hidden_dim: int = None,
        shape: Sequence[int] = (640, 640),
        patch_size: int = 16,
        weights: Optional[str] = None,      # path or URL to *.pth (hub)
        **__,
        ):
        """
        A DINOv3 wrapper for RF-DETR.

        Args:
            shape (Sequence[int]): Input image shape (H, W).
            out_feature_indexes (Sequence[int]): Layer indexes to return.
            size (str): DINOv3 model size: "small", "base", or "large".
            patch_size (int): Patch size for the model.
            load_dinov3_weights (bool): If True, load DINOv3 weights from HF or hub.
            hf_token (Optional[str]): Hugging Face token for private models.
            repo_dir (Optional[str]): Path to the local DINOv3 repository.
            weights (Optional[str]): Path to the DINOv3 weights file.
            pretrained_name (Optional[str]): Pretrained model name for HF.
        """

        super().__init__()

        self.shape = tuple(shape)
        self.patch_size = int(patch_size)
        self.num_register_tokens = 0
        self.embed_dim = _SIZE2WIDTH[size]
        self.hidden_dim = hidden_dim
        self._out_feature_channels = [self.embed_dim] * len(out_feature_indexes)
        self.out_feature_indexes = list(out_feature_indexes)

        self.is_pretrained = weights is not None
        self.pretrained_weights = weights

        if size and size in _MODEL_MAP:
            model_func = _MODEL_MAP[size]
        else:
            model_func = dinov3_vitb16

        self.encoder = model_func(
            pretrained=self.is_pretrained,
            weights=self.pretrained_weights,
            check_hash=False
        )

        # best-effort introspection (these attrs may or may not exist on hub module)
        self.num_register_tokens = int(getattr(self.encoder, "num_register_tokens", 0))
        self.model_patch = int(getattr(self.encoder, "patch_size", self.patch_size))

    def _tokens_to_map(self, hidden_state: torch.Tensor, B: int, H: int, W: int) -> torch.Tensor:
        """
        Accepts either:
        - [B, 1+R+HW, C]  (CLS + register + patch tokens)
        - [B, HW, C]      (no special tokens)
        Returns:
        - [B, C, H/ps, W/ps]
        """
        ps = self.model_patch
        assert H % ps == 0 and W % ps == 0, f"Input must be divisible by patch size {ps}, got {(H, W)}"
        hp, wp = H // ps, W // ps
        C = hidden_state.shape[-1]

        if hidden_state.dim() == 2:
            # e.g., [HW, C] (no batch) -> try to recover batch
            seq = hidden_state.shape[0]
            assert seq % B == 0, f"Cannot infer batch from tokens of shape {hidden_state.shape} with B={B}"
            hidden_state = hidden_state.view(B, seq // B, C)

        assert hidden_state.dim() == 3, f"Expected tokens [B, S, C], got {tuple(hidden_state.shape)}"
        S = hidden_state.shape[1]
        expected_hw = hp * wp

        if S == expected_hw:
            seq = hidden_state  # already patch tokens
        elif S >= expected_hw + 1 + self.num_register_tokens:
            # drop CLS + registers, then take the last expected_hw tokens
            seq = hidden_state[:, 1 + self.num_register_tokens :, :]
            seq = seq[:, -expected_hw:, :]
        else:
            # unknown extra tokens count; take the last expected_hw tokens
            seq = hidden_state[:, -expected_hw:, :]

        return seq.view(B, hp, wp, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor):
        #print('dinov3.input.shape=', x.shape)

        B, _, H, W = x.shape

        # --- Hub path: try get_intermediate_layers -----------
        if hasattr(self.encoder, "get_intermediate_layers"):
            max_idx = max(self.out_feature_indexes)
            hs_list = self.encoder.get_intermediate_layers(x, n=max_idx + 1, reshape=False)
            proc = []
            for h in hs_list:
                # some impls return (tokens, cls) tuples
                if isinstance(h, (list, tuple)):
                    h = h[0]
                proc.append(h)
            feats = [self._tokens_to_map(proc[i], B, H, W) for i in self.out_feature_indexes]
            return feats



