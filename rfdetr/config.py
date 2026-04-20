# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from pydantic import BaseModel, field_validator, model_validator, Field
from pydantic_core.core_schema import ValidationInfo  # for field_validator(info)
from typing import List, Optional, Literal
import os, torch

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


EncoderName = Literal[
    "dinov3_small",
    "dinov3_base",
    "dinov3_large",
    "dinov3_huge",
    "dinov3_7b"
]

def _encoder_default():
    """Default encoder name for the model config."""
    # default to v2 unless explicitly overridden by env
    val = os.getenv("RFD_ENCODER", "").strip() or "dinov3_base"

    # guardrail: only accept known names
    allowed = {"dinov3_small", "dinov3_base", "dinov3_large", "dinov3_huge", "dinov3_7b"}
    return val if val in allowed else "dinov3_base"


class ModelConfig(BaseModel):
    """Base configuration for RF-DETR models."""
    # WAS: only dinov2_windowed_*; NOW: include dinov3_* as drop-in options
    encoder: EncoderName = _encoder_default()

    out_feature_indexes: List[int]
    dec_layers: int
    two_stage: bool = True
    projector_scale: List[Literal["P3", "P4", "P5"]]
    hidden_dim: int
    patch_size: int
    num_windows: int
    sa_nheads: int
    ca_nheads: int
    dec_n_points: int
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    amp: bool = True
    num_classes: int = 6    #类别数需要修改
    pretrain_weights: Optional[str] = None
    pretrained_encoder: Optional[str] = None
    eval: bool = False           #只进行测试集计算
    device: Literal["cpu", "cuda", "mps"] = DEVICE
    resolution: int
    group_detr: int = 13
    gradient_checkpointing: bool = False
    positional_encoding_size: int
    # used only when encoder startswith("dinov3")
    dinov3_weights_path: Optional[str] = None  # e.g., r"C:\models\dinov3-vitb16.pth"


    # force /16 for v3
    @field_validator("patch_size", mode="after")
    def _coerce_patch_for_dinov3(cls, v, info: ValidationInfo):
        """Ensure patch size is 16 for DINOv3 encoders."""
        enc = str(info.data.get("encoder", ""))
        return 16 if enc.startswith("dinov3") else v

    # keep pos-encoding grid consistent with resolution / patch
    @field_validator("positional_encoding_size", mode="after")
    def _sync_pos_enc_with_resolution(cls, v, info: ValidationInfo):
        """Sync positional encoding size with resolution and patch size."""
        values = info.data or {}
        res, ps = values.get("resolution"), values.get("patch_size")
        return max(1, res // ps) if (res and ps) else v



class RFDETRBaseConfig(ModelConfig):
    """
    The configuration for an RF-DETR Base model.
    """
    # Allow choosing dinov3_* without changing call sites
    encoder: EncoderName = _encoder_default()
    hidden_dim: int = 256
    patch_size: int = 16  # will auto-become 16 if encoder startswith("dinov3")
    num_windows: int = 1  # ignored by DINOv3 branch
    dec_layers: int = 3     # 解码器层的数量，通常为6
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P4"]
    out_feature_indexes: List[int] = [2, 4, 5, 9]   #根据主干网络层数修改
    pretrain_weights: Optional[str] = "rf-detr-base.pth"
    resolution: int = 512  # 512//16=32 → pos grid auto=32 for both v2/v3
    positional_encoding_size: int = 36  # will auto-sync to resolution//patch_size


class RFDETRLargeConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Large model.
    """
    out_feature_indexes: List[int] = [11, 17, 23]   #提取层数
    hidden_dim: int = 384
    sa_nheads: int = 12
    ca_nheads: int = 24
    dec_n_points: int = 4
    patch_size: int = 16
    resolution: int = 640       #尝试修改
    projector_scale: List[Literal["P2", "P3", "P4", "P5"]] = ["P3", "P4", "P5"]
    pretrain_weights: Optional[str] = "rf-detr-large.pth"


class RFDETRNanoConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Nano model.
    """
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    dec_layers: int = 2
    patch_size: int = 16
    resolution: int = 384  # 384//16=24 → pos grid auto=24 for both v2/v3
    positional_encoding_size: int = 24
    pretrain_weights: Optional[str] = "rf-detr-nano.pth"


class RFDETRSmallConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Small model.
    """
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    dec_layers: int = 3
    patch_size: int = 16
    resolution: int = 512  # 512//16=32 → pos grid auto=32
    positional_encoding_size: int = 32
    pretrain_weights: Optional[str] = "rf-detr-small.pth"


class RFDETRMediumConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Medium model.
    """
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    dec_layers: int = 4
    patch_size: int = 16
    # resolution: int = 504          # 576//16=36 → pos grid auto=36
    resolution: int = 512
    positional_encoding_size: int = 36
    pretrain_weights: Optional[str] = "rf-detr-medium.pth"


class TrainConfig(BaseModel):
    lr: float = 1e-4                #学习率
    lr_encoder: float = 5e-5      #编码器学习率
    batch_size: int = 1            # batch_size * grad_accum_steps =16
    grad_accum_steps: int = 1
    epochs: int = 1            # 20/30/50
    ema_decay: float = 0.99    #optional: [0.999 0.9995 0.9999]
    ema_tau: int = 0
    lr_drop: int = 100
    checkpoint_interval: int = 10       #每隔一定epoch保存权重
    warmup_epochs: int = 0      # 热身机制，分布式训练开启
    lr_vit_layer_decay: float = 0.8
    lr_component_decay: float = 0.7
    drop_path: float = 0.0
    group_detr: int = 13
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    num_select: int = 300
    dataset_file: Literal["coco", "o365", "roboflow"] = "roboflow"
    square_resize_div_64: bool = True
    dataset_dir: str = "/data/SAR_datasets/SSDD-COCO/"         #数据集路径
    output_dir: str = "./output/"    #结果保存路径
    multi_scale: bool = True
    expanded_scales: bool = True
    do_random_resize_via_padding: bool = False
    use_ema: bool = True
    num_workers: int = 2
    weight_decay: float = 1e-4
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_use_ema: bool = False
    tensorboard: bool = False        #不生成这些日志文件
    wandb: bool = False
    project: Optional[str] = None
    run: Optional[str] = None
    class_names: List[str] = None
    run_test: bool = False           #跑测试集
