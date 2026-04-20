# MilDINO: An Object Detection Framework based on DINOv3


## 🛠️ 环境准备

### 1. 安装依赖

请根据您的 CUDA 版本安装对应的 PyTorch，然后安装其他依赖：

```
pip install -r requirements.txt
```

### 2. 数据集准备

请确保您的数据集遵循 **COCO 格式**。目录结构应如下所示：

```
your_dataset_root/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── ...
│   └── _annotations.coco.json
├── valid/
│   ├── images/
│   │   ├── img2.jpg
│   │   └── ...
│   └── _annotations.coco.json
└── test/
    ├── images/
    │   ├── img3.jpg
    │   └── ...
    └── _annotations.coco.json
```

**注意：** 标注文件的命名必须包含下划线前缀，即 `_annotations.coco.json`。

---

## 🚀 模型训练

配置好数据路径后，您可以开始训练模型。

### 关键参数说明

- `batch_size`: 单张 GPU 的 Batch Size。
- `grad_accum_steps`: 梯度累积步数。
    - **推荐 Batch Size** = `batch_size` × `grad_accum_steps` × GPU 数量。
- `epochs`: 训练轮数。
    - 小数据集建议设为 **50**。
    - 中等数据集建议设为 **30**。
    - 大数据集建议设为 **10**。
- `ema_decay`: EMA（指数移动平均）衰减系数。推荐设置为 **0.9999** 以获得更高的稳定性。

---

## 🔍 模型推理与测试

MilDINO 支持两种主要的推理模式：针对测试集的批量推理和针对超大分辨率图片的单图推理。

### 1. 批量推理

用于处理整个测试集文件夹，并生成带有检测框的图片结果。

**核心配置项：**

- `pretrain_weights`: 训练好的模型权重路径。
- `encoder`: 使用的编码器类型（如 `dinov3_vitb`）。
- `input_dir`: 待检测图片所在的目录。
- `output_dir`: 结果保存目录。

### 2. 单图推理

专门用于处理 SAR、遥感等超大分辨率图片。采用切片推理的方式。

**核心配置项：**

- `subsize`: 切片大小。根据目标物体的大小调整（例如 512, 640, 1024 等）。
- `gap`: 切片之间的重叠区域。建议设置为 **100-200**，以防止边缘目标被漏检。
- `model_threshold`: 置信度阈值。
- `nms_thresh`: NMS（非极大值抑制）阈值。

---

## 💡 常见问题提示

- **大图检测优化**：在进行单图推理时，如果发现小目标漏检，可以尝试减小 `subsize` 或增大 `gap`。
- **显存优化**：如果遇到 OOM (Out Of Memory)，请尝试减小 `batch_size` 或增加 `grad_accum_steps` 以保持等效的训练批次大小。
- **模型稳定性**：推荐使用 EMA（指数移动平均）来提高模型训练的稳定性。
