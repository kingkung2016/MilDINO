
import os

# 加载模型
from rfdetr import RFDETRLarge
os.environ["RFD_ENCODER"] = "dinov3_large"


if __name__ == "__main__":
    # 定义模型和推理参数
    train_config = {
        'pretrain_weights' : "/data/weight/rf-detr/rf-detr-large.pth",
        'encoder' : os.environ["RFD_ENCODER"],
        'pretrained_encoder' : "/data/weight/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    }
    model = RFDETRLarge(**train_config)
    model.train()
