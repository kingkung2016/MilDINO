

# 加载模型
from rfdetr import RFDETRLarge

if __name__ == "__main__":
    # 定义模型和推理参数
    inference_config = {
        'pretrain_weights' : "./output/checkpoint_best_total.pth",
        'encoder' : "dinov3_large",
        'eval': True
    }
    model = RFDETRLarge(**inference_config)
    model.train()
