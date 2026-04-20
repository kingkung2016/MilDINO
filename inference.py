import os
import time
import traceback
from tqdm import tqdm
import supervision as sv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from rfdetr import RFDETRSmall, RFDETRBase, RFDETRMedium, RFDETRLarge
from rfdetr.util.classes import SAR_CLASSES
Inference_classes = SAR_CLASSES

def detect_folder():
    try:
        # 初始化配置
        start_time = time.time()
        inference_config = {
            'pretrain_weights': "./output/checkpoint_best_total.pth",
            'encoder': "dinov3_large",
            'eval': True
        }

        # 创建输出目录
        output_dir = "./inference_results/inference_images"
        results_dir = "./inference_results/inference_text"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # 初始化模型
        model = RFDETRLarge(**inference_config)

        # 加载测试图片
        input_dir = "/data/MilDINO/inference_images/"    #测试图片路径
        image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_paths:
            raise ValueError("未找到任何测试图片")


        # 带进度条的推理
        total_images = len(image_paths)
        print(f"开始检测，共发现 {total_images} 张图片")

        with tqdm(total=total_images, desc="处理进度") as pbar:
            for image_path in image_paths:
                try:
                    # 读取图片
                    with Image.open(image_path) as img:
                        # 模型推理
                        detections = model.predict([img], threshold=0.5)

                        # 生成结果文件名
                        base_name = os.path.basename(image_path)
                        result_txt = os.path.join(results_dir, f"result_{base_name}.txt")

                        # 写入检测结果
                        with open(result_txt, 'w') as f:
                            # 写入基础信息
                            f.write(f"Image: {image_path}\n\n")

                            # 遍历每个检测结果
                            for i, (xyxy, class_id, confidence) in enumerate(zip(
                                    detections.xyxy,
                                    detections.class_id,
                                    detections.confidence
                            )):
                                # 修正坐标格式为四元组 (x1, y1, x2, y2)
                                coordinates = [round(float(x), 2) for x in xyxy]

                                # 写入检测项
                                f.write(
                                    f"目标 {i + 1}:\n"
                                    f"类别: {Inference_classes[class_id]}\n"    
                                    f"置信度: {confidence:.4f}\n"
                                    f"坐标: {coordinates}\n\n"
                                )

                        # 可视化标注
                        labels = [
                            f"{Inference_classes[class_id]} {confidence:.2f}"
                            for class_id, confidence
                            in zip(detections.class_id, detections.confidence)
                        ]

                        annotated_image = sv.BoxAnnotator().annotate(np.array(img.copy()), detections)
                        annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

                        # 保存结果图
                        output_path = os.path.join(output_dir, f"output_{base_name}")
                        plt.imsave(output_path, annotated_image)

                except Exception as e:
                    print(f"\n处理图片 {image_path} 时出错: {str(e)}")
                    traceback.print_exc()
                finally:
                    pbar.update(1)

        # 统计时间
        total_time = time.time() - start_time
        print(f"\n检测完成！共处理 {total_images} 张图片")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均每张耗时: {total_time / total_images:.2f} 秒")

    except Exception as e:
        print(f"发生严重错误: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    detect_folder()