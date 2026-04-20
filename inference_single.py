
import os
import numpy as np
from PIL import Image
import supervision as sv
import matplotlib.pyplot as plt

# -----------------------------
# 1. 模型加载（假设你已有 rfdetr）
# -----------------------------
from rfdetr import RFDETRLarge
from rfdetr.util.classes import JS_CLASSES    #修改类别
os.environ["RFD_ENCODER"] = "dinov3_large"

def load_model():
    inference_config = {
        'pretrain_weights': "./checkpoint_best.pth",    #修改权重
        'encoder': os.environ["RFD_ENCODER"],
        'eval': True
    }
    return RFDETRLarge(**inference_config)


# -----------------------------
# 2. NMS（纯 CPU）
# -----------------------------
def py_cpu_nms(dets, thresh):
    if len(dets) == 0:
        return []
    dets = np.array(dets, dtype=np.float32)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


# -----------------------------
# 3. 主函数：大图检测 + 合并 + 可视化
# -----------------------------
def detect_large_image_in_memory(
    image_path: str,
    output_dir: str,
    subsize: int = 512,
    gap: int = 50,
    model_threshold: float = 0.5,
    nms_thresh: float = 0.1
):
    # 1. 用 PIL 读图（自动处理 RGB）
    pil_img = Image.open(image_path).convert("RGB")  # 确保是 RGB 模式
    W, H = pil_img.size
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # 2. 加载模型
    model = load_model()

    slide = subsize - gap
    all_detections = []  # [x1, y1, x2, y2, conf, class_name]

    left = 0
    while left < W:
        if left + subsize > W:
            left = max(W - subsize, 0)
        up = 0
        while up < H:
            if up + subsize > H:
                up = max(H - subsize, 0)

            # 3. 切出子图（PIL crop 使用 (left, top, right, bottom)）
            sub_pil = pil_img.crop((left, up, left + subsize, up + subsize))

            # 4. 推理
            detections = model.predict(sub_pil, threshold=model_threshold)

            # 5. 映射回原图坐标
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i]
                conf = detections.confidence[i]
                class_id = detections.class_id[i]
                class_name = JS_CLASSES[class_id]        #修改类别映射

                # 映射：子图坐标 + 偏移
                x1_abs = x1 + left
                y1_abs = y1 + up
                x2_abs = x2 + left
                y2_abs = y2 + up

                all_detections.append([x1_abs, y1_abs, x2_abs, y2_abs, conf, class_id])

            # 滑动
            if up + subsize >= H:
                break
            else:
                up += slide
        if left + subsize >= W:
            break
        else:
            left += slide

    # 6. 按类别 NMS
    from collections import defaultdict
    class_dict = defaultdict(list)
    for det in all_detections:
        x1, y1, x2, y2, conf, cls_id = det
        class_dict[cls_id].append([x1, y1, x2, y2, conf])

    final_boxes = []
    final_labels = []
    final_class_ids = []

    for cls_id, dets in class_dict.items():
        keep_indices = py_cpu_nms(dets, nms_thresh)
        cls_name = JS_CLASSES[cls_id]        # 修改类别映射
        for idx in keep_indices:
            x1, y1, x2, y2, conf = dets[idx]
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            final_boxes.append([x1, y1, x2, y2])
            final_labels.append(f"{cls_name} {conf:.2f}")
            final_class_ids.append(cls_id)

    # 7. 可视化（PIL → numpy array for supervision）
    img_np = np.array(pil_img)  # shape: (H, W, 3), RGB

    if final_boxes:
        detections_sv = sv.Detections(
            xyxy=np.array(final_boxes, dtype=np.float32),
            confidence=np.array([float(lbl.split()[-1]) for lbl in final_labels]),
            class_id=np.array(final_class_ids, dtype=int)
        )
        box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.CLASS)
        label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.CLASS)
        annotated_image = box_annotator.annotate(img_np.copy(), detections_sv)
        annotated_image = label_annotator.annotate(annotated_image, detections_sv, final_labels)
    else:
        annotated_image = img_np
        print("⚠️ No detections after merging.")

    # 8. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"vis_{basename}.png")
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ Visualization saved to: {output_path}")


# -----------------------------
# 4. 运行入口
# -----------------------------
if __name__ == "__main__":
    detect_large_image_in_memory(
        image_path="./ship_test_2.jpg",
        output_dir="./inference_results/",
        subsize=512,
        gap=100,
        model_threshold=0.5,
        nms_thresh=0.1
    )












