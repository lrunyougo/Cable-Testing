"""
YOLOv8 线缆缺陷检测训练脚本（使用COCO格式）
先转换 COCO 标注为 YOLO 格式，再进行训练
"""
from ultralytics import YOLO
import torch
import os
from pathlib import Path
import json
import shutil


def coco_to_yolo_bbox(coco_bbox, img_width, img_height):
    """
    将 COCO 格式的 bbox 转换为 YOLO 格式
    COCO: [x_min, y_min, width, height]
    YOLO: [x_center, y_center, width, height] (归一化到0-1)
    """
    # 将 bbox 值转换为 float 类型（COCO JSON中可能是字符串）
    x_min = float(coco_bbox[0])
    y_min = float(coco_bbox[1])
    width = float(coco_bbox[2])
    height = float(coco_bbox[3])

    # 将图像尺寸转换为 float
    img_width = float(img_width)
    img_height = float(img_height)

    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    return x_center, y_center, width_norm, height_norm


def convert_coco_to_yolo(json_file, image_dir, output_dir):
    """
    将 COCO 格式的标注转换为 YOLO 格式

    Args:
        json_file: COCO 标注文件路径
        image_dir: 图像文件目录
        output_dir: 输出目录
    """
    # 加载 COCO 标注
    print(f"正在加载 COCO 标注文件: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 提取类别信息
    class_names = [cat['name'] for cat in coco_data['categories']]
    print(f"检测到 {len(class_names)} 个类别:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")

    # 创建类别ID到索引的映射
    category_id_to_idx = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}

    # 创建图像ID到信息的映射
    image_id_to_info = {img['id']: img for img in coco_data['images']}

    # 创建图像ID到标注的映射
    image_id_to_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_annotations:
            image_id_to_annotations[img_id] = []
        image_id_to_annotations[img_id].append(ann)

    # 创建输出目录
    output_path = Path(output_dir)
    images_dir = output_path / 'images'
    labels_dir = output_path / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 转换每张图像的标注
    print(f"\n正在转换标注...")
    converted_count = 0

    for img_id, img_info in image_id_to_info.items():
        # 获取图像文件名（去掉 .jpg 后缀，用于命名标签文件）
        img_file = img_info['file_name']
        img_stem = Path(img_file).stem

        # 复制图像文件到输出目录
        src_image_path = Path(image_dir) / img_file
        dst_image_path = images_dir / img_file
        if src_image_path.exists():
            shutil.copy2(src_image_path, dst_image_path)
        else:
            print(f"  警告: 图像文件不存在: {src_image_path}")
            continue

        # 转换标注并保存
        annotations = image_id_to_annotations.get(img_id, [])
        label_file = labels_dir / f"{img_stem}.txt"

        with open(label_file, 'w', encoding='utf-8') as f:
            for ann in annotations:
                # 转换 bbox 格式
                x_center, y_center, width_norm, height_norm = coco_to_yolo_bbox(
                    ann['bbox'],
                    img_info['width'],
                    img_info['height']
                )

                # 获取类别索引
                category_id = ann['category_id']
                class_idx = category_id_to_idx[category_id]

                # 写入 YOLO 格式标签
                f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")

        converted_count += 1

    print(f"成功转换 {converted_count} 张图像的标注")

    # 保存类别名称文件
    classes_file = output_path / 'classes.txt'
    with open(classes_file, 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(f"{name}\n")

    # 生成 data.yaml
    yaml_file = output_path / 'data.yaml'
    yaml_content = f"""# YOLOv8 数据集配置文件
# 自动生成，请勿手动修改

# 数据集路径
path: {str(output_path.absolute())}  # 数据集根目录（绝对路径）
train: images     # 训练集图像路径（相对于path）
val: images       # 验证集图像路径（与train相同，YOLOv8会自动划分）

# 类别数量
nc: {len(class_names)}

# 类别名称
names:
"""

    for idx, name in enumerate(class_names):
        yaml_content += f"  {idx}: {name}\n"

    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"数据集配置文件已生成: {yaml_file}")

    return yaml_file, class_names


def main():
    # 获取项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))

    # COCO 标注文件路径
    coco_json = os.path.join(project_root, 'data', 'train', '_annotations.coco.json')
    image_dir = os.path.join(project_root, 'data', 'train')

    # 输出目录（转换后的 YOLO 格式数据）
    output_dir = os.path.join(project_root, 'data_yolo')

    # 检查文件是否存在
    if not os.path.exists(coco_json):
        raise FileNotFoundError(f"COCO标注文件不存在: {coco_json}")

    print("=" * 60)
    print("COCO -> YOLO 格式转换 + YOLOv8 训练")
    print("=" * 60)

    # 步骤1: 将 COCO 格式转换为 YOLO 格式
    print("\n步骤1: 转换 COCO 标注为 YOLO 格式")
    print("-" * 60)
    data_yaml, _ = convert_coco_to_yolo(coco_json, image_dir, output_dir)

    # 检查CUDA可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    if device == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    # 加载预训练模型
    print("\n步骤2: 加载 YOLOv8n 预训练模型")
    print("-" * 60)
    model = YOLO('yolov8n.pt')

    # 配置训练参数
    epochs = 100
    batch_size = 16
    img_size = 640

    print(f"训练参数:")
    print(f"  训练轮数: {epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  图像尺寸: {img_size}")

    # 步骤3: 开始训练
    print(f"\n步骤3: 开始训练")
    print("-" * 60)
    model.train(
        data=data_yaml,           # 使用生成的data.yaml
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project='output',
        name='train_yolo',
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"转换后的数据集目录: {output_dir}")
    print(f"最佳模型保存路径: output/train_yolo/weights/best.pt")
    print(f"最后一个epoch模型保存路径: output/train_yolo/weights/last.pt")


if __name__ == '__main__':
    main()
