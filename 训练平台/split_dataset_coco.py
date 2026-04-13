"""
COCO格式数据集划分工具
从 _annotations.coco.json 读取标注，并按比例划分为 train/val/test
"""
import json
import os
import shutil
import random
from pathlib import Path


def load_coco_annotations(json_file):
    """
    加载 COCO 格式的标注文件
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 创建映射
    image_id_to_info = {img['id']: img for img in data['images']}
    image_id_to_annotations = {}

    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_annotations:
            image_id_to_annotations[img_id] = []
        image_id_to_annotations[img_id].append(ann)

    # 创建类别映射
    category_id_to_idx = {cat['id']: idx for idx, cat in enumerate(data['categories'])}
    class_names = [cat['name'] for cat in data['categories']]

    return image_id_to_info, image_id_to_annotations, category_id_to_idx, class_names


def coco_to_yolo_label(coco_bbox, img_width, img_height):
    """
    将 COCO 格式的 bbox 转换为 YOLO 格式
    COCO: [x_min, y_min, width, height]
    YOLO: [x_center, y_center, width, height] (归一化到0-1)
    """
    x_min, y_min, width, height = coco_bbox

    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    return x_center, y_center, width_norm, height_norm


def save_yolo_label(annotation, img_info, label_dir, category_id_to_idx):
    """
    保存 YOLO 格式的标签文件
    """
    label_file = Path(label_dir) / f"{img_info['id']}.txt"

    with open(label_file, 'w', encoding='utf-8') as f:
        for ann in annotation:
            x_center, y_center, width_norm, height_norm = coco_to_yolo_label(
                ann['bbox'],
                img_info['width'],
                img_info['height']
            )

            category_id = ann['category_id']
            class_idx = category_id_to_idx[category_id]

            f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")


def split_coco_dataset(json_file, image_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=None):
    """
    划分 COCO 格式的数据集

    Args:
        json_file: COCO 标注文件路径
        image_dir: 图像文件目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
    """
    # 设置随机种子
    if random_seed is not None:
        random.seed(random_seed)

    json_path = Path(json_file)
    image_path = Path(image_dir)
    output_path = Path(output_dir)

    # 加载 COCO 标注
    print("正在加载 COCO 标注文件...")
    image_id_to_info, image_id_to_annotations, category_id_to_idx, class_names = load_coco_annotations(json_file)

    # 获取所有图像 ID
    image_ids = list(image_id_to_info.keys())

    # 随机打乱
    random.shuffle(image_ids)

    # 计算划分数量
    total = len(image_ids)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    print(f"\n总共 {total} 张图像")
    print(f"检测到 {len(class_names)} 个类别:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")

    print(f"\n数据集划分:")
    print(f"  训练集: {train_count} 张 ({train_ratio*100:.1f}%)")
    print(f"  验证集: {val_count} 张 ({val_ratio*100:.1f}%)")
    print(f"  测试集: {test_count} 张 ({test_ratio*100:.1f}%)")

    # 划分数据集
    train_ids = image_ids[:train_count]
    val_ids = image_ids[train_count:train_count + val_count]
    test_ids = image_ids[train_count + val_count:]

    # 创建输出目录
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 处理数据集
    print("\n正在处理数据集...")
    process_split(train_ids, image_id_to_info, image_id_to_annotations, image_path, output_path / 'train', category_id_to_idx, '训练集')
    process_split(val_ids, image_id_to_info, image_id_to_annotations, image_path, output_path / 'val', category_id_to_idx, '验证集')
    process_split(test_ids, image_id_to_info, image_id_to_annotations, image_path, output_path / 'test', category_id_to_idx, '测试集')

    # 保存类别信息
    classes_file = output_path / 'classes.txt'
    with open(classes_file, 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"\n类别信息已保存到: {classes_file}")

    # 生成 data.yaml
    generate_data_yaml(output_path, class_names)

    print(f"\n输出目录结构:")
    print(f"{output_path}/")
    print(f"  ├── train/")
    print(f"  │   ├── images/  ({train_count} 张)")
    print(f"  │   └── labels/")
    print(f"  ├── val/")
    print(f"  │   ├── images/  ({val_count} 张)")
    print(f"  │   └── labels/")
    print(f"  ├── test/")
    print(f"  │   ├── images/  ({test_count} 张)")
    print(f"  │   └── labels/")
    print(f"  ├── classes.txt")
    print(f"  └── data.yaml")


def process_split(image_ids, image_id_to_info, image_id_to_annotations, image_dir, output_dir, category_id_to_idx, split_name):
    """
    处理一个数据集划分
    """
    output_images_dir = output_dir / 'images'
    output_labels_dir = output_dir / 'labels'

    count = 0
    for img_id in image_ids:
        img_info = image_id_to_info[img_id]
        annotations = image_id_to_annotations.get(img_id, [])

        if not annotations:
            # 如果图像没有标注，创建一个空的标签文件
            label_file = output_labels_dir / f"{img_info['id']}.txt"
            label_file.touch()

        # 复制图像
        src_image = image_dir / img_info['file_name']
        if src_image.exists():
            shutil.copy2(src_image, output_images_dir / img_info['file_name'])
            count += 1
        else:
            print(f"  警告: 图像文件不存在: {src_image}")

        # 保存标签
        if annotations:
            save_yolo_label(annotations, img_info, output_labels_dir, category_id_to_idx)

    print(f"  {split_name}处理完成 ({count} 张)")


def generate_data_yaml(output_dir, class_names):
    """
    生成 data.yaml 配置文件
    """
    yaml_file = output_dir / 'data.yaml'

    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write("# YOLOv8 数据集配置文件\n\n")
        f.write("# 数据集路径\n")
        f.write("path: .  # 数据集根目录（相对于本yaml文件的路径）\n")
        f.write("train: train/images  # 训练集图像路径\n")
        f.write("val: val/images      # 验证集图像路径\n")
        f.write("test: test/images    # 测试集图像路径\n\n")
        f.write("# 类别数量\n")
        f.write(f"nc: {len(class_names)}\n\n")
        f.write("# 类别名称\n")
        f.write("names:\n")
        for idx, name in enumerate(class_names):
            f.write(f"  {idx}: {name}\n")

    print(f"数据集配置文件已生成: {yaml_file}")


if __name__ == '__main__':
    print("=" * 60)
    print("COCO 格式数据集划分工具")
    print("=" * 60)

    # 配置
    json_file = 'data/train/_annotations.coco.json'
    image_dir = 'data/train'
    output_dir = 'data_split'
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    random_seed = 42

    print(f"\nCOCO 标注文件: {json_file}")
    print(f"图像目录: {image_dir}")
    print(f"输出目录: {output_dir}")
    print(f"\n划分比例:")
    print(f"  训练集: {train_ratio * 100:.1f}%")
    print(f"  验证集: {val_ratio * 100:.1f}%")
    print(f"  测试集: {test_ratio * 100:.1f}%")
    print(f"\n随机种子: {random_seed}")

    # 确认
    confirm = input("\n确认开始划分？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        exit(0)

    try:
        split_coco_dataset(
            json_file=json_file,
            image_dir=image_dir,
            output_dir=output_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed
        )

        print("\n" + "=" * 60)
        print("数据集划分完成!")
        print("=" * 60)
        print("\n下一步:")
        print("  1. 检查 data_split/ 目录下的文件结构")
        print("  2. 运行 python train.py 开始训练")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
