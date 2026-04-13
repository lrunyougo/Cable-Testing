"""
YOLOv8 模型测试与评估脚本
"""
from ultralytics import YOLO
import cv2
import os

def test_model(model_path='weights/best.pt', test_images='data/test/images'):
    """
    测试训练好的模型

    Args:
        model_path: 模型权重文件路径
        test_images: 测试图像文件夹路径
    """
    # 加载训练好的模型
    print(f"正在加载模型: {model_path}")
    model = YOLO(model_path)

    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 获取测试图像列表
    image_files = [f for f in os.listdir(test_images) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if len(image_files) == 0:
        print(f"错误: 在 {test_images} 中未找到图像文件")
        return

    print(f"\n找到 {len(image_files)} 张测试图像")

    # 创建输出文件夹
    output_dir = 'output/test_results'
    os.makedirs(output_dir, exist_ok=True)

    # 逐张测试
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(test_images, image_file)

        print(f"\n[{i+1}/{len(image_files)}] 处理: {image_file}")

        # 推理
        results = model(image_path, conf=0.5, iou=0.7)

        # 显示结果
        for r in results:
            # 显示检测到的物体信息
            print(f"  检测到 {len(r.boxes)} 个目标")

            # 保存结果图像
            output_path = os.path.join(output_dir, f'result_{image_file}')
            r.save(output_path)
            print(f"  结果已保存: {output_path}")

    print(f"\n测试完成！结果保存在: {output_dir}")

def evaluate_model(model_path='weights/best.pt'):
    """
    在验证集上评估模型性能
    """
    print("正在评估模型性能...")

    # 加载模型
    model = YOLO(model_path)

    # 在验证集上评估
    metrics = model.val(data='data/data.yaml', split='val')

    # 打印评估结果
    print("\n" + "="*50)
    print("模型评估结果")
    print("="*50)
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50:95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print("="*50)

    return metrics

if __name__ == '__main__':
    import torch

    print("YOLOv8 模型测试")
    print("="*50)

    # 选择功能
    print("1. 在测试图像上进行推理")
    print("2. 在验证集上评估模型性能")
    choice = input("\n请选择功能 (1/2): ").strip()

    if choice == '1':
        model_path = input("请输入模型路径 (默认: weights/best.pt): ").strip()
        model_path = model_path if model_path else 'weights/best.pt'

        test_images = input("请输入测试图像文件夹路径 (默认: data/test/images): ").strip()
        test_images = test_images if test_images else 'data/test/images'

        test_model(model_path, test_images)

    elif choice == '2':
        model_path = input("请输入模型路径 (默认: weights/best.pt): ").strip()
        model_path = model_path if model_path else 'weights/best.pt'

        evaluate_model(model_path)

    else:
        print("无效选择")
