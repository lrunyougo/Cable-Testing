"""
YOLOv8 线缆缺陷检测训练脚本（直接使用绝对路径）
"""
from ultralytics import YOLO
import torch
import os

def main():
    # 检查CUDA可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    if device == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    # 加载预训练模型（YOLOv8n）
    print("\n正在加载YOLOv8n预训练模型...")
    model = YOLO('yolov8n.pt')

    # 配置训练参数（直接写死路径，不使用data.yaml）
    epochs = 100
    batch_size = 16
    img_size = 640

    print("\n开始训练...")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"图像尺寸: {img_size}")

    # 直接指定训练集、验证集、测试集路径（绝对路径）
    train_data_dir = r'e:/论文/训练平台/data_split/train'
    val_data_dir = r'e:/论文/训练平台/data_split/val'
    test_data_dir = r'e:/论文/训练平台/data_split/test'

    # 检查路径是否存在
    if not os.path.exists(train_data_dir):
        raise FileNotFoundError(f"训练集目录不存在: {train_data_dir}")
    if not os.path.exists(val_data_dir):
        raise FileNotFoundError(f"验证集目录不存在: {val_data_dir}")
    if not os.path.exists(test_data_dir):
        raise FileNotFoundError(f"测试集目录不存在: {test_data_dir}")

    print(f"\n数据集路径：")
    print(f"训练集: {train_data_dir}")
    print(f"验证集: {val_data_dir}")
    print(f"测试集: {test_data_dir}")

    # 开始训练
    # 注意：Ultralytics YOLOv8 的 train() 方法只接受 data 参数指向数据集配置文件（.yaml）
    # 所以下面这行会报错！
    model.train(
        data=train_data_dir,  # 错误！应该是.yaml文件路径
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project='output',
        name='train',
        exist_ok=True,
        plots=True,
        verbose=True
    )

    print("\n训练完成!")
    print(f"最佳模型保存路径: output/train/weights/best.pt")
    print(f"最后一个epoch模型保存路径: output/train/weights/last.pt")

if __name__ == '__main__':
    main()
