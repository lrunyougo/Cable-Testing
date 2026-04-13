"""
YOLOv8 线缆缺陷检测训练脚本
"""
from ultralytics import YOLO
import torch
import os

def main():
    # 获取项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 使用相对路径配置 dataset
    data_yaml = os.path.join(project_root, 'data_split', 'data.yaml')
    
    # 检查CUDA可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载预训练模型
    print("\n正在加载YOLOv8n预训练模型...")
    model = YOLO('yolov8n.pt')
    
    # 配置训练参数
    epochs = 100
    batch_size = 16
    img_size = 640
    
    print(f"\n数据集配置文件: {data_yaml}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"图像尺寸: {img_size}")
    
    # 开始训练
    model.train(
        data=data_yaml,           # ← 使用正确的数据集配置文件路径
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

if __name__ == '__main__':
    main()