# YOLOv8 线缆缺陷检测训练平台

## 文件夹结构

```
训练平台/
├── train.py          # 训练脚本
├── test.py           # 测试与评估脚本
├── inference.py      # 推理脚本（图像/视频/摄像头）
├── weights/          # 模型权重文件夹
│   └── best.pt       # 训练后的最佳模型（需要放置）
├── data/             # 数据集文件夹
│   ├── data.yaml     # 数据集配置文件
│   ├── train/        # 训练集
│   │   ├── images/   # 训练图像
│   │   └── labels/   # 训练标签
│   ├── val/          # 验证集
│   │   ├── images/
│   │   └── labels/
│   └── test/         # 测试集
│       ├── images/
│       └── labels/
└── output/           # 训练输出文件夹
    ├── train/        # 训练结果
    └── test_results/ # 测试结果
```

## 使用说明

### 1. 环境准备

安装必要的依赖包：

```bash
pip install ultralytics opencv-python torch torchvision
```

### 2. 准备数据集

将你的数据集按照以下结构放置：

```
data/
├── train/
│   ├── images/      # 放置训练图像 (jpg/png)
│   └── labels/      # 放置YOLO格式标签 (.txt)
├── val/
│   ├── images/      # 放置验证图像
│   └── labels/      # 放置验证标签
└── test/
    ├── images/      # 放置测试图像
    └── labels/      # 放置测试标签
```

**重要：** 根据 `data/data.yaml` 文件配置你的类别名称和数量。

### 3. 模型训练

运行训练脚本：

```bash
python train.py
```

训练参数可以在 `train.py` 中修改：
- `epochs`: 训练轮数（默认100）
- `batch_size`: 批次大小（默认16）
- `img_size`: 图像尺寸（默认640）

训练完成后，最佳模型保存在 `output/train/weights/best.pt`

### 4. 模型测试

**方式1：在测试图像上推理**
```bash
python test.py
选择 1
```

**方式2：在验证集上评估性能**
```bash
python test.py
选择 2
```

### 5. 模型推理

**图像推理：**
```bash
python inference.py
选择 1
输入图像路径
```

**视频推理：**
```bash
python inference.py
选择 2
输入视频路径
```

**摄像头实时推理：**
```bash
python inference.py
选择 3
输入摄像头ID（默认0）
```

## 配置说明

### data/data.yaml 配置文件

根据你的实际缺陷类型修改：

```yaml
# 类别数量
nc: 3

# 类别名称
names:
  0: 裂纹
  1: 表面破损
  2: 凹坑
```

如果有更多缺陷类型，按照格式添加，并修改 `nc` 的值。

## 常见问题

**Q: 训练时显示CUDA不可用怎么办？**
A: 检查显卡驱动和CUDA版本，或自动使用CPU训练（速度较慢）

**Q: 如何使用已经训练好的模型？**
A: 将 `best.pt` 放在 `weights/` 文件夹下，或指定完整路径

**Q: 如何调整置信度阈值？**
A: 在运行 `inference.py` 或 `test.py` 时输入想要的置信度值

**Q: 训练中断后如何继续训练？**
A: 修改 `train.py` 中的 `model.load('output/train/weights/last.pt')` 加载最后一次检查点

## 注意事项

1. 确保数据集标签格式为YOLO格式（类别号 x_center y_center width height，全部归一化到0-1）
2. 图像和标签的文件名必须一致（仅扩展名不同）
3. 训练前检查data.yaml中的路径是否正确
4. GPU显存不足时减小batch_size
