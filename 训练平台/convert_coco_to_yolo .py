"""
数据集划分工具（简化版）
将 data/train 下的所有文件划分为 train/val/test
"""
import os
import shutil
from pathlib import Path
import random

def split_dataset_simple(data_dir='data/train', output_dir='data_split', train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    简单的数据集划分
    将 data/train 下的所有文件按比例移动到 train/val/test 文件夹
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # 检查比例和是否为 1
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError(f"划分比例之和必须为 1，当前为 {train_ratio + val_ratio + test_ratio}")

    # 获取所有文件
    all_files = list(data_path.glob('*.*'))
    all_files = [f for f in all_files if f.is_file()]  # 只包含文件，不包含文件夹

    if not all_files:
        raise ValueError(f"未找到任何文件: {data_path}")

    print(f"\n总共找到 {len(all_files)} 个文件")

    # 随机打乱
    random.shuffle(all_files)

    # 计算划分数量
    total = len(all_files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    print(f"\n数据集划分:")
    print(f"  训练集: {train_count} 个文件 ({train_ratio*100:.1f}%)")
    print(f"  验证集: {val_count} 个文件 ({val_ratio*100:.1f}%)")
    print(f"  测试集: {test_count} 个文件 ({test_ratio*100:.1f}%)")

    # 划分数据集
    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count + val_count]
    test_files = all_files[train_count + val_count:]

    # 创建输出目录
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    # 复制/移动文件
    print(f"\n开始处理文件...")
    for i, f in enumerate(train_files):
        shutil.copy2(f, output_path / 'train' / f.name)
    print(f"  训练集处理完成 ({len(train_files)} 个文件)")

    for i, f in enumerate(val_files):
        shutil.copy2(f, output_path / 'val' / f.name)
    print(f"  验证集处理完成 ({len(val_files)} 个文件)")

    for i, f in enumerate(test_files):
        shutil.copy2(f, output_path / 'test' / f.name)
    print(f"  测试集处理完成 ({len(test_files)} 个文件)")

    print(f"\n输出目录结构:")
    print(f"{output_path}/")
    print(f"  ├── train/  ({len(train_files)} 个文件)")
    print(f"  ├── val/    ({len(val_files)} 个文件)")
    print(f"  └── test/   ({len(test_files)} 个文件)")

    return {
        'total': total,
        'train': {'count': len(train_files), 'ratio': train_ratio},
        'val': {'count': len(val_files), 'ratio': val_ratio},
        'test': {'count': len(test_files), 'ratio': test_ratio},
    }

if __name__ == '__main__':
    print("="*60)
    print("数据集划分工具（简化版）")
    print("="*60)
    print("\n使用方法:")
    print("  python split_dataset_simple.py")
    print("\n可选参数:")
    print("  --data_dir     原始数据目录 (默认: data/train)")
    print("  --output_dir   输出目录     (默认: data_split)")
    print("  --train_ratio  训练集比例   (默认: 0.7)")
    print("  --val_ratio    验证集比例   (默认: 0.2)")
    print("  --test_ratio   测试集比例   (默认: 0.1)")
    print("  --seed         随机种子     (默认: None)")

    # 默认参数
    data_dir = 'data/train'
    output_dir = 'data_split'
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    seed = None

    # 解析命令行参数（简化版）
    import sys
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--data_dir':
            data_dir = args[i+1]
            i += 2
        elif args[i] == '--output_dir':
            output_dir = args[i+1]
            i += 2
        elif args[i] == '--train_ratio':
            train_ratio = float(args[i+1])
            i += 2
        elif args[i] == '--val_ratio':
            val_ratio = float(args[i+1])
            i += 2
        elif args[i] == '--test_ratio':
            test_ratio = float(args[i+1])
            i += 2
        elif args[i] == '--seed':
            seed = int(args[i+1])
            i += 2
        else:
            i += 1

    print(f"\n输入目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"\n划分比例:")
    print(f"  训练集: {train_ratio*100:.1f}%")
    print(f"  验证集: {val_ratio*100:.1f}%")
    print(f"  测试集: {test_ratio*100:.1f}%")
    print(f"\n随机种子: {seed if seed is not None else '未设置'}")

    # 确认
    confirm = input("\n确认开始划分？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        sys.exit(0)

    # 设置随机种子
    if seed is not None:
        random.seed(seed)

    try:
        split_info = split_dataset_simple(data_dir, output_dir, train_ratio, val_ratio, test_ratio)

        print("\n" + "="*60)
        print("数据集划分完成!")
        print("="*60)
        print("\n下一步:")
        print("1. 检查 data_split/ 目录下的文件结构")
        print("2. 更新 data/data.yaml 配置文件")
        print("3. 运行 python train.py 开始训练")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
