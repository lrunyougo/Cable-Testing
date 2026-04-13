import sys
import os
import subprocess
import time
from pathlib import Path
import traceback
import platform
import shutil

# 环境配置：检测CUDA并安装YOLOv8
def setup_environment():
    """检测CUDA可用性并配置YOLOv8环境"""
    print("=" * 60)
    print("环境配置开始...")
    print("=" * 60)
    
    # 检测PyTorch
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
    except ImportError:
        print("PyTorch未安装，正在安装...")
        install_pytorch()
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
    
    # 检测CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用: {cuda_available}")
    if cuda_available:
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    
    # 安装ultralytics（YOLOv8）
    try:
        import ultralytics
        print(f"Ultralytics 版本: {ultralytics.__version__}")
    except ImportError:
        print("正在安装ultralytics (YOLOv8)...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            print("ultralytics安装成功")
        except Exception as e:
            print(f"安装ultralytics失败: {e}")
    
    print("=" * 60)
    print("环境配置完成")
    print("=" * 60)
    return cuda_available

def install_pytorch():
    """根据CUDA可用性安装合适的PyTorch版本"""
    # 首先检查CUDA
    try:
        import torch
        # 如果已有torch，检查CUDA支持
        if torch.cuda.is_available():
            print("当前PyTorch已支持CUDA")
            return
    except ImportError:
        pass
    
    # 检测系统是否有CUDA
    cuda_detected = False
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            cuda_detected = True
            print("检测到NVIDIA GPU和CUDA驱动")
    except FileNotFoundError:
        print("未检测到nvidia-smi，将安装CPU版本")
    
    if cuda_detected:
        print("正在安装PyTorch with CUDA支持...")
        try:
            # 安装支持CUDA的PyTorch
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
            print("PyTorch (CUDA) 安装成功")
        except Exception as e:
            print(f"CUDA版本安装失败: {e}，回退到CPU版本")
            install_pytorch_cpu()
    else:
        install_pytorch_cpu()

def install_pytorch_cpu():
    """安装CPU版本的PyTorch"""
    print("正在安装PyTorch CPU版本...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
        print("PyTorch (CPU) 安装成功")
    except Exception as e:
        print(f"PyTorch安装失败: {e}")
        # 尝试默认安装
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])

def install_package(package):
    """安装Python包（兼容打包环境）"""
    if getattr(sys, 'frozen', False):
        print(f"打包环境中跳过安装 {package}")
        return
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception as e:
        print(f"安装 {package} 失败: {str(e)}")

# 执行环境配置
CUDA_AVAILABLE = setup_environment()

import torch
from ultralytics import YOLO

# 资源路径处理函数
def resource_path(relative_path):
    """获取资源绝对路径，适用于开发环境和PyInstaller打包后的环境"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# 导入必要的包或安装它们
try:
    import numpy as np
except ImportError:
    print("正在安装numpy...")
    install_package("numpy")
    import numpy as np

try:
    import cv2
except ImportError:
    print("正在安装opencv-python...")
    install_package("opencv-python")
    import cv2

# 强制使用PySide6
print("正在导入PySide6...")
try:
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
                                 QHBoxLayout, QLabel, QFileDialog, QComboBox, QSlider, QProgressBar,
                                 QMessageBox, QTextEdit, QRadioButton, QButtonGroup)
    from PySide6.QtGui import QPixmap, QImage
    from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer
    USING_PYSIDE6 = True
    print("PySide6导入成功")
except ImportError as e:
    print(f"PySide6导入失败: {e}")
    print("请确保已安装PySide6: pip install PySide6")
    sys.exit(1)

# YOLOv8检测线程
class DetectionThread(QThread):
    update_image = Signal(np.ndarray)
    update_progress = Signal(int)
    update_log = Signal(str)
    detection_finished = Signal()
    update_objects = Signal(dict, bool)
    
    def __init__(self, model_type, source, confidence, device='auto', is_camera=False, camera_id=0, custom_model_path=""):
        super().__init__()
        # 自动选择设备
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.update_log.emit(f"使用设备: {self.device.upper()}")
        self.model_type = model_type
        self.source = source
        self.confidence = confidence
        self.running = True
        self.is_camera = is_camera
        self.camera_id = camera_id
        self.custom_model_path = custom_model_path

    def run(self):
        try:
            self.update_log.emit(f"加载YOLOv8模型，使用设备: {self.device}...")
            
            # 加载模型
            model = self.load_model()
            if model is None:
                self.detection_finished.emit()
                return
            
            # 设置置信度
            model.conf = self.confidence
            
            # 处理源
            if self.is_camera:
                self.process_camera(model)
            else:
                # 检查文件是否存在
                if not os.path.exists(self.source):
                    self.update_log.emit(f"错误: 找不到文件 {self.source}")
                    self.detection_finished.emit()
                    return
                self.process_file(model)
                
        except Exception as e:
            self.update_log.emit(f"错误: {str(e)}")
            error_trace = traceback.format_exc()
            self.update_log.emit(f"详细错误: {error_trace}")
        finally:
            self.detection_finished.emit()

    def load_model(self):
        """加载YOLOv8模型"""
        try:
            # 判断是否使用自定义模型
            if self.model_type == "自定义模型" and self.custom_model_path:
                model_path = self.custom_model_path
                self.update_log.emit(f"使用自定义模型: {Path(model_path).name}")
                model = YOLO(model_path)
            else:
                # 根据模型类型选择模型
                model_map = {
                    "yolov8n": "yolov8n.pt",
                    "yolov8s": "yolov8s.pt",
                    "yolov8m": "yolov8m.pt",
                    "yolov8l": "yolov8l.pt",
                    "yolov8x": "yolov8x.pt",
                    "quexian": "quexian.pt"
                }
                
                # 支持YOLOv5模型名的兼容
                if self.model_type in ["yolov5s", "yolov5m", "yolov5l", "yolov5x"]:
                    yolo8_model = self.model_type.replace("yolov5", "yolov8")
                    self.update_log.emit(f"将 {self.model_type} 映射到 {yolo8_model}")
                    model_type = yolo8_model
                else:
                    model_type = self.model_type
                
                # 搜索本地模型文件
                possible_paths = [
                    resource_path(f"{model_type}.pt"),
                    resource_path(f"models/{model_type}.pt"),
                    f"{model_type}.pt",
                    f"models/{model_type}.pt"
                ]
                
                model_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        model_path = path
                        self.update_log.emit(f"找到本地模型: {path}")
                        break
                
                if model_path:
                    model = YOLO(model_path)
                else:
                    # 从预训练模型加载
                    self.update_log.emit(f"从在线加载预训练模型: {model_type}")
                    model = YOLO(f"{model_type}.pt")
            
            self.update_log.emit(f"模型加载成功: {self.model_type}")
            return model
            
        except Exception as e:
            self.update_log.emit(f"模型加载失败: {str(e)}")
            return None

    def process_camera(self, model):
        """处理摄像头输入"""
        self.update_log.emit("开始摄像头检测...")
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            self.update_log.emit(f"无法打开摄像头 {self.camera_id}")
            return
        
        while self.running:
            ret, img0 = cap.read()
            if not ret:
                break
            
            try:
                # 使用YOLOv8进行推理
                results = model(img0, device=self.device, conf=self.confidence, verbose=False)
                
                # 渲染结果
                annotated_frame = results[0].plot()
                
                # 统计检测到的物体
                objects_count = {}
                has_person = False
                
                if len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls)
                        cls_name = model.names[cls_id]
                        
                        if cls_name in objects_count:
                            objects_count[cls_name] += 1
                        else:
                            objects_count[cls_name] = 1
                        
                        if cls_name.lower() == 'person':
                            has_person = True
                
                # 发送信号
                self.update_objects.emit(objects_count, has_person)
                self.update_image.emit(annotated_frame)
                self.update_progress.emit(100)
                
            except Exception as e:
                self.update_log.emit(f"处理图像时出错: {str(e)}")
                break
            
            time.sleep(0.03)  # 控制帧率
        
        cap.release()
        self.update_log.emit("摄像头检测结束")

    def process_file(self, model):
        """处理文件（图片或视频）"""
        self.update_log.emit(f"处理文件: {self.source}")
        
        is_video = self.source.lower().endswith(('mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'))
        
        if is_video:
            self.process_video(model)
        else:
            self.process_image(model)

    def process_video(self, model):
        """处理视频文件"""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.update_log.emit("无法打开视频文件")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.update_log.emit(f"视频总帧数: {total_frames}, FPS: {fps}")
        
        frame_count = 0
        
        while self.running and cap.isOpened():
            ret, img0 = cap.read()
            if not ret:
                break
            
            frame_count += 1
            self.update_progress.emit(int(100 * frame_count / total_frames))
            
            try:
                # 使用YOLOv8进行推理
                results = model(img0, device=self.device, conf=self.confidence, verbose=False)
                
                # 渲染结果
                annotated_frame = results[0].plot()
                
                # 统计检测到的物体
                objects_count = {}
                has_person = False
                
                if len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls)
                        cls_name = model.names[cls_id]
                        
                        if cls_name in objects_count:
                            objects_count[cls_name] += 1
                        else:
                            objects_count[cls_name] = 1
                        
                        if cls_name.lower() == 'person':
                            has_person = True
                
                # 发送信号
                self.update_objects.emit(objects_count, has_person)
                self.update_image.emit(annotated_frame)
                
            except Exception as e:
                self.update_log.emit(f"处理帧 {frame_count} 时出错: {str(e)}")
                continue
            
            time.sleep(1/fps)
        
        cap.release()
        self.update_log.emit(f"视频处理完成，共处理 {frame_count} 帧")

    def process_image(self, model):
        """处理单张图片"""
        img0 = cv2.imread(self.source)
        if img0 is None:
            self.update_log.emit("无法打开图像文件")
            return
        
        try:
            # 使用YOLOv8进行推理
            results = model(img0, device=self.device, conf=self.confidence, verbose=False)
            
            # 渲染结果
            annotated_frame = results[0].plot()
            
            # 统计检测到的物体
            objects_count = {}
            has_person = False
            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls_id = int(box.cls)
                    cls_name = model.names[cls_id]
                    
                    if cls_name in objects_count:
                        objects_count[cls_name] += 1
                    else:
                        objects_count[cls_name] = 1
                    
                    if cls_name.lower() == 'person':
                        has_person = True
            
            # 发送信号
            self.update_objects.emit(objects_count, has_person)
            self.update_image.emit(annotated_frame)
            self.update_progress.emit(100)
            self.update_log.emit("图像处理完成")
            
        except Exception as e:
            self.update_log.emit(f"处理图像时出错: {str(e)}")

    def stop(self):
        self.running = False
        self.wait()


class YoloV8App(QMainWindow):
    def __init__(self, device=None):
        super().__init__()
        
        # 检测CUDA可用性
        self.cuda_available = torch.cuda.is_available()
        self.device = 'cpu'
        
        self.init_ui()
        self.detection_thread = None
        self.camera_thread = None

    def init_ui(self):
        # 设置窗口标题和大小
        self.setWindowTitle("YOLOv8 目标检测")
        self.setGeometry(100, 100, 1000, 800)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建控制面板
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        # 设备选择布局
        device_layout = QVBoxLayout()
        device_label = QLabel("运行设备:")
        
        # 设备选择按钮组
        self.device_group = QButtonGroup()
        self.cpu_radio = QRadioButton("CPU")
        self.gpu_radio = QRadioButton("GPU (CUDA)")
        
        # 默认选择CPU
        self.cpu_radio.setChecked(True)
        
        # 如果CUDA不可用，禁用GPU选项
        if not self.cuda_available:
            self.gpu_radio.setEnabled(False)
            self.gpu_radio.setToolTip("CUDA不可用")
        
        self.device_group.addButton(self.cpu_radio)
        self.device_group.addButton(self.gpu_radio)
        
        # 连接设备切换事件
        self.cpu_radio.toggled.connect(self.toggle_device)
        self.gpu_radio.toggled.connect(self.toggle_device)
        
        device_radio_layout = QHBoxLayout()
        device_radio_layout.addWidget(self.cpu_radio)
        device_radio_layout.addWidget(self.gpu_radio)
        
        device_layout.addWidget(device_label)
        device_layout.addLayout(device_radio_layout)
        
        # 添加到控制布局
        control_layout.addLayout(device_layout)
        
        # 模型选择
        model_layout = QVBoxLayout()
        model_label = QLabel("模型:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x", "quexian", "自定义模型"])
        self.model_combo.currentIndexChanged.connect(self.on_model_selection_changed)
        
        self.model_info_label = QLabel("选择'自定义模型'后点击下方按钮加载您的模型文件(.pt)")
        self.model_info_label.setStyleSheet("color: blue; font-size: 10px;")
        
        self.custom_model_path = ""
        self.custom_model_btn = QPushButton("加载自定义模型")
        self.custom_model_btn.clicked.connect(self.select_custom_model)
        self.custom_model_btn.setEnabled(False)
        self.custom_model_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.model_info_label)
        model_layout.addWidget(self.custom_model_btn)
        
        # 确保将模型布局添加到控制布局
        control_layout.addLayout(model_layout)
        
        # 置信度阈值
        conf_layout = QVBoxLayout()
        conf_label = QLabel("置信度阈值:")
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(99)
        self.conf_slider.setValue(25)
        self.conf_value_label = QLabel("0.25")
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value_label)
        control_layout.addLayout(conf_layout)
        
        # 输入源选择
        source_layout = QVBoxLayout()
        source_label = QLabel("输入源:")
        
        # 单选按钮组
        self.source_group = QButtonGroup()
        self.file_radio = QRadioButton("文件")
        self.camera_radio = QRadioButton("摄像头")
        self.file_radio.setChecked(True)
        
        self.source_group.addButton(self.file_radio)
        self.source_group.addButton(self.camera_radio)
        
        source_radio_layout = QHBoxLayout()
        source_radio_layout.addWidget(self.file_radio)
        source_radio_layout.addWidget(self.camera_radio)
        
        # 摄像头ID选择
        self.camera_id_combo = QComboBox()
        self.camera_id_combo.addItems(["0", "1", "2", "3"])
        self.camera_id_combo.setEnabled(False)
        
        # 文件选择按钮
        self.file_btn = QPushButton("选择文件")
        self.file_btn.clicked.connect(self.open_file_dialog)
        
        # 连接单选按钮变化事件
        self.file_radio.toggled.connect(self.toggle_source_controls)
        self.camera_radio.toggled.connect(self.on_camera_selected)
        
        source_layout.addWidget(source_label)
        source_layout.addLayout(source_radio_layout)
        source_layout.addWidget(self.camera_id_combo)
        source_layout.addWidget(self.file_btn)
        
        control_layout.addLayout(source_layout)
        
        # 开始检测按钮
        self.detect_btn = QPushButton("开始检测")
        self.detect_btn.clicked.connect(self.start_detection)
        control_layout.addWidget(self.detect_btn)
        
        # 停止检测按钮
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        main_layout.addWidget(control_panel)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # 图像显示区
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("选择一个输入源以开始检测")
        
        # 创建水平布局，左侧是图像，右侧是检测结果和指示灯
        image_results_layout = QHBoxLayout()
        
        # 将图像标签放入左侧
        image_container = QWidget()
        image_container_layout = QVBoxLayout(image_container)
        image_container_layout.addWidget(self.image_label)
        image_results_layout.addWidget(image_container, 7)
        
        # 右侧容器，包含检测结果和指示灯
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 检测结果显示区
        results_group = QWidget()
        results_layout = QVBoxLayout(results_group)
        results_layout.addWidget(QLabel("检测到的物体:"))
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)
        right_layout.addWidget(results_group)
        
        # 添加占位符，确保上面的元素靠顶部
        right_layout.addStretch()
        
        # 将右侧面板添加到水平布局
        image_results_layout.addWidget(right_panel, 3)
        
        # 将新的布局添加到主布局中
        main_layout.addLayout(image_results_layout)
        
        # 日志区域
        log_label = QLabel("日志:")
        main_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        main_layout.addWidget(self.log_text)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
        self.source_path = ""
        self.log("YOLOv8 目标检测应用已启动")
        
        # 检查CUDA状态
        if self.cuda_available:
            self.log(f"CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            self.log("CUDA不可用，将使用CPU模式")
        
        # 检查是否在打包环境中
        if getattr(sys, 'frozen', False):
           self.log(f"在打包环境中运行: {sys._MEIPASS}")
    
    def toggle_device(self):
        """切换CPU/GPU设备"""
        if self.gpu_radio.isChecked() and self.cuda_available:
            self.device = 'cuda'
            self.log("切换到GPU(CUDA)模式")
            self.statusBar().showMessage("使用GPU进行推理")
        else:
            self.device = 'cpu'
            self.log("切换到CPU模式")
            self.statusBar().showMessage("使用CPU进行推理")
    
    def on_model_selection_changed(self, index):
        """当模型选择改变时更新UI"""
        is_custom = self.model_combo.currentText() == "自定义模型"
        self.custom_model_btn.setEnabled(is_custom)
        if is_custom and not self.custom_model_path:
           self.log("请选择自定义模型文件")
    
    def select_custom_model(self):
        """打开文件对话框选择自定义YOLOv8权重文件"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择YOLOv8模型权重", "",
            "模型文件 (*.pt);;所有文件 (*)", options=options
        )
        
        if file_name:
            self.custom_model_path = file_name
            model_name = Path(file_name).name
            self.log(f"已选择自定义模型: {model_name}")
            self.statusBar().showMessage(f"已加载自定义模型: {model_name}")
    
    def update_conf_label(self, value):
        """更新置信度标签显示"""
        confidence = value / 100.0
        self.conf_value_label.setText(f"{confidence:.2f}")
    
    def open_file_dialog(self):
        """打开文件选择对话框"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择图像或视频文件",
            "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp);;视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;所有文件 (*)",
            options=options
        )
        
        if file_name:
            self.source_path = file_name
            self.log(f"已选择文件: {file_name}")
            self.statusBar().showMessage(f"已选择: {Path(file_name).name}")
            
            # 如果是图像，预览一下
            if file_name.lower().endswith(('jpg', 'jpeg', 'png', 'bmp')):
                try:
                    pixmap = QPixmap(file_name)
                    pixmap = pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio)
                    self.image_label.setPixmap(pixmap)
                except Exception as e:
                    self.log(f"无法预览图像: {str(e)}")
            else:
                self.image_label.setText("视频文件将在检测时显示")
    
    def on_camera_selected(self, checked):
        """当选择摄像头时启动预览"""
        if checked:
            self.camera_id_combo.setEnabled(True)
            self.file_btn.setEnabled(False)
            self.start_camera_preview()
        else:
            if self.camera_thread and self.camera_thread.isRunning():
                self.camera_thread.stop()
                self.camera_thread.wait()
                self.camera_thread = None
    
    def start_camera_preview(self):
        """启动摄像头预览"""
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()
        
        camera_id = int(self.camera_id_combo.currentText())
        self.log(f"启动摄像头预览, ID: {camera_id}")
        
        # 停止当前运行的线程（如果存在）
        if hasattr(self, 'detection_thread') and self.detection_thread is not None:
            self.detection_thread.running = False
            self.detection_thread.quit()
            self.detection_thread.wait()
        
        # 直接使用YOLOv8处理摄像头流
        self.detection_thread = DetectionThread(
            model_type="yolov8s",
            source="0",
            confidence=0.5,
            is_camera=True,
            camera_id=camera_id
        )
        self.detection_thread.update_image.connect(self.update_image)
        self.detection_thread.start()
    
    def toggle_source_controls(self):
        """切换文件/摄像头控制"""
        is_file = self.file_radio.isChecked()
        self.file_btn.setEnabled(is_file)
        self.camera_id_combo.setEnabled(not is_file)
        
        if is_file and self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread = None
    
    def update_image(self, img):
        """更新图像显示"""
        if isinstance(img, np.ndarray):
            height, width, channel = img.shape
            bytesPerLine = 3 * width
            qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qImg)
            
            # 根据窗口大小调整图像
            label_size = self.image_label.size()
            pixmap = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
            
            self.image_label.setPixmap(pixmap)
        else:
            self.log("收到无效图像数据")
    
    def update_detected_objects(self, objects_dict, person):
        """更新检测到的物体列表"""
        self.results_text.clear()

        if objects_dict:
            sorted_objects = sorted(objects_dict.items(), key=lambda x: x[1], reverse=True)
            for obj_name, count in sorted_objects:
                self.results_text.append(f"{obj_name}: {count}")
        else:
            self.results_text.append("未检测到物体")
    
    def start_detection(self):
        """开始检测过程"""
        if self.file_radio.isChecked() and not self.source_path:
            QMessageBox.warning(self, "警告", "请先选择一个文件")
            return
        
        model_type = self.model_combo.currentText()
        confidence = self.conf_slider.value() / 100.0
        is_camera = self.camera_radio.isChecked()
        
        if is_camera:
            source = int(self.camera_id_combo.currentText())
        else:
            source = self.source_path
        
        # 停止任何正在运行的线程
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.detection_thread.wait()
        
        if self.camera_thread and self.camera_thread.isRunning() and not is_camera:
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread = None
        
        # 更新UI状态
        self.detect_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # 自定义模型的处理
        custom_model_path = ""
        if model_type == "自定义模型":
            custom_model_path = self.custom_model_path
            if not custom_model_path:
                QMessageBox.warning(self, "警告", "请选择自定义模型文件")
                self.detect_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                return
        
        # 创建并启动检测线程
        self.log(f"开始检测: 模型={model_type}, 置信度={confidence}, 设备={self.device}, 源={'摄像头' if is_camera else source}")
        self.detection_thread = DetectionThread(
            model_type, source, confidence,
            device=self.device,
            is_camera=is_camera,
            camera_id=int(self.camera_id_combo.currentText()) if is_camera else 0,
            custom_model_path=custom_model_path
        )
        
        # 连接信号
        self.detection_thread.update_image.connect(self.update_image)
        self.detection_thread.update_progress.connect(self.progress_bar.setValue)
        self.detection_thread.update_log.connect(self.log)
        self.detection_thread.detection_finished.connect(self.on_detection_finished)
        self.detection_thread.update_objects.connect(self.update_detected_objects)
        
        # 启动线程
        self.detection_thread.start()
    
    def on_detection_finished(self):
        """检测完成后的回调"""
        self.detect_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log("检测过程已完成")
    
    def stop_detection(self):
        """停止检测过程"""
        if self.detection_thread and self.detection_thread.isRunning():
            self.log("正在停止检测...")
            self.detection_thread.stop()
            self.detection_thread.wait()
            self.log("检测已停止")
        
        self.stop_btn.setEnabled(False)
        self.detect_btn.setEnabled(True)
    
    def log(self, message):
        """向日志区域添加消息"""
        timestamp = time.strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        self.log_text.append(log_message)
        print(log_message)
        
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


def main():
    try:
        # 创建日志目录和文件
        log_dir = os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else __file__)
        log_file = os.path.join(log_dir, "app_startup.log")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 应用启动\n")
            f.write(f"Python版本: {sys.version}\n")
            f.write(f"操作系统: {platform.platform()}\n")
            f.write(f"应用路径: {os.path.abspath('.')}\n")
            
            # 检测CUDA可用性并记录
            try:
                import torch
                f.write(f"PyTorch版本: {torch.__version__}\n")
                cuda_available = torch.cuda.is_available()
                f.write(f"CUDA可用: {cuda_available}\n")
                if cuda_available:
                    f.write(f"CUDA版本: {torch.version.cuda}\n")
                    f.write(f"CUDA设备: {torch.cuda.get_device_name(0)}\n")
            except Exception as e:
                f.write(f"检查CUDA时出错: {e}\n")
        
        # 打包环境特殊处理
        if getattr(sys, 'frozen', False):
            os.chdir(os.path.dirname(sys.executable))
            
            if hasattr(sys, "_MEIPASS"):
                os.environ["PATH"] = sys._MEIPASS + os.pathsep + os.environ["PATH"]
        
        # 创建应用和主窗口
        app = QApplication(sys.argv)
        QApplication.setStyle("Fusion")
        window = YoloV8App()
        window.show()
        
        return app.exec() if USING_PYSIDE6 else app.exec_()
    
    except Exception as e:
        error_msg = f"启动失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
        # 保存错误日志
        try:
            log_dir = os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else __file__)
            error_log = os.path.join(log_dir, "startup_error.log")
            with open(error_log, "w", encoding="utf-8") as f:
                f.write(error_msg)
            print(f"错误已记录到: {error_log}")
        except:
            pass
        
        # 弹出提示框
        try:
            app = QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(None, "启动错误",
                               f"程序启动失败:\n{str(e)}\n\n详细错误已保存到startup_error.log")
        except:
            pass
        
        input("按Enter键退出...")
        return 1


if __name__ == "__main__":
    sys.exit(main())
