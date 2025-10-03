# Computer Vision Learning: 实践与笔记仓库

本仓库是一个系统化的计算机视觉（CV）学习项目，整合了交互式笔记、可复用代码模块、实验脚本和学习文档，旨在通过“理论+实践”结合的方式，逐步掌握CV核心技术（从基础图像处理到深度学习模型应用）。


## 📂 目录结构说明
以下是仓库的核心目录与文件功能解析，按“学习流程优先级”排序，方便快速定位所需内容：

| 目录/文件               | 核心功能与内容说明                                                                 |
|-------------------------|----------------------------------------------------------------------------------|
| `README.md`             | 项目总览、目录导航、环境配置指南（当前文档）                                      |
| `requirements.txt`      | Python 依赖包列表（含 OpenCV、PyTorch、Matplotlib 等 CV 常用库），用于 `pip` 安装  |
| `environment.yml`       | （可选）Conda 环境配置文件，一键创建兼容的虚拟环境（适合需要完整环境隔离的场景）  |
| `LICENSE`               | MIT 开源许可证（允许自由使用、修改、分发，包括商用，需保留版权声明）              |
| `.gitignore`            | Git 忽略规则（排除 `data/raw/`、虚拟环境、日志文件等，避免仓库臃肿）              |
|                         |                                                                                  |
| `notebooks/`            | **交互式学习核心**：Jupyter Notebook 实践笔记，按 CV 学习主题排序，可直接运行验证 |
| ├─ `01_image_basics.ipynb`  | 基础图像处理（读取/保存、色彩空间转换、几何变换等，OpenCV 入门）                 |
| ├─ `02_edge_detection.ipynb` | 边缘检测算法（Sobel、Canny、Laplacian 等，附原理与代码对比）                     |
| └─ `03_object_detection_yolo.ipynb` | YOLO 目标检测实践（加载预训练模型、图像/视频推理、结果可视化）                   |
|                         |                                                                                  |
| `src/`                  | **可复用代码模块**：封装通用功能，避免重复开发，支持在 notebooks/experiments 中调用 |
| ├─ `__init__.py`        | Python 包标识，使 `src` 可作为模块导入                                           |
| ├─ `utils.py`           | 核心工具函数（如图像加载/预处理、结果可视化（画检测框/分割掩码）、指标计算（IOU））|
| └─ `models/`            | 自定义模型或第三方模型封装（如简化版 CNN 分类器、YOLO 推理封装）                 |
|    └─ `__init__.py`     | 模型包标识，方便导入特定模型（如 `from src.models import SimpleCNN`）             |
|                         |                                                                                  |
| `data/`                 | **数据存储与管理**：区分原始数据与处理后数据，符合 CV 项目数据流程规范           |
| ├─ `raw/`               | 原始数据（如小样本图像、测试视频，大数据集建议通过脚本下载，此目录已加入 `.gitignore`） |
| └─ `processed/`         | 预处理后的数据（如裁剪后的图像、标注文件，可选，用于实验直接调用）                |
|                         |                                                                                  |
| `experiments/`          | **命令行实验脚本**：用于批量训练、模型评估等场景，适合脱离 Notebook 自动化运行   |
| ├─ `train_classifier.py` | 图像分类模型训练脚本（如训练 CNN 识别 MNIST/CIFAR，支持调参、日志记录）          |
| └─ `evaluate_model.py`  | 模型评估脚本（计算准确率、AP、mIoU 等指标，输出评估报告或可视化结果）            |
|                         |                                                                                  |
| `docs/`                 | **学习文档与笔记**：补充理论知识，记录学习过程中的关键总结                       |
| └─ `notes.md`           | CV 核心知识点总结（如边缘检测原理、YOLO 网络结构、数据增强技巧等，配合 notebooks 阅读） |
|                         |                                                                                  |
| `assets/`               | **文档资源**：存放 README 或 notebooks 中引用的图片、动图，提升文档可读性       |
| └─ `demo.gif`           | 示例动图（如目标检测效果演示、图像分割结果对比，用于 README 展示）                |


## 🛠️ 环境配置
### 方式1：使用 pip（通用 Python 环境）
```bash
# 克隆仓库到本地
git clone https://github.com/你的用户名/computer-vision-learning.git
cd computer-vision-learning

# 安装依赖（建议先创建虚拟环境）
pip install -r requirements.txt
```

### 方式2：使用 Conda（推荐，避免版本冲突）
```bash
# 基于 environment.yml 创建环境（环境名默认在文件中定义，如 cv-learning）
conda env create -f environment.yml

# 激活环境
conda activate cv-learning
```


## 🚀 快速开始
1. **从交互式笔记入门**（推荐新手）：
   ```bash
   # 启动 Jupyter Notebook
   jupyter notebook

   # 在浏览器中打开 notebooks/ 目录，按编号顺序运行（01→02→03）
   ```

2. **运行实验脚本**（适合有基础后批量训练/评估）：
   ```bash
   # 训练图像分类模型（示例：训练 MNIST 分类器）
   python experiments/train_classifier.py --dataset mnist --epochs 10

   # 评估已训练模型
   python experiments/evaluate_model.py --model_path ./models/trained_cnn.pth
   ```


## ⚠️ 注意事项
1. **数据存储**：`data/raw/` 目录已加入 `.gitignore`，若需使用大型数据集（如 COCO、ImageNet），建议在 `data/` 下添加 `download_dataset.py` 脚本（记录下载逻辑），避免直接上传大文件。
2. **代码复用**：`src/` 目录下的工具函数（如 `src.utils.plot_detection`）可在 Notebook 中直接导入使用，示例：
   ```python
   from src.utils import load_image, plot_detection
   img = load_image("./data/raw/test.jpg")
   plot_detection(img, bboxes, labels)  # 可视化目标检测结果
   ```
3. **文档更新**：若新增目录/文件（如 `notebooks/04_segmentation_unet.ipynb`），建议同步更新本 `README.md` 的目录说明，保持结构清晰。


## 📄 开源许可
本项目基于 [MIT 许可证](LICENSE) 开源，你可以：
- 自由使用、修改、分发本项目代码（包括商用）；
- 无需向作者申请许可，但需在衍生作品中保留原始版权声明和许可证文本；
- 作者不对代码的使用风险（如 bug、数据安全）承担法律责任。