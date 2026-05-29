# Planktoscope — 显微图像切割预览与批处理工具

交互式预览和调参显微镜图像中的浮游生物粒子检测参数，一键批量切割输出单粒子图像。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行

**Windows:**
```
双击 run.bat
```

**macOS / Linux:**
```bash
bash run.sh
```

或直接：
```bash
python planktoscope_segment_viewer.py
```

## 功能

- **交互式预览**：打开图片文件夹，浏览图像，实时显示粒子检测的绿色边界框
- **多种二值化算法**：支持 Adaptive Threshold、Otsu、Global Threshold、Canny Edge 四种方法，下拉切换后参数面板自动适配
- **参数实时调节**：最小粒子尺寸、模糊核大小、形态学迭代次数，以及各算法特有参数，修改即刷新预览
- **批量切割**：调好参数后一键批量处理整个文件夹，输出 `{原文件名}_{序号}.png`
- **设置自动保存**：关闭时自动保存所有参数到 `planktoscope.cfg`，下次打开自动恢复

## 检测参数说明

| 参数 | 说明 | 适用算法 |
|------|------|----------|
| Min particle size | 最小粒子直径(um)，小于此值的轮廓被过滤 | 全部 |
| Blur kernel | 高斯模糊核大小（奇数） | 全部 |
| Morphology close | 形态学闭操作迭代次数 | 全部 |
| Adaptive block size | 自适应阈值的邻域块大小（奇数） | Adaptive Threshold |
| Adaptive constant C | 自适应阈值的偏移常数 | Adaptive Threshold |
| Threshold value | 全局固定阈值 (0-255) | Global Threshold |
| Canny low/high | Canny 边缘检测的低/高阈值 | Canny Edge |

## 支持的图像格式

TIF, TIFF, JPG, JPEG, PNG, BMP, GIF, JFIF

## 项目文件

```
Planktoscope/
├── planktoscope_segment_viewer.py   # 主程序
├── imagecut.py                      # 独立切割函数（Microscope_Cut / Flowcam_Cut）
├── planktoscope.cfg                 # 自动生成的参数配置文件
├── requirements.txt                 # Python 依赖
├── run.bat                          # Windows 启动脚本
├── run.sh                           # macOS/Linux 启动脚本
└── README.md
```

## 系统要求

- Python 3.7+
- 支持 Windows / macOS / Linux
