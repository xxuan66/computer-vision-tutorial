# 🖼️ 计算机视觉从入门到实战

> 系统化的计算机视觉学习教程，涵盖图像处理、目标检测、图像生成、视频理解等核心领域

[![Stars](https://img.shields.io/github/stars/xxuan66/computer-vision-tutorial)](https://github.com/xxuan66/computer-vision-tutorial/stargazers)
[![Issues](https://img.shields.io/github/issues/xxuan66/computer-vision-tutorial)](https://github.com/xxuan66/computer-vision-tutorial/issues)
[![License](https://img.shields.io/github/license/xxuan66/computer-vision-tutorial)](https://github.com/xxuan66/computer-vision-tutorial/blob/main/LICENSE)

---

## 📚 学习路线

```
入门篇 → 基础篇 → 进阶篇 → 实战篇 → 高级篇
```

### 📖 第一阶段：入门篇（2-3 周）

**目标：** 建立计算机视觉整体认知

| 章节 | 内容 | 预计时间 |
|------|------|---------|
| 01 | 什么是计算机视觉 | 2h |
| 02 | 应用场景与发展历程 | 3h |
| 03 | 开发环境搭建 | 3h |
| 04 | Python+OpenCV 基础 | 6h |

### 📖 第二阶段：基础篇（4-6 周）

**目标：** 掌握图像处理基础

| 章节 | 内容 | 预计时间 |
|------|------|---------|
| 05 | 图像基础（颜色空间、直方图） | 4h |
| 06 | 图像滤波与增强 | 6h |
| 07 | 边缘检测与特征提取 | 6h |
| 08 | 图像变换（傅里叶、小波） | 6h |
| 09 | 形态学操作 | 4h |
| 10 | 图像分割基础 | 6h |

### 📖 第三阶段：进阶篇（6-8 周）

**目标：** 深度学习与 CV 结合

| 章节 | 内容 | 预计时间 |
|------|------|---------|
| 11 | CNN 基础与经典网络 | 8h |
| 12 | 目标检测（R-CNN, YOLO, SSD） | 10h |
| 13 | 图像分割（FCN, U-Net, Mask R-CNN） | 8h |
| 14 | 图像分类实战 | 6h |
| 15 | 人脸识别与检测 | 6h |

### 📖 第四阶段：实战篇（4-6 周）

**目标：** 完整项目实战

| 项目 | 技术栈 | 难度 |
|------|--------|------|
| 图像分类系统 | PyTorch + ResNet | ⭐⭐ |
| 目标检测应用 | YOLOv5 | ⭐⭐⭐ |
| 图像分割工具 | U-Net | ⭐⭐⭐ |
| 人脸识别门禁 | FaceNet + OpenCV | ⭐⭐⭐⭐ |

### 📖 第五阶段：高级篇（持续学习）

**目标：** 前沿技术探索

| 章节 | 内容 | 热点 |
|------|------|------|
| 16 | 图像生成基础（GAN） | 🔥 |
| 17 | Diffusion Model 详解 | 🔥🔥 |
| 18 | 视频生成与处理 | 🔥🔥 |
| 19 | 3D 视觉（NeRF、高斯泼溅） | 🔥 |
| 20 | 多模态大模型（CLIP、DALL-E） | 🔥🔥🔥 |
| 21 | AIGC 与内容创作 | 🔥🔥🔥 |
| 22 | 边缘计算与部署优化 | 🔥 |
| 23 | 前沿技术展望 | 🔥 |

---

## 📁 目录结构

```
computer-vision-tutorial/
├── README.md                      # 项目说明
├── 01-入门篇/
│   ├── 01-什么是计算机视觉.md
│   ├── 02-应用场景与发展历程.md
│   ├── 03-开发环境搭建.md
│   └── 04-Python+OpenCV基础.md
├── 02-基础篇/
│   ├── 05-图像基础.md
│   ├── 06-图像滤波与增强.md
│   ├── 07-边缘检测与特征提取.md
│   ├── 08-图像变换.md
│   ├── 09-形态学操作.md
│   └── 10-图像分割基础.md
├── 03-进阶篇/
│   ├── 11-CNN与经典网络.md
│   ├── 12-目标检测.md
│   ├── 13-图像分割.md
│   ├── 14-图像分类实战.md
│   └── 15-人脸识别.md
├── 04-实战篇/
│   ├── 图像分类系统/
│   ├── 目标检测应用/
│   ├── 图像分割工具/
│   └── 人脸识别门禁/
├── 05-高级篇/
│   ├── 16-图像生成基础.md
│   ├── 17-DiffusionModel详解.md
│   ├── 18-视频生成与处理.md
│   ├── 19-3D视觉.md
│   ├── 20-多模态大模型.md
│   ├── 21-AIGC与内容创作.md
│   ├── 22-边缘计算与部署.md
│   └── 23-前沿技术展望.md
├── code/                          # 代码示例
│   ├── opencv/
│   ├── pytorch/
│   └── projects/
└── resources/                     # 学习资源
    └── 学习资源汇总.md
```

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/xxuan66/computer-vision-tutorial.git
cd computer-vision-tutorial
```

### 2. 安装依赖

```bash
# 基础依赖
pip install opencv-python numpy matplotlib pillow

# 深度学习
pip install torch torchvision

# 项目依赖
pip install -r requirements.txt
```

### 3. 开始学习

从 [01-入门篇](./01-入门篇/) 开始，按顺序学习。

---

## 📊 核心知识点

### 图像处理

| 技术 | 说明 | 应用场景 |
|------|------|---------|
| **滤波** | 去噪、平滑 | 图像预处理 |
| **边缘检测** | 提取轮廓 | 特征提取 |
| **形态学** | 膨胀、腐蚀 | 图像后处理 |
| **分割** | 前景/背景分离 | 目标提取 |

### 深度学习

| 技术 | 代表模型 | 应用场景 |
|------|---------|---------|
| **图像分类** | ResNet, EfficientNet | 图像识别 |
| **目标检测** | YOLO, Faster R-CNN | 物体定位 |
| **图像分割** | U-Net, Mask R-CNN | 像素级分类 |
| **图像生成** | GAN, Diffusion | 图像合成 |

### 前沿技术

| 技术 | 代表作品 | 应用场景 |
|------|---------|---------|
| **Diffusion** | Stable Diffusion, DALL-E 3 | 文生图 |
| **视频生成** | Sora, Video Diffusion | 文生视频 |
| **3D 视觉** | NeRF, 3D Gaussian Splatting | 3D 重建 |
| **多模态** | CLIP, GPT-4V | 图文理解 |

---

## 🎓 学习资源

### 推荐书籍

- 《数字图像处理》- Gonzalez
- 《计算机视觉：算法与应用》- Szeliski
- 《深度学习》- Goodfellow

### 推荐课程

- Stanford CS231n（计算机视觉）
- Coursera Deep Learning Specialization
- OpenCV 官方教程

### 工具框架

| 工具 | 用途 | 链接 |
|------|------|------|
| OpenCV | 图像处理 | [官网](https://opencv.org/) |
| PyTorch | 深度学习 | [官网](https://pytorch.org/) |
| TensorFlow | 深度学习 | [官网](https://tensorflow.org/) |
| Albumentations | 数据增强 | [GitHub](https://github.com/albumentations-team) |
| Diffusers | 扩散模型 | [Hugging Face](https://huggingface.co/docs/diffusers) |

### 公开数据集

| 数据集 | 规模 | 类型 | 链接 |
|--------|------|------|------|
| ImageNet | 1400 万 | 图像分类 | [链接](https://www.image-net.org/) |
| COCO | 33 万 | 目标检测 | [链接](https://cocodataset.org/) |
| Pascal VOC | 2 万 | 目标检测 | [链接](http://host.robots.ox.ac.uk/pascal/VOC/) |
| CelebA | 20 万 | 人脸属性 | [链接](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| LAION | 50 亿 + | 图文对 | [链接](https://laion.ai/) |

---

## 📅 学习计划

### 3 个月速成计划

| 周数 | 内容 | 产出 |
|------|------|------|
| 1-2 | 入门篇 | 环境搭建、OpenCV 基础 |
| 3-6 | 基础篇 | 图像处理算法实现 |
| 7-10 | 进阶篇 | 深度学习模型 |
| 11-14 | 实战篇 | 2-3 个完整项目 |
| 15-16 | 复习提升 | 简历准备、面试 |

### 6 个月系统学习

| 阶段 | 时间 | 目标 |
|------|------|------|
| 基础 | 1-2 月 | 图像处理基础 |
| 进阶 | 3-4 月 | 深度学习 CV |
| 实战 | 5 月 | 完整项目 |
| 前沿 | 6 月 | GAN/Diffusion/多模态 |

---

## 🤝 贡献指南

欢迎贡献内容！

### 贡献类型

- ✅ 修正错别字和错误
- ✅ 补充代码示例
- ✅ 添加新的算法教程
- ✅ 分享实战项目
- ✅ 翻译英文资料

### 提交流程

1. Fork 本仓库
2. 创建分支 `git checkout -b feature/your-feature`
3. 提交更改 `git commit -m 'Add some feature'`
4. 推送到分支 `git push origin feature/your-feature`
5. 提交 Pull Request

---

## 📝 更新日志

### 2026-03-12
- ✅ 创建仓库
- ✅ 完成整体框架设计（27 章）
- ✅ 添加入门篇内容
- ✅ 扩展高级篇（图像生成、视频生成、多模态）

### TODO
- [ ] 完成入门篇详细教程
- [ ] 添加 OpenCV 代码示例
- [ ] 创建实战项目
- [ ] 搭建项目 1 框架

---

## 📧 联系方式

- **GitHub Issues:** [提问](https://github.com/xxuan66/computer-vision-tutorial/issues)

---

## 📄 许可证

MIT License

---

**⭐ 如果这个项目对你有帮助，请给一个 Star！**

**📢 欢迎分享给更多需要的朋友！**
