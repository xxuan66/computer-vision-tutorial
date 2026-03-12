# 第四章：Python+OpenCV 基础

> 掌握 OpenCV 的基本操作，开始图像处理之旅

---

## 4.1 OpenCV 简介

### 4.1.1 什么是 OpenCV

**OpenCV**（Open Source Computer Vision Library）是一个开源的计算机视觉库。

**特点：**
- ✅ 跨平台（Windows、Linux、macOS）
- ✅ 多语言支持（Python、C++、Java）
- ✅ 2500+ 个算法
- ✅ 高性能（C/C++ 实现）
- ✅ 活跃的社区支持

### 4.1.2 应用领域

| 领域 | 应用 |
|------|------|
| **图像处理** | 滤波、增强、变换 |
| **目标检测** | 人脸、车辆、行人 |
| **图像分割** | 前景/背景分离 |
| **3D 重建** | 立体视觉、SLAM |
| **机器学习** | 分类、聚类 |

---

## 4.2 图像读取与显示

### 4.2.1 读取图像

```python
import cv2

# 读取图像
img = cv2.imread('image.jpg')

# 读取为灰度图
img_gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 读取为彩色图（默认）
img_color = cv2.imread('image.jpg', cv2.IMREAD_COLOR)

# 读取为包含 alpha 通道
img_alpha = cv2.imread('image.png', cv2.IMREAD_UNCHANGED)

# 检查是否成功
if img is None:
    print("图像读取失败！")
else:
    print(f"图像形状：{img.shape}")
```

### 4.2.2 显示图像

```python
import cv2

# 读取图像
img = cv2.imread('image.jpg')

# 创建窗口
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

# 显示图像
cv2.imshow('Image', img)

# 等待按键（0 表示无限等待）
cv2.waitKey(0)

# 关闭所有窗口
cv2.destroyAllWindows()
```

### 4.2.3 保存图像

```python
import cv2

# 读取图像
img = cv2.imread('image.jpg')

# 保存图像
cv2.imwrite('output.jpg', img)

# 保存为 PNG（无损压缩）
cv2.imwrite('output.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

# 保存为 JPEG（有损压缩）
cv2.imwrite('output.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
```

---

## 4.3 图像基本属性

### 4.3.1 获取图像信息

```python
import cv2

img = cv2.imread('image.jpg')

# 图像形状（高度，宽度，通道数）
height, width, channels = img.shape
print(f"高度：{height}")
print(f"宽度：{width}")
print(f"通道数：{channels}")

# 数据类型
print(f"数据类型：{img.dtype}")

# 像素总数
print(f"像素总数：{img.size}")
```

### 4.3.2 访问像素

```python
import cv2

img = cv2.imread('image.jpg')

# 访问单个像素（BGR 格式）
pixel = img[100, 100]
print(f"像素值：{pixel}")

# 访问单个通道
blue = img[100, 100, 0]   # 蓝色通道
green = img[100, 100, 1]  # 绿色通道
red = img[100, 100, 2]    # 红色通道

# 修改像素值
img[100, 100] = [0, 0, 0]  # 设置为黑色
img[100, 100] = [255, 255, 255]  # 设置为白色

# 批量修改
img[0:100, 0:100] = [0, 0, 255]  # 左上角设置为红色
```

---

## 4.4 颜色空间转换

### 4.4.1 常见颜色空间

| 颜色空间 | 说明 | 应用 |
|---------|------|------|
| **BGR** | OpenCV 默认格式 | 图像显示 |
| **RGB** | 标准格式 | matplotlib 显示 |
| **GRAY** | 灰度图 | 边缘检测 |
| **HSV** | 色调、饱和度、明度 | 颜色分割 |
| **LAB** | 亮度、颜色通道 | 颜色匹配 |

### 4.4.2 转换方法

```python
import cv2

img = cv2.imread('image.jpg')

# BGR → RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# BGR → GRAY
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# BGR → HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# BGR → LAB
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# RGB → GRAY
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
```

### 4.4.3 实战：颜色分割

```python
import cv2
import numpy as np

# 读取图像
img = cv2.imread('image.jpg')

# 转换为 HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义蓝色范围
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

# 创建掩码
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 应用掩码
result = cv2.bitwise_and(img, img, mask=mask)

# 显示
cv2.imshow('Original', img)
cv2.imshow('Mask', mask)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 4.5 图像几何变换

### 4.5.1 缩放

```python
import cv2

img = cv2.imread('image.jpg')

# 方法 1：指定缩放比例
resized1 = cv2.resize(img, None, fx=0.5, fy=0.5)

# 方法 2：指定目标尺寸
resized2 = cv2.resize(img, (256, 256))

# 插值方法
resized_nearest = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
resized_linear = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
resized_cubic = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
```

### 4.5.2 平移

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')
height, width = img.shape[:2]

# 平移矩阵（向右 50 像素，向下 30 像素）
M = np.float32([[1, 0, 50],
                [0, 1, 30]])

# 应用平移
translated = cv2.warpAffine(img, M, (width, height))
```

### 4.5.3 旋转

```python
import cv2

img = cv2.imread('image.jpg')
height, width = img.shape[:2]

# 旋转中心
center = (width // 2, height // 2)

# 旋转矩阵（逆时针 45 度，缩放 1.0）
M = cv2.getRotationMatrix2D(center, 45, 1.0)

# 应用旋转
rotated = cv2.warpAffine(img, M, (width, height))
```

### 4.5.4 翻转

```python
import cv2

img = cv2.imread('image.jpg')

# 水平翻转
flipped_h = cv2.flip(img, 1)

# 垂直翻转
flipped_v = cv2.flip(img, 0)

# 水平和垂直翻转
flipped_both = cv2.flip(img, -1)
```

---

## 4.6 图像绘图

### 4.6.1 绘制几何图形

```python
import cv2
import numpy as np

# 创建黑色图像
img = np.zeros((512, 512, 3), dtype=np.uint8)

# 绘制直线
cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 2)

# 绘制矩形
cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)

# 绘制圆形
cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)

# 绘制椭圆
cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, (255, 255, 0), 2)

# 绘制多边形
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(img, [pts], True, (0, 255, 255), 3)

# 显示
cv2.imshow('Drawing', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 4.6.2 添加文字

```python
import cv2
import numpy as np

img = np.zeros((512, 512, 3), dtype=np.uint8)

# 添加文字
cv2.putText(img, 'Hello OpenCV', 
            (50, 250), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2)

# 显示
cv2.imshow('Text', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 4.7 鼠标事件

### 4.7.1 鼠标回调函数

```python
import cv2

# 鼠标回调函数
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 50, (0, 255, 0), -1)

# 创建图像
img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')

# 绑定回调函数
cv2.setMouseCallback('image', draw_circle)

while True:
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

### 4.7.2 鼠标事件类型

| 事件 | 说明 |
|------|------|
| `EVENT_MOUSEMOVE` | 鼠标移动 |
| `EVENT_LBUTTONDOWN` | 左键按下 |
| `EVENT_LBUTTONUP` | 左键释放 |
| `EVENT_RBUTTONDOWN` | 右键按下 |
| `EVENT_LBUTTONDBLCLK` | 左键双击 |

---

## 4.8 实战项目

### 4.8.1 图像浏览器

```python
import cv2
import os

class ImageViewer:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.images = [f for f in os.listdir(folder_path) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.index = 0
        
    def show_image(self):
        img_path = os.path.join(self.folder_path, self.images[self.index])
        img = cv2.imread(img_path)
        
        cv2.putText(img, f'{self.index+1}/{len(self.images)}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Image Viewer', img)
        
    def next_image(self):
        self.index = (self.index + 1) % len(self.images)
        self.show_image()
        
    def prev_image(self):
        self.index = (self.index - 1) % len(self.images)
        self.show_image()

# 使用
viewer = ImageViewer('./images')
viewer.show_image()

while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord('n'):
        viewer.next_image()
    elif key == ord('p'):
        viewer.prev_image()
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
```

---

## 4.9 本章小结

### 核心知识点

1. **图像读取与显示** — `imread`, `imshow`, `imwrite`
2. **图像属性** — 形状、数据类型、像素访问
3. **颜色空间** — BGR、RGB、GRAY、HSV
4. **几何变换** — 缩放、平移、旋转、翻转
5. **绘图功能** — 直线、矩形、圆形、文字
6. **鼠标事件** — 回调函数

### 练习题目

1. 编写程序批量转换图像格式（JPG → PNG）
2. 实现图像水印添加功能
3. 创建简单的图像编辑器（裁剪、旋转、缩放）

### 下章预告

下一章我们将进入**基础篇**，学习图像滤波、边缘检测等核心算法。

---

## 参考资料

1. [OpenCV 官方文档](https://docs.opencv.org/)
2. [OpenCV-Python 教程](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
3. [PyImageSearch](https://www.pyimagesearch.com/)

---

**更新日期：** 2026-03-12  
**作者：** OpenClaw
