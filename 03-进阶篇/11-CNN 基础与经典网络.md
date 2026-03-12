# 第十一章：CNN 基础与经典网络

> 掌握卷积神经网络原理，学习 LeNet、AlexNet、VGG、ResNet 等经典网络架构

---

## 11.1 卷积神经网络基础

### 11.1.1 为什么需要 CNN

**传统全连接网络的问题：**
- 参数过多（过拟合）
- 忽略空间结构
- 无法处理大图像

**CNN 的优势：**
- ✅ 局部连接（减少参数）
- ✅ 权值共享（平移不变性）
- ✅ 池化操作（降维）

### 11.1.2 CNN 核心组件

| 组件 | 作用 | 特点 |
|------|------|------|
| **卷积层** | 特征提取 | 局部连接、权值共享 |
| **池化层** | 降维 | 最大池化、平均池化 |
| **激活函数** | 非线性 | ReLU、Sigmoid |
| **全连接层** | 分类 | 输出预测结果 |

---

## 11.2 卷积层

### 11.2.1 卷积操作

```python
import torch
import torch.nn as nn

# 2D 卷积
conv2d = nn.Conv2d(
    in_channels=3,      # 输入通道数（RGB）
    out_channels=64,    # 输出通道数（滤波器数量）
    kernel_size=3,      # 卷积核大小
    stride=1,           # 步长
    padding=1,          # 填充
    bias=True           # 偏置
)

# 输入：[batch_size, channels, height, width]
input_tensor = torch.randn(1, 3, 224, 224)
output = conv2d(input_tensor)

print(f"输入形状：{input_tensor.shape}")
print(f"输出形状：{output.shape}")
```

### 11.2.2 卷积参数计算

```
输出尺寸 = (输入尺寸 - 卷积核大小 + 2×填充) / 步长 + 1

参数量 = (输入通道 × 输出通道 × 卷积核大小²) + 输出通道（偏置）

计算量 = 输出高度 × 输出宽度 × 输出通道 × 输入通道 × 卷积核大小²
```

### 11.2.3 代码示例

```python
import torch
import torch.nn as nn

class SimpleConv(nn.Module):
    """简单卷积层示例"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv1(x)      # [1, 3, 224, 224] → [1, 32, 224, 224]
        x = self.relu(x)       # ReLU 激活
        x = self.pool(x)       # [1, 32, 224, 224] → [1, 32, 112, 112]
        return x

# 使用
model = SimpleConv()
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)

print(f"输出形状：{output.shape}")
```

---

## 11.3 池化层

### 11.3.1 最大池化

```python
import torch.nn as nn

# 最大池化
max_pool = nn.MaxPool2d(
    kernel_size=2,    # 池化核大小
    stride=2,         # 步长
    padding=0         # 填充
)

input_tensor = torch.randn(1, 64, 112, 112)
output = max_pool(input_tensor)

print(f"输入：{input_tensor.shape}")
print(f"输出：{output.shape}")  # [1, 64, 56, 56]
```

### 11.3.2 平均池化

```python
import torch.nn as nn

# 平均池化
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

input_tensor = torch.randn(1, 64, 112, 112)
output = avg_pool(input_tensor)

print(f"输出：{output.shape}")
```

### 11.3.3 全局池化

```python
import torch.nn as nn

# 全局平均池化
global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

input_tensor = torch.randn(1, 512, 7, 7)
output = global_avg_pool(input_tensor)

print(f"输入：{input_tensor.shape}")
print(f"输出：{output.shape}")  # [1, 512, 1, 1]
```

---

## 11.4 经典网络架构

### 11.4.1 LeNet-5（1998）

**第一个成功的 CNN，用于手写数字识别。**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """LeNet-5 实现"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, 5)    # 28x28 → 24x24
        self.conv2 = nn.Conv2d(6, 16, 5)   # 10x10 → 6x6
        
        # 池化层
        self.pool = nn.AvgPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 → 12x12
        x = self.pool(F.relu(self.conv2(x)))  # 12x12 → 2x2
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 使用
model = LeNet5()
input_tensor = torch.randn(1, 1, 28, 28)  # MNIST 图像
output = model(input_tensor)
print(f"输出：{output.shape}")
```

### 11.4.2 AlexNet（2012）

**深度学习元年，ImageNet 竞赛冠军。**

```python
import torch.nn as nn

class AlexNet(nn.Module):
    """AlexNet 实现"""
    
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # 特征提取
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, 4, 2),   # 224x224 → 55x55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),           # 55x55 → 27x27
            
            nn.Conv2d(64, 192, 5, 1, 2),  # 27x27 → 27x27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),           # 27x27 → 13x13
            
            nn.Conv2d(192, 384, 3, 1, 1), # 13x13 → 13x13
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, 3, 1, 1), # 13x13 → 13x13
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, 3, 1, 1), # 13x13 → 13x13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),           # 13x13 → 6x6
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

### 11.4.3 VGG（2014）

**使用小卷积核，网络更深。**

```python
import torch.nn as nn

class VGG16(nn.Module):
    """VGG-16 实现"""
    
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

### 11.4.4 ResNet（2015）

**残差连接，解决梯度消失。**

```python
import torch.nn as nn

class BasicBlock(nn.Module):
    """ResNet 基本块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = nn.ReLU(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = nn.ReLU(out)
        return out

class ResNet18(nn.Module):
    """ResNet-18 实现"""
    
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = nn.ReLU(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

---

## 11.5 网络对比

| 网络 | 年份 | 层数 | Top-5 错误率 | 特点 |
|------|------|------|------------|------|
| **LeNet-5** | 1998 | 7 | - | 第一个 CNN |
| **AlexNet** | 2012 | 8 | 15.3% | ReLU、Dropout |
| **VGG-16** | 2014 | 16 | 7.3% | 小卷积核 |
| **ResNet-18** | 2015 | 18 | 5.1% | 残差连接 |

---

## 11.6 实战：图像分类

### 11.6.1 使用预训练模型

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 加载预训练模型
model = models.resnet18(pretrained=True)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# 加载图像
img = Image.open('image.jpg')
img_tensor = transform(img).unsqueeze(0)

# 预测
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)

print(f"预测类别：{predicted.item()}")
```

---

## 11.7 本章小结

### 核心知识点

1. **卷积层** — 特征提取
2. **池化层** — 降维
3. **经典网络** — LeNet、AlexNet、VGG、ResNet
4. **残差连接** — 解决梯度消失
5. **预训练模型** — 迁移学习

### 练习题目

1. 实现 LeNet-5 并训练 MNIST
2. 使用预训练 ResNet 进行图像分类
3. 对比不同网络的性能

### 下章预告

下一章我们将学习**目标检测**，掌握 R-CNN、YOLO、SSD 等核心算法。

---

## 参考资料

1. [PyTorch 官方文档](https://pytorch.org/docs/)
2. [CNN 可视化](https://poloclub.github.io/cnn-explainer/)
3. [Papers With Code](https://paperswithcode.com/)

---

**更新日期：** 2026-03-12  
**作者：** OpenClaw
