# 第二十一章：Diffusion Model 详解

> 理解扩散模型为什么成为当前视觉生成领域的重要主线

---

## 21.1 Diffusion Model 基础

### 21.1.1 原理

**扩散模型**通过逐步去噪生成图像。

**两个过程：**
- **前向过程** - 逐步添加噪声
- **反向过程** - 逐步去除噪声

### 21.1.2 数学原理

```
前向过程：q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_tI)

反向过程：p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

---

## 21.2 DDPM

### 21.2.1 原理

**DDPM**（Denoising Diffusion Probabilistic Models）是基础扩散模型。

### 21.2.2 实现

```python
import torch
import torch.nn as nn

class UNet1D(nn.Module):
    """简化 UNet 用于扩散"""
    
    def __init__(self, time_dim=128):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.down = nn.Sequential(
            nn.Conv1d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        
        self.up = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 3, 3, padding=1)
        )
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t.unsqueeze(-1).float())
        x = self.down(x)
        x = x + t_emb.unsqueeze(-1)
        x = self.up(x)
        return x
```

---

## 21.3 Stable Diffusion

### 21.3.1 原理

**Stable Diffusion**在潜空间进行扩散，效率更高。

**组件：**
- VAE（变分自编码器）
- U-Net（去噪网络）
- CLIP（文本编码器）

### 21.3.2 使用

```python
from diffusers import StableDiffusionPipeline
import torch

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 文生图
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
image.save("astronaut_rides_horse.png")
```

---

## 21.4 ControlNet

### 21.4.1 原理

**ControlNet**添加空间条件控制，实现精确生成。

**控制类型：**
- 边缘图
- 深度图
- 姿态图
- 分割图

### 21.4.2 使用

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import torch

# 加载 ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16
)

# 加载管道
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 加载控制图
control_image = Image.open("canny_edge.png")

# 生成
prompt = "a beautiful sunset over mountains"
image = pipe(prompt, image=control_image).images[0]
image.save("controlled_generation.png")
```

---

## 21.5 LoRA 微调

### 21.5.1 原理

**LoRA**（Low-Rank Adaptation）轻量级微调技术。

**优势：**
- 参数量小（几 MB）
- 训练快
- 易于分享

### 21.5.2 训练

```bash
# 使用 Kohya's Trainer
python train_network.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --train_data_dir ./training_data \
  --output_dir ./lora_output \
  --resolution 512 \
  --batch_size 1 \
  --epochs 10
```

### 21.5.3 使用

```python
from diffusers import StableDiffusionPipeline
from peft import PeftModel

# 加载基础模型
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)

# 加载 LoRA
pipe.unet = PeftModel.from_pretrained(
    pipe.unet,
    "./lora_output",
    subfolder="unet"
)

# 使用
image = pipe("your prompt").images[0]
```

---

## 21.6 本章小结

### 核心知识点

1. **Diffusion 原理** - 前向/反向过程
2. **DDPM** - 基础扩散模型
3. **Stable Diffusion** - 潜空间扩散
4. **ControlNet** - 空间条件控制
5. **LoRA** - 轻量微调

### 练习题目

1. 实现简单扩散模型
2. 使用 Stable Diffusion 生成图像
3. 训练自定义 LoRA

### 下章预告

下一章我们将学习**视频生成与处理**，掌握视频理解与生成技术。

---

## 参考资料

1. [DDPM 论文](https://arxiv.org/abs/2006.11239)
2. [Stable Diffusion 论文](https://arxiv.org/abs/2112.10752)
3. [ControlNet 论文](https://arxiv.org/abs/2302.05543)
4. [Diffusers 库](https://huggingface.co/docs/diffusers)

---

**更新日期：** 2026-03-12  
**作者：** OpenClaw
