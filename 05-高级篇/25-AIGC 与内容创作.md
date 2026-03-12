# 第二十五章：AIGC 与内容创作

> 掌握 AI 绘画、数字人、虚拟人等 AIGC 技术

---

## 25.1 AIGC 概述

### 25.1.1 什么是 AIGC

**AIGC**（AI Generated Content）人工智能生成内容。

**应用：**
- AI 绘画
- 数字人
- 音乐生成
- 视频生成

### 25.1.2 技术栈

| 技术 | 应用 |
|------|------|
| **Diffusion** | 图像生成 |
| **GAN** | 人脸生成 |
| **NeRF** | 3D 内容 |
| **Transformer** | 文本生成 |

---

## 25.2 AI 绘画

### 25.2.1 Stable Diffusion WebUI

```bash
# 安装
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
cd stable-diffusion-webui
./webui.sh

# 访问 http://localhost:7860
```

### 25.2.2 提示词工程

```
正面提示词：
masterpiece, best quality, high quality, beautiful, detailed

负面提示词：
low quality, worst quality, ugly, blurry, deformed
```

### 25.2.3 ControlNet 应用

```python
from diffusers import StableDiffusionControlNetPipeline

# 加载
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet="lllyasviel/control_v11p_sd15_canny"
)

# 生成
image = pipe(
    prompt="a beautiful landscape",
    image=canny_image,
    num_inference_steps=50
).images[0]
```

---

## 25.3 数字人

### 25.3.1 人脸生成

```python
from stylegan2_pytorch import StyleGAN2

# 加载模型
model = StyleGAN2(image_size=1024)

# 生成人脸
latent = torch.randn(1, 512)
face = model(latent)
```

### 25.3.2 语音合成

```python
from TTS.api import TTS

# 加载模型
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# 合成语音
tts.tts_to_file(text="Hello world", file_path="output.wav")
```

### 25.3.3 唇形同步

```python
# 使用 Wav2Lip
from wav2lip import Wav2Lip

model = Wav2Lip()

# 同步唇形
video = model.sync(video_path, audio_path)
```

---

## 25.4 虚拟人

### 25.4.1 3D 虚拟人

```python
import pyrender

# 创建场景
scene = pyrender.Scene()

# 添加虚拟人模型
mesh = pyrender.Mesh.from_trimesh(human_model)
scene.add(mesh)

# 渲染
renderer = pyrender.OffscreenRenderer(1920, 1080)
color, depth = renderer.render(scene)
```

### 25.4.2 动作捕捉

```python
import mediapipe as mp

# 加载姿态检测
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 检测姿态
results = pose.process(image)
pose_landmarks = results.pose_landmarks
```

---

## 25.5 实战：AI 绘画工具

### 25.5.1 Gradio 界面

```python
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

def generate(prompt, negative_prompt, steps):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps
    ).images[0]
    return image

# 创建界面
iface = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Textbox(label="Negative Prompt"),
        gr.Slider(1, 100, 50, label="Steps")
    ],
    outputs=gr.Image(),
    title="AI 绘画工具"
)

iface.launch()
```

---

## 25.6 本章小结

### 核心知识点

1. **AI 绘画** - Stable Diffusion
2. **数字人** - 人脸 + 语音
3. **虚拟人** - 3D+ 动作捕捉
4. **AIGC 应用** - 内容创作

### 练习题目

1. 搭建 AI 绘画工具
2. 创建数字人
3. 制作虚拟主播

### 下章预告

下一章我们将学习**边缘计算与部署**，掌握模型优化与部署技术。

---

## 参考资料

1. [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
2. [StyleGAN](https://github.com/NVlabs/stylegan2)
3. [MediaPipe](https://mediapipe.dev/)
4. [Gradio](https://gradio.app/)

---

**更新日期：** 2026-03-12  
**作者：** OpenClaw
