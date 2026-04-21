# V-JEPA 2.1 Standalone Embedding Extractor

独立的 V-JEPA 2.1 视觉编码器，可在原仓库之外使用，支持加载官方预训练权重并提取图像/视频序列的 embedding。

## 特点

- **完全独立**：不依赖 `vjepa2` 原仓库，复制了原库的核心模型定义并修正了 import 路径
- **与官方一致**：模型结构、权重加载、预处理方式均与官方实现保持一致
- **支持多种输入**：`torch.Tensor`、`numpy.ndarray`、`PIL.Image` 列表均可
- **灵活聚合**：支持输出全部 patch tokens、逐时间步空间平均、或全局平均

## 安装依赖

```bash
pip install torch timm einops numpy pillow
```

- `torch` >= 2.0 (推荐)
- `timm`
- `einops`
- `numpy`
- `pillow` (如果用 PIL Image 输入)

## 文件结构

```
vjepa21/
├── extractor.py              # 高层 API：VJEPA21Encoder
├── models/
│   ├── vision_transformer.py # ViT Encoder 定义
│   └── utils/
│       ├── modules.py        # Transformer Block / RoPE Attention
│       ├── patch_embed.py    # Patch Embedding
│       └── pos_embs.py       # 位置编码
├── masks/
│   └── utils.py              # Mask 工具
└── utils/
    └── tensors.py            # Tensor 工具
```

## 下载预训练权重

| 模型 | 参数量 | 分辨率 | 下载地址 |
|------|--------|--------|----------|
| ViT-B/16 | 80M | 384 | [vjepa2_1_vitb_dist_vitG_384.pt](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt) |
| ViT-L/16 | 300M | 384 | [vjepa2_1_vitl_dist_vitG_384.pt](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt) |
| ViT-g/16 | 1B | 384 | [vjepa2_1_vitg_384.pt](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt) |
| ViT-G/16 | 2B | 384 | [vjepa2_1_vitG_384.pt](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt) |

```bash
# 示例：下载 ViT-L/16
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt
```

## 使用方法

### 基础用法

把 `vjepa21/` 文件夹放到你的项目里，然后：

```python
from vjepa21.extractor import VJEPA21Encoder
import torch

# 初始化编码器
encoder = VJEPA21Encoder(
    checkpoint_path="vjepa2_1_vitl_dist_vitG_384.pt",
    model_name="vit_large",   # 可选: vit_base / vit_large / vit_giant_xformers / vit_gigantic_xformers
    img_size=224,             # 你的输入分辨率
    num_frames=8,             # 输入帧数（必须是偶数）
    device="cuda",
)

# 输入：8 张 224×224 图像
images = torch.randn(8, 3, 224, 224)  # [T, 3, H, W]

# 提取全部 patch tokens
emb = encoder.extract(images, return_type="all_tokens")
print(emb.shape)  # [1, 784, 1024]
# 解释: 8帧 / tubelet_size(2) = 4 时间token, 空间 14×14=196, 总计 4×196=784

# 全局平均（单图/整clip的一个向量）
emb_global = encoder.extract(images, return_type="global_mean")
print(emb_global.shape)  # [1, 1024]

# 逐时间步空间平均
emb_temporal = encoder.extract(images, return_type="spatial_mean")
print(emb_temporal.shape)  # [1, 4, 1024]
```

### 2 张图像输入

如果你只有 2 帧（比如 stereo 双目或前后两帧）：

```python
encoder = VJEPA21Encoder(
    checkpoint_path="vjepa2_1_vitl_dist_vitG_384.pt",
    model_name="vit_large",
    img_size=224,
    num_frames=2,   # 2帧
)

images = torch.randn(2, 3, 224, 224)
emb = encoder.extract(images, return_type="all_tokens")
print(emb.shape)  # [1, 196, 1024]
# 2帧 / 2 = 1 时间token, 空间 14×14=196
```

### 使用 PIL Image 列表

```python
from PIL import Image

images = [Image.open(f"frame_{i:03d}.png") for i in range(8)]
emb = encoder.extract(images, return_type="all_tokens")
```

### 使用 NumPy 数组

```python
import numpy as np

# uint8, range [0, 255]
images = np.random.randint(0, 256, size=(8, 224, 224, 3), dtype=np.uint8)
emb = encoder.extract(images, return_type="global_mean")
```

### 批量处理 (Batch)

```python
# [B, T, 3, H, W]
batch_images = torch.randn(4, 8, 3, 224, 224)
emb = encoder.extract(batch_images, return_type="all_tokens")
print(emb.shape)  # [4, 784, 1024]
```

## 输入格式详细说明

`extractor.preprocess()` / `extractor.extract()` 支持以下输入格式：

| 格式 | Shape | 说明 |
|------|-------|------|
| torch.Tensor (视频格式) | `[B, 3, T, H, W]` | 最标准的视频张量 |
| torch.Tensor (序列格式) | `[B, T, 3, H, W]` | batch + 帧序列 |
| torch.Tensor (单样本) | `[T, 3, H, W]` | 单样本，无 batch 维度 |
| numpy.ndarray | `[T, H, W, 3]` 或 `[B, T, H, W, 3]` | uint8 [0,255] 或 float [0,1] |
| List[PIL.Image] | 长度=T | 会自动转为 tensor 并归一化 |

预处理流程（内部自动完成）：
1. 转换为 `torch.float32`
2. 若值为 uint8 范围，除以 255 缩放到 `[0, 1]`
3. 维度重排为 `[B, 3, T, H, W]`
4. ImageNet 归一化：`mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`

## `return_type` 说明

| 选项 | 输出形状 | 用途 |
|------|---------|------|
| `"all_tokens"` | `[B, N, D]` | 全部 patch tokens，保留时空结构。N=(T/2)×(H/16)×(W/16) |
| `"spatial_mean"` | `[B, T/2, D]` | 每个时间步内空间平均，保留时序信息 |
| `"global_mean"` | `[B, D]` | 全局平均池化，得到整段视频/图像的 single vector |

## 关于分辨率

V-JEPA 2.1 的预训练权重大多是 **384×384** 分辨率，但你可以直接输入 **224×224**：

- Encoder 使用 **RoPE (Rotary Position Embedding)**，天然支持不同空间分辨率
- 当 `img_size=224` 时，patch grid 为 `14×14`（而非预训练的 `24×24`）
- 模型内部会自动插值位置编码，无需额外处理

## 关于帧数

- `tubelet_size=2`，所以 **输入帧数必须是偶数**
- 常见选择：`2`（单图退化模式）、`8`（短视频 clip）
- 你不需要和预训练时的 64 帧保持一致，RoPE 支持变长时序输入

## 模型名对应表

| model_name 参数 | 模型 | embed_dim (D) | 官方文件名 |
|----------------|------|--------------|-----------|
| `"vit_base"` | ViT-B/16 | 768 | `vjepa2_1_vitb_dist_vitG_384.pt` |
| `"vit_large"` | ViT-L/16 | 1024 | `vjepa2_1_vitl_dist_vitG_384.pt` |
| `"vit_giant_xformers"` | ViT-g/16 | 1408 | `vjepa2_1_vitg_384.pt` |
| `"vit_gigantic_xformers"` | ViT-G/16 | 1664 | `vjepa2_1_vitG_384.pt` |

## 常见问题

**Q: 加载权重时提示 missing keys?**  
A: 正常。预训练权重可能包含 `pos_embed`（sincos 位置编码），但 V-JEPA 2.1 使用 RoPE 不需要它，这些 key 会被自动跳过。

**Q: 可以用 CPU 跑吗?**  
A: 可以，`device="cpu"` 即可，但大模型（ViT-G/16, 2B 参数）在 CPU 上会很慢。

**Q: 需要整个 vjepa2 仓库吗?**  
A: **不需要**。只要把这个 `vjepa21/` 文件夹复制到你的项目里即可独立运行。

**Q: 支持混合精度推理吗?**  
A: 当前实现默认使用 `torch.cuda.amp.autocast(enabled=False)` 保持 fp32。如果你需要 fp16，可以自行在外部包裹 `torch.autocast`。
