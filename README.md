# AI Training Data Annotation (geoskill-ai-training-data-annotation)

> Small U-Net binary semantic segmentation (torch+CUDA) pre-labeling + active-learning uncertainty sampling, outputting COCO/GeoJSON annotations and an uncertainty raster

---

## 1. Overview

Automatically generates pre-labels (pseudo-labels) for remote sensing imagery and uses active learning to select the most uncertain samples for human review, outputting standard COCO JSON and GeoJSON annotation formats plus an uncertainty raster. **Core model**: a small U-Net (`unet-lite`, base=8, 3-level encoder-decoder with skip connections), performing target-vs-background binary semantic segmentation on torch + CUDA by default. The model infers on single-band imagery to produce an (H, W, 2) softmax probability map; the target probability is Otsu-thresholded and processed with 8-neighbor connected components to generate bbox pre-labels. Uncertainty is the Shannon entropy of the probability map, and Top-k samples by regional mean entropy are selected for human review. Pre-trained weights `anno_unet_weights.pt` are bundled (if missing, they are automatically trained on the GPU at first run and saved to disk). A classical `--method otsu` threshold baseline is also kept for comparison and GPU-free environments.

## 2. Features

Automatically generates pre-labels (pseudo-labels) for remote sensing imagery and uses active learning to select the most uncertain samples for human review, outputting standard COCO JSON and GeoJSON annotation formats plus an uncertainty raster. **Core model**: a small U-Net (`unet-lite`, base=8, 3-level encoder-decoder with skip connections), performing target-vs-background binary semantic segmentation on torch + CUDA by default. The model infers on single-band imagery to produce an (H, W, 2) softmax probability map; the target probability is Otsu-thresholded and processed with 8-neighbor connected components to generate bbox pre-labels. Uncertainty is the Shannon entropy of the probability map, and Top-k samples by regional mean entropy are selected for human review. Pre-trained weights `anno_unet_weights.pt` are bundled (if missing, they are automatically trained on the GPU at first run and saved to disk). A classical `--method otsu` threshold baseline is also kept for comparison and GPU-free environments.

## 3. Quick Start

```bash
pip install -r requirements.txt
python geoskill-ai-training-data-annotation.py --bbox 116 39 117 40 --synthetic --output-dir ./out
```

## 4. CLI Parameters

Run `python <skill>.py --help` for the full list. Common parameters:

| Parameter | Type | Description |
|---|---|---|
| `--bbox` | `float[4]` | WGS84 bounding box `min_lon min_lat max_lon max_lat` |
| `--input` | `path` | Local input file (GeoJSON/GeoTIFF/etc.) |
| `--output-dir` | `path` | Output directory (default `./output`) |
| `--synthetic` | `flag` | Use synthetic data instead of real input |
| `--quiet` | `flag` | Suppress non-essential stdout |

## 5. Input / Output

| File | Format | Description |
|---|---|---|
| `annotations_coco.json` | JSON | Standard COCO annotations (images/annotations/categories) |
| `annotations.geojson` | GeoJSON | Georeferenced pre-label boxes + confidence/uncertainty/review flags |
| `uncertainty.tif` | GeoTIFF | Per-pixel entropy (uncertainty) raster |
| `output-manifest.json` | JSON | Run manifest (input/output/QA/exit code) |


## 6. Technical Principle

(see SKILL.md for details)

## 7. Methodology

This skill has been methodologically reviewed. See [`REVIEW.md`](./REVIEW.md) for:

- P0/P1/P2 issue counts and verdicts
- Reproduction commands
- Known limitations and edge cases

## 8. License

MIT License. See [`LICENSE`](./LICENSE) for full text.

---

# AI训练数据标注（中文版）

> 本部分为完整中文文档，与上方英文部分对应。

---
name: geoskill-ai-training-data-annotation
description: '小型 U-Net 二值语义分割 (torch+CUDA) 预标注 + 主动学习不确定性选样，输出 COCO/GeoJSON 标注与不确定性栅格'
---

# AI训练数据标注 | AI Training Data Annotation

为遥感影像自动生成预标注（伪标签），并用主动学习挑出最不确定的样本送人工复核，输出标准 COCO JSON 与 GeoJSON 两种标注格式及不确定性栅格。

**核心模型**：小型 U-Net (`unet-lite`，base=8，3 级编解码 + skip)，默认在 torch + CUDA 上做 target vs background 二值语义分割。模型对单波段影像推理得到 (H, W, 2) softmax 概率图，对 target 概率做 Otsu 阈值化 + 8 邻域连通域，生成 bbox 预标注；不确定性 = 概率图香农熵，按区域平均熵 Top-k 选样送人工复核。随附预训练权重 `anno_unet_weights.pt`（缺失时在首次运行时 GPU 自动训练并落盘）。同时保留 `--method otsu` 经典阈值基线用于对比与无 GPU 环境。

## 依赖

```bash
pip install numpy rasterio scipy torch --index-url https://download.pytorch.org/whl/cu121
```

如要跑经典基线（`--method otsu`）无需 torch。

## 使用方法

### 示例 1（合成数据，离线）

```bash
python geoskill-ai-training-data-annotation.py --bbox 116.0 39.0 117.0 40.0 --synthetic --output-dir ./out
```

### 示例 2：合成数据自动标注（离线）

```bash
python geoskill-ai-training-data-annotation.py --bbox 116.0 39.0 117.0 40.0 --synthetic --n-review 2 --output-dir ./out
```

### 示例 3：真实影像预标注

```bash
python geoskill-ai-training-data-annotation.py --input scene.tif --threshold 60 --min-area 9 --format both --output-dir ./out
```

### 示例 4：仅导出 COCO

```bash
python geoskill-ai-training-data-annotation.py --bbox 121.0 31.0 122.0 32.0 --synthetic --format coco --output-dir ./out
```

## 输出

| 文件 | 格式 | 说明 |
|---|---|---|
| `annotations_coco.json` | JSON | 标准 COCO 标注（images/annotations/categories） |
| `annotations.geojson` | GeoJSON | 地理坐标预标注框 + 置信度/不确定性/复核标记 |
| `uncertainty.tif` | GeoTIFF | 逐像元熵（不确定性）栅格 |
| `output-manifest.json` | JSON | 运行清单（输入/输出/QA/退出码） |

## 数据源 / Source

本地 GeoTIFF，或 --synthetic（影像 + 边界高熵的概率图，目标真值已知）。

## 隐私声明 / Privacy

- 默认离线运行，`--synthetic` 模式完全无网络。
- 所有处理在本地完成，不上传任何用户数据。

## License

MIT
