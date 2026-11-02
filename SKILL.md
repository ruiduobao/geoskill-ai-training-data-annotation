---
name: geoskill-ai-training-data-annotation
description: '小型 U-Net 二值语义分割 (torch+CUDA) 预标注 + 主动学习不确定性选样，输出 COCO/GeoJSON 标注与不确定性栅格'
---

# AI训练数据标注 | AI Training Data Annotation

Automatically generates pre-annotations (pseudo-labels) for remote sensing imagery and uses active learning to select the most uncertain samples for manual review, outputting both standard COCO JSON and GeoJSON annotation formats plus an uncertainty raster.

**Core model**: a small U-Net (`unet-lite`, base=8, 3-level encoder-decoder with skip connections) performing target vs background binary semantic segmentation by default on torch + CUDA. The model infers on single-band imagery to produce an (H, W, 2) softmax probability map; the target probability is Otsu-thresholded and connected components are extracted with 8-neighborhood connectivity to generate bbox pre-annotations. Uncertainty is computed as the Shannon entropy of the probability map; the Top-k samples by regional mean entropy are selected for manual review. Pretrained weights `anno_unet_weights.pt` are shipped with the skill (automatically trained on GPU and persisted at first run when missing). The `--method otsu` classical threshold baseline is also retained for comparison and GPU-free environments.

## Dependencies / 依赖

```bash
pip install numpy rasterio scipy torch --index-url https://download.pytorch.org/whl/cu121
```

The classical baseline (`--method otsu`) does not require torch.

## Usage / 使用方法

### Example 1 (Synthetic Data, Offline)

```bash
python geoskill-ai-training-data-annotation.py --bbox 116.0 39.0 117.0 40.0 --synthetic --output-dir ./out
```

### Example 2: Automatic Annotation of Synthetic Data (Offline)

```bash
python geoskill-ai-training-data-annotation.py --bbox 116.0 39.0 117.0 40.0 --synthetic --n-review 2 --output-dir ./out
```

### Example 3: Real-Image Pre-Annotation

```bash
python geoskill-ai-training-data-annotation.py --input scene.tif --threshold 60 --min-area 9 --format both --output-dir ./out
```

### 示例 4：仅导出 COCO

```bash
python geoskill-ai-training-data-annotation.py --bbox 121.0 31.0 122.0 32.0 --synthetic --format coco --output-dir ./out
```

## Output / 输出

| File | Format | Description |
|---|---|---|
| `annotations_coco.json` | JSON | Standard COCO annotations (images/annotations/categories) |
| `annotations.geojson` | GeoJSON | Georeferenced pre-annotation boxes + confidence/uncertainty/review flags |
| `uncertainty.tif` | GeoTIFF | Per-pixel entropy (uncertainty) raster |
| `output-manifest.json` | JSON | Run manifest (inputs/outputs/QA/exit code) |

## Data Source / 数据源 / Source

Local GeoTIFF, or --synthetic (probability map with high entropy at boundaries; target ground truth known).

## Privacy / 隐私声明 / Privacy

- Runs offline by default; `--synthetic` mode requires no network at all.
- All processing is done locally; no user data is ever uploaded.

## License / License

MIT

---

> 中文完整版：需要时再读 [references/SKILL.zh-CN.md](references/SKILL.zh-CN.md)，不要默认全量加载。
