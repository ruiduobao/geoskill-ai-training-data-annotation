# AI Training Data Annotation (geoskill-ai-training-data-annotation)

> Small U-Net binary semantic segmentation (torch+CUDA) pre-labeling + active-learning uncertainty sampling, outputting COCO/GeoJSON annotations and an uncertainty raster

---

## 1. Overview

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

## 8. License

MIT License. See [`LICENSE`](./LICENSE) for full text.

---

> 中文完整版：[references/SKILL.zh-CN.md](references/SKILL.zh-CN.md)
