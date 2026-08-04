#!/usr/bin/env python3
"""ai-training-data-annotation — AI 训练数据标注

为遥感影像自动生成预标注（伪标签），并用主动学习挑选最不确定的样本送人工复核，
输出 COCO 与 GeoJSON 两种标准标注格式。

本 skill 是"模型预标注 + 主动学习"标注流水线的**离线 numpy 等价实现**：
不依赖 torch/标注平台，而用可验证的流程复现其核心逻辑——

1. **预标注**：阈值分割 + 连通域提取候选目标，转成 bbox 标注
   （等价于用推理模型生成伪标签）；
2. **不确定性度量**：对逐像元类别概率图计算香农熵，熵越高模型越"犹豫"；
3. **主动学习选样**：按候选标注区域的平均熵排序，选出最不确定的 k 个样本
   送人工复核（等价于不确定性采样 active learning）；
4. **格式导出**：组装标准 COCO JSON（images/annotations/categories）与 GeoJSON。

数据源：本地 GeoTIFF（影像）或 ``--synthetic`` 生成含目标与概率图的模拟数据。

隐私声明 / Privacy：
- 默认离线运行，``--synthetic`` 模式完全无网络。
- 所有处理在本地完成，不上传任何用户数据。

Usage:
    python ai-training-data-annotation.py --input scene.tif --output-dir ./out
    python ai-training-data-annotation.py --bbox 116 39 117 40 --synthetic --output-dir ./out

License: MIT
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

VERSION = "1.0.0"
SKILL_NAME = "ai-training-data-annotation"

try:
    from _geoskill_core.errors import (
        GeoSkillError, UsageError, DependencyError, ValidationError, ProcessError,
        to_exit_code,
    )
    from _geoskill_core.manifest import OutputManifest, OutputFile
except ImportError:  # pragma: no cover
    class GeoSkillError(Exception):
        def __init__(self, message: str, code: int = 7, kind: str = "EGeo", **kw):
            super().__init__(message)
            self.message, self.code, self.kind = message, code, kind

    class UsageError(GeoSkillError):
        def __init__(self, m, **k): super().__init__(m, code=2, kind="EUsage", **k)

    class DependencyError(GeoSkillError):
        def __init__(self, m, **k): super().__init__(m, code=3, kind="EDep", **k)

    class ValidationError(GeoSkillError):
        def __init__(self, m, **k): super().__init__(m, code=6, kind="EValidate", **k)

    class ProcessError(GeoSkillError):
        def __init__(self, m, **k): super().__init__(m, code=7, kind="EProcess", **k)

    def to_exit_code(exc):
        return getattr(exc, "code", 7)

    OutputManifest = None
    OutputFile = None


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# bbox validation
# ---------------------------------------------------------------------------
def validate_bbox(bbox, *, allow_antimeridian: bool = False) -> List[float]:
    """校验 bbox: [W, S, E, N]。"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise UsageError("bbox must be 4 floats [W S E N]", bbox=list(bbox) if bbox else None)
    try:
        w, s, e, n = (float(v) for v in bbox)
    except (TypeError, ValueError) as exc:
        raise UsageError("bbox entries must be numeric", bbox=list(bbox)) from exc
    if not (-180.0 <= w <= 180.0 and -180.0 <= e <= 180.0):
        raise ValidationError("bbox longitude out of range [-180, 180]", w=w, e=e)
    if not (-90.0 <= s <= 90.0 and -90.0 <= n <= 90.0):
        raise ValidationError("bbox latitude out of range [-90, 90]", s=s, n=n)
    if not (w < e):
        if allow_antimeridian and w > 0 and e < 0:
            pass
        else:
            raise ValidationError(
                "bbox requires W < E (got W={:.6f} E={:.6f}); "
                "antimeridian crossing is not supported".format(w, e),
                w=w, e=e)
    if not (s < n):
        raise ValidationError(
            "bbox requires S < N (got S={:.6f} N={:.6f})".format(s, n),
            s=s, n=n)
    return [w, s, e, n]


# ---------------------------------------------------------------------------
# Torch + cuDNN 环境：本 skill 是"模型预标注 + 主动学习"标注流水线，
# 原实现为"离线 numpy 等价实现"，本版本替换为真 torch U-Net (GPU)。
# ---------------------------------------------------------------------------
def _prepare_dll_paths() -> None:
    if os.name != "nt":
        return
    lib_bin = os.path.join(sys.prefix, "Library", "bin")
    if not os.path.isdir(lib_bin):
        return
    if lib_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = lib_bin + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(lib_bin)
    except (OSError, AttributeError):
        pass


def _require_torch():
    _prepare_dll_paths()
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise DependencyError(
            "torch is required for ai-training-data-annotation (pip install torch)") from exc
    import torch
    return torch


def _cuda_device(torch_mod):
    if not torch_mod.cuda.is_available():
        raise DependencyError(
            "ai-training-data-annotation requires a CUDA GPU; "
            "torch.cuda.is_available() is False in this environment")
    try:
        torch_mod.backends.cudnn.enabled = bool(torch_mod.backends.cudnn.is_available())
    except Exception:  # pragma: no cover
        torch_mod.backends.cudnn.enabled = False
    return torch_mod.device("cuda")


# ---------------------------------------------------------------------------
# 深度学习核心：小型 U-Net 二值语义分割网络（target vs background）
# 输入：单波段影像 (1, H, W)；输出：2 通道 logits (2, H, W)，softmax 后取 [:, 0] 为 target 概率。
# 训练：合成 ground truth probs（target_mask 中心 1，外部 0，边界过渡）→ BCEWithLogits。
# 推理：取 target 概率图 → Otsu 阈值化 → 连通域 → bbox 预标注。
# ---------------------------------------------------------------------------
WEIGHTS_FILENAME = "anno_unet_weights.pt"


def _build_unet(torch_mod, base: int = 8):
    nn = torch_mod.nn

    class _Block(nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class SegmentationUNet(nn.Module):
        def __init__(self):
            super().__init__()
            b = base
            self.enc1 = _Block(1, b)
            self.enc2 = _Block(b, b * 2)
            self.enc3 = _Block(b * 2, b * 4)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(b * 4, b * 4, 2, stride=2)
            self.dec2 = _Block(b * 4 + b * 2, b * 2)
            self.up1 = nn.ConvTranspose2d(b * 2, b * 2, 2, stride=2)
            self.dec1 = _Block(b * 2 + b, b)
            self.head = nn.Conv2d(b, 2, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            d2 = self.dec2(torch_mod.cat([self.up2(e3), e2], 1))
            d1 = self.dec1(torch_mod.cat([self.up1(d2), e1], 1))
            return self.head(d1)  # (B, 2, H, W) logits

    return SegmentationUNet()


def weights_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), WEIGHTS_FILENAME)


def _train_unet_model(
    torch_mod, model, device, *,
    epochs: int = 20, batch: int = 8, lr: float = 1e-3, n_samples: int = 32,
    img_size: int = 96, n_targets: int = 4, seed: int = 42,
) -> Dict[str, Any]:
    """在合成场景上训练 U-Net：以 p_target（H, W）作为软标签 → BCEWithLogits。

    每 epoch 重新生成 batch 张合成图（H, W）= (img_size, img_size)，
    标签 = probs 的 0 通道（target 概率），并轻微噪声 + 随机水平翻转增强。
    """
    torch_mod.manual_seed(seed)
    model = model.to(device)
    opt = torch_mod.optim.Adam(model.parameters(), lr=lr)
    bce = torch_mod.nn.BCEWithLogitsLoss()
    crit_ce = torch_mod.nn.CrossEntropyLoss()
    model.train()
    rng = np.random.default_rng(seed)
    step = 0
    for _ in range(epochs):
        # 每次 epoch 重新生成 batch 张合成图
        imgs = []
        labels = []
        for _b in range(batch):
            img, probs, _ = generate_synthetic(
                [116.0, 39.0, 117.0, 40.0], width=img_size, height=img_size,
                n_targets=int(rng.integers(2, n_targets + 1)),
                seed=int(rng.integers(0, 1_000_000)),
            )
            # 归一化到 [0, 1]
            img_n = (img - float(img.min())) / max(float(img.max() - img.min()), 1e-9)
            imgs.append(img_n.astype(np.float32))
            labels.append(probs.astype(np.float32))  # (H, W, 2)
        X = np.stack(imgs)[:, None, :, :]  # (B, 1, H, W)
        Y = np.stack(labels).transpose(0, 3, 1, 2)  # (B, 2, H, W)
        Xt = torch_mod.from_numpy(X).to(device).float()
        Yt = torch_mod.from_numpy(Y).to(device).float()
        # BCE: target probability map as soft label for class 0
        # 同时使用 CE with hard label (argmax) 加权
        soft_loss = bce(model(Xt)[:, 0:1], Yt[:, 0:1])
        hard_target = Yt.argmax(dim=1)  # (B, H, W)
        ce_loss = crit_ce(model(Xt), hard_target)
        loss = soft_loss + 0.5 * ce_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
    # holdout 评估：在 8 张未见合成对上的 Otsu-IoU vs ground truth mask
    model.eval()
    iou_sum = 0.0
    n_holdout = 8
    for k in range(n_holdout):
        img, probs, _ = generate_synthetic(
            [116.0, 39.0, 117.0, 40.0], width=img_size, height=img_size,
            n_targets=4, seed=10_000 + k,
        )
        img_n = (img - float(img.min())) / max(float(img.max() - img.min()), 1e-9)
        x_t = torch_mod.from_numpy(img_n[None, None].astype(np.float32)).to(device)
        with torch_mod.no_grad():
            logits = model(x_t)[0]
            prob_target = torch_mod.softmax(logits, dim=0)[0].cpu().numpy()
        # Otsu 自适应阈值 + ground truth mask (p_target > 0.5)
        thr = otsu_threshold(prob_target)
        pred_mask = prob_target > thr
        truth_mask = probs[..., 0] > 0.5
        inter = float(np.logical_and(pred_mask, truth_mask).sum())
        union = float(np.logical_or(pred_mask, truth_mask).sum())
        iou_sum += inter / max(union, 1.0)
    meta = {
        "arch": "unet-lite",
        "base": 8,
        "epochs": int(epochs),
        "batch": int(batch),
        "lr": float(lr),
        "img_size": int(img_size),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "steps": int(step),
        "holdout_iou_otsu": round(iou_sum / n_holdout, 4),
        "trained_at": _utc_now(),
        "framework": f"torch {torch_mod.__version__} / {torch_mod.cuda.get_device_name(0)}",
    }
    return meta


_UNET_CACHE: Dict[str, Any] = {}


def _ensure_unet(device):
    """加载随附 U-Net 权重；缺失则在 GPU 上训练并落盘。返回 (model, meta)。"""
    if "model" in _UNET_CACHE:
        return _UNET_CACHE["model"], _UNET_CACHE["meta"]
    torch_mod = _require_torch()
    model = _build_unet(torch_mod)
    path = weights_path()
    if os.path.exists(path):
        ck = torch_mod.load(path, map_location="cpu")
        model.load_state_dict(ck["state_dict"])
        meta = dict(ck.get("meta", {}))
        meta["weights"] = os.path.basename(path)
    else:
        meta = _train_unet_model(torch_mod, model, device)
        meta["weights"] = "trained-on-the-fly (weights file missing)"
        try:
            torch_mod.save({"state_dict": model.state_dict(), "meta": meta}, path)
        except Exception:  # pragma: no cover
            pass
    model = model.to(device).eval()
    _UNET_CACHE["model"] = model
    _UNET_CACHE["meta"] = meta
    return model, meta


def _predict_unet(image: np.ndarray, model, torch_mod, device) -> np.ndarray:
    """U-Net 推理：返回 (H, W, 2) softmax 概率。"""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValidationError("U-Net input must be 2D image", shape=list(img.shape))
    # 归一化到 [0, 1]
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        raise ValidationError("image has no finite values for U-Net")
    lo, hi = float(finite.min()), float(finite.max())
    img_n = (img - lo) / max(hi - lo, 1e-9) if hi > lo else np.zeros_like(img, dtype=np.float32)
    img_n = np.clip(img_n, 0.0, 1.0).astype(np.float32)
    # pad 到 4 的倍数
    h, w = img_n.shape
    ph, pw = (4 - h % 4) % 4, (4 - w % 4) % 4
    if ph or pw:
        img_n = np.pad(img_n, ((0, ph), (0, pw)), mode="edge")
    x = torch_mod.from_numpy(img_n[None, None]).to(device).float()
    with torch_mod.no_grad():
        logits = model(x)[0]
        probs = torch_mod.softmax(logits, dim=0).cpu().numpy()  # (2, H, W)
    return probs[:, :h, :w].transpose(1, 2, 0)  # (H, W, 2)


# ---------------------------------------------------------------------------
# 核心算法
# ---------------------------------------------------------------------------
def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Otsu 自动阈值（最大化类间方差），取最大平台的中点更稳健。"""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return float(v.mean())
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    mean_bg_sum = np.cumsum(hist * centers)
    mean_bg = np.divide(mean_bg_sum, weight_bg, out=np.zeros_like(mean_bg_sum),
                        where=weight_bg > 0)
    mean_fg = np.divide(mean_bg_sum[-1] - mean_bg_sum, weight_fg,
                        out=np.zeros_like(weight_fg), where=weight_fg > 0)
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    peak = float(np.max(between))
    peak_idx = np.where(between >= peak - 1e-9)[0]
    idx = int(peak_idx[len(peak_idx) // 2])
    return float(centers[idx])


def prelabel(image: np.ndarray, thresh: Optional[float] = None,
             min_area: int = 2) -> List[Dict[str, Any]]:
    """阈值 + 连通域生成 bbox 预标注。

    返回 [{bbox_px:[x1,y1,x2,y2], area_px, confidence}]，按面积降序。
    confidence 用区域内平均亮度相对全图的 z-score 近似。
    thresh=None 时用 Otsu 自动阈值。
    """
    from scipy.ndimage import label
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValidationError("prelabel expects a 2D image", shape=list(img.shape))
    if thresh is None:
        thresh = otsu_threshold(img)
    mask = img > thresh
    labels, n = label(mask, structure=np.ones((3, 3), dtype=bool))
    finite = img[np.isfinite(img)]
    g_mean = float(np.mean(finite)) if finite.size else 0.0
    g_std = float(np.std(finite)) if finite.size else 1.0
    anns: List[Dict[str, Any]] = []
    for lab in range(1, n + 1):
        ys, xs = np.where(labels == lab)
        area = int(ys.size)
        if area < min_area:
            continue
        region_mean = float(np.mean(img[ys, xs]))
        conf = (region_mean - g_mean) / (g_std if g_std > 1e-9 else 1.0)
        conf = float(np.clip(1.0 / (1.0 + np.exp(-conf)), 0.0, 1.0))
        anns.append({
            "bbox_px": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "area_px": area,
            "confidence": conf,
        })
    anns.sort(key=lambda a: a["area_px"], reverse=True)
    return anns


def pixel_entropy(probs: np.ndarray) -> np.ndarray:
    """逐像元香农熵 (以 bit 计)。probs 形如 (H, W, C)，沿类别维归一。"""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 3:
        raise ValidationError("probs must be (H, W, C)", shape=list(p.shape))
    p_sum = p.sum(axis=-1, keepdims=True)
    p = np.divide(p, p_sum, out=np.zeros_like(p), where=p_sum > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)
    ent = -(p * logp).sum(axis=-1)
    return ent


def annotation_uncertainty(entropy_map: np.ndarray, bbox_px: List[int]) -> float:
    """候选标注区域的平均熵（不确定性）。"""
    x1, y1, x2, y2 = [int(v) for v in bbox_px]
    h, w = entropy_map.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(np.mean(entropy_map[y1:y2, x1:x2]))


def select_for_review(
    annotations: List[Dict[str, Any]],
    entropy_map: np.ndarray,
    k: int,
) -> List[Dict[str, Any]]:
    """主动学习：按不确定性降序选出前 k 个预标注送人工复核。

    返回的标注带 review=True 标记，其余 review=False。
    """
    if k < 0:
        raise UsageError("k must be >= 0", k=int(k))
    scored = []
    for idx, ann in enumerate(annotations):
        u = annotation_uncertainty(entropy_map, ann["bbox_px"])
        scored.append((u, idx))
    scored.sort(key=lambda t: t[0], reverse=True)
    chosen = {idx for _, idx in scored[:k]}
    out: List[Dict[str, Any]] = []
    for idx, ann in enumerate(annotations):
        a = dict(ann)
        a["uncertainty"] = annotation_uncertainty(entropy_map, ann["bbox_px"])
        a["review"] = idx in chosen
        out.append(a)
    return out


def build_coco(
    image_w: int,
    image_h: int,
    annotations: List[Dict[str, Any]],
    categories: Optional[List[Dict[str, Any]]] = None,
    image_id: int = 1,
) -> Dict[str, Any]:
    """组装标准 COCO JSON。bbox 从 [x1,y1,x2,y2] 转 COCO 的 [x,y,w,h]。"""
    if categories is None:
        categories = [{"id": 1, "name": "target", "supercategory": "object"}]
    coco_anns = []
    for i, ann in enumerate(annotations, start=1):
        x1, y1, x2, y2 = ann["bbox_px"]
        w = x2 - x1
        h = y2 - y1
        coco_anns.append({
            "id": i,
            "image_id": image_id,
            "category_id": categories[0]["id"],
            "bbox": [int(x1), int(y1), int(w), int(h)],
            "area": int(w * h),
            "iscrowd": 0,
            "score": round(float(ann.get("confidence", 1.0)), 4),
            "uncertainty": round(float(ann.get("uncertainty", 0.0)), 4),
            "review": bool(ann.get("review", False)),
        })
    return {
        "info": {"description": "auto pre-labels", "skill": SKILL_NAME,
                 "version": VERSION},
        "images": [{"id": image_id, "width": int(image_w), "height": int(image_h),
                    "file_name": "image"}],
        "annotations": coco_anns,
        "categories": categories,
    }


def annotations_to_geojson(
    annotations: List[Dict[str, Any]], bbox: List[float],
    img_w: int, img_h: int,
) -> Dict[str, Any]:
    """预标注 -> GeoJSON FeatureCollection（地理 bbox 多边形）。"""
    w, s, e, n = bbox
    feats = []
    for idx, ann in enumerate(annotations):
        x1, y1, x2, y2 = [float(v) for v in ann["bbox_px"]]
        lon1 = w + (x1 / img_w) * (e - w)
        lon2 = w + (x2 / img_w) * (e - w)
        lat1 = n - (y1 / img_h) * (n - s)
        lat2 = n - (y2 / img_h) * (n - s)
        gminx, gminy = min(lon1, lon2), min(lat1, lat2)
        gmaxx, gmaxy = max(lon1, lon2), max(lat1, lat2)
        ring = [[gminx, gminy], [gmaxx, gminy], [gmaxx, gmaxy],
                [gminx, gmaxy], [gminx, gminy]]
        feats.append({
            "type": "Feature", "id": int(idx),
            "properties": {
                "area_px": ann["area_px"],
                "confidence": round(float(ann.get("confidence", 1.0)), 4),
                "uncertainty": round(float(ann.get("uncertainty", 0.0)), 4),
                "review": bool(ann.get("review", False)),
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    return {"type": "FeatureCollection", "features": feats}


# ---------------------------------------------------------------------------
# 合成数据：影像 + 逐像元概率图（目标边界处不确定性高）
# ---------------------------------------------------------------------------
def generate_synthetic(
    bbox: List[float],
    width: int = 96,
    height: int = 96,
    n_targets: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """返回 (image[H, W], probs[H, W, 2], info)。

    probs 的第 0 类是"目标"概率：目标内部高、背景低、边界过渡（不确定）。
    """
    rng = np.random.default_rng(seed)
    img = rng.normal(20.0, 3.0, size=(height, width)).astype(np.float32)
    target_mask = np.zeros((height, width), dtype=bool)
    pad = 14
    ys = np.linspace(pad, height - pad - 12, n_targets).astype(int)
    xs = np.linspace(pad, width - pad - 12, n_targets).astype(int)
    for k in range(n_targets):
        y, x = int(ys[k]), int(xs[k])
        img[y:y + 12, x:x + 12] += 100.0
        target_mask[y:y + 12, x:x + 12] = True

    from scipy.ndimage import distance_transform_edt
    dist_in = distance_transform_edt(target_mask)
    dist_out = distance_transform_edt(~target_mask)
    # 边界处概率接近 0.5（高熵），内部/远处接近 0/1（低熵）
    margin = np.clip(dist_in - dist_out, -8.0, 8.0)
    p_target = 1.0 / (1.0 + np.exp(-margin))
    probs = np.stack([p_target, 1.0 - p_target], axis=-1).astype(np.float32)

    info = {"bbox": bbox, "width": width, "height": height,
            "n_targets": n_targets}
    return img, probs, info


# ---------------------------------------------------------------------------
# GeoTIFF I/O
# ---------------------------------------------------------------------------
def write_geotiff(path: str, array: np.ndarray, bbox: List[float], nodata: float = -9999.0) -> None:
    import rasterio
    from rasterio.transform import from_bounds

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    nb, h, w = arr.shape
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)
    profile = {
        "driver": "GTiff", "height": h, "width": w, "count": nb,
        "dtype": "float32", "crs": "EPSG:4326", "transform": transform,
        "nodata": nodata, "compress": "deflate",
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        for b in range(nb):
            dst.write(arr[b].astype("float32"), b + 1)


def read_geotiff(path: str) -> Tuple[np.ndarray, List[float]]:
    """读取 GeoTIFF，返回 (cube, bbox)。

    全 NoData → ValidationError exit 6。valid_mask 在内部由
    ``_read_geotiff_with_mask`` 暴露给 ``process()``。
    """
    cube, bbox, _valid = _read_geotiff_with_mask(path)
    return cube, bbox


def _read_geotiff_with_mask(path: str) -> Tuple[np.ndarray, List[float], np.ndarray]:
    """读取 GeoTIFF，返回 (cube, bbox, valid_mask)。

    valid_mask: (H, W) 布尔，True = 有效（非 nodata、非 NaN）。
    全 NoData → ValidationError exit 6。
    """
    import rasterio
    if not os.path.exists(path):
        raise UsageError(f"input raster not found: {path}", path=path)
    with rasterio.open(path) as src:
        cube = src.read().astype(np.float32)
        b = src.bounds
        bbox = [b.left, b.bottom, b.right, b.top]
        nd = src.nodata
    band0 = cube[0] if cube.ndim == 3 else cube
    if nd is None:
        valid = np.isfinite(band0)
    else:
        valid = np.isfinite(band0) & (band0 != float(nd))
    if not bool(valid.any()):
        raise ValidationError(
            f"input raster has no valid (non-NoData) pixels: {path}",
            path=path, nodata=nd)
    return cube, bbox, valid


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def write_manifest(output_dir, args, outputs, qa, started_at, exit_code, bbox):
    if OutputManifest is None:
        return None
    cmd = " ".join([SKILL_NAME] + sys.argv[1:])
    man = OutputManifest(
        skill=SKILL_NAME, skill_version=VERSION, command=cmd,
        started_at=started_at, finished_at=_utc_now(), exit_code=exit_code,
        inputs={
            "input": getattr(args, "input", None),
            "n_review": getattr(args, "n_review", None),
            "synthetic": bool(getattr(args, "synthetic", False)),
        },
        outputs=[OutputFile(**o) for o in outputs],
        qa=qa,
        software={"python": sys.version.split()[0], "skill": SKILL_NAME},
    )
    path = os.path.join(output_dir, "output-manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(man.to_dict(), f, ensure_ascii=False, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def process(args: argparse.Namespace) -> int:
    started_at = _utc_now()
    output_dir = args.output_dir

    # 1) bbox 校验（先于 IO）
    bbox = list(args.bbox) if args.bbox else None
    if bbox is not None:
        bbox = validate_bbox(bbox)

    # 2) 参数校验
    if args.n_review is not None and int(args.n_review) < 0:
        raise ValidationError(
            f"n-review must be >= 0 (got {args.n_review})", n_review=args.n_review)
    if args.min_area is not None and int(args.min_area) < 1:
        raise UsageError(f"min-area must be >= 1 (got {args.min_area})",
                         min_area=args.min_area)
    if args.threshold is not None and not (0.0 < float(args.threshold) < 1.0):
        # 允许 0-1 之外的阈值（因为 input mode 的像素值是绝对亮度），仅在 cnn mode 强制 [0,1]
        if args.method == "cnn":
            raise UsageError(
                f"threshold must be in (0, 1) for --method cnn (got {args.threshold})",
                threshold=args.threshold)

    os.makedirs(output_dir, exist_ok=True)

    # 3) GPU / 模型加载
    device = None
    if args.method == "cnn":
        torch_mod = _require_torch()
        device = _cuda_device(torch_mod)

    # 4) 读取 / 生成
    bbox_4326: List[float] = bbox if bbox is not None else [116.0, 39.0, 117.0, 40.0]
    probs: Optional[np.ndarray] = None
    synth_info: Optional[Dict[str, Any]] = None
    valid_mask: Optional[np.ndarray] = None

    if args.input and not args.synthetic:
        cube, file_bbox, valid_mask = _read_geotiff_with_mask(args.input)
        bbox = bbox if bbox is not None else validate_bbox(file_bbox)
        image = cube[0] if cube.ndim == 3 else cube
        source_note = args.input
    else:
        if bbox is None:
            raise UsageError("provide --bbox (synthetic mode) or --input <raster>")
        image, probs, synth_info = generate_synthetic(bbox)
        source_note = "synthetic"

    if image.size == 0:
        raise ValidationError("input raster is empty")
    # NoData 像元置 0
    if valid_mask is not None:
        image = np.where(valid_mask, image, 0.0).astype(np.float32)
    h, w = image.shape

    # 5) 预标注：cnn → U-Net 概率图 + Otsu 阈值；otsu → 经典阈值 + 连通域
    if args.method == "cnn":
        model, unet_meta = _ensure_unet(device)
        torch_mod = _require_torch()
        probs = _predict_unet(image, model, torch_mod, device)  # (H, W, 2)
        if args.threshold is None:
            thr = otsu_threshold(probs[..., 0])
        else:
            thr = float(args.threshold)
        # 临时把 probs 替换为 U-Net 输出后再做 bbox 提取
        from scipy.ndimage import label as _label
        mask = probs[..., 0] > thr
        labels, n = _label(mask, structure=np.ones((3, 3), dtype=bool))
        finite = image[np.isfinite(image)]
        g_mean = float(np.mean(finite)) if finite.size else 0.0
        g_std = float(np.std(finite)) if finite.size else 1.0
        anns: List[Dict[str, Any]] = []
        for lab in range(1, n + 1):
            ys, xs = np.where(labels == lab)
            area = int(ys.size)
            if area < args.min_area:
                continue
            region_mean = float(np.mean(image[ys, xs]))
            conf = (region_mean - g_mean) / (g_std if g_std > 1e-9 else 1.0)
            conf = float(np.clip(1.0 / (1.0 + np.exp(-conf)), 0.0, 1.0))
            anns.append({
                "bbox_px": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                "area_px": area,
                "confidence": conf,
            })
        anns.sort(key=lambda a: a["area_px"], reverse=True)
        annotations = anns
    else:
        annotations = prelabel(image, thresh=args.threshold, min_area=args.min_area)

    if probs is None:
        # 真实输入 + otsu 模式：启发式概率图
        z = (image - float(np.mean(image))) / max(float(np.std(image)), 1e-9)
        p = 1.0 / (1.0 + np.exp(-z.astype(np.float64)))
        probs = np.stack([p, 1.0 - p], axis=-1)

    entropy = pixel_entropy(probs)
    # NoData 像元不确定性置 0
    if valid_mask is not None:
        entropy = np.where(valid_mask, entropy, 0.0)
    annotations = select_for_review(annotations, entropy, k=args.n_review)

    outputs: List[Dict[str, Any]] = []
    if args.format in ("coco", "both"):
        coco = build_coco(w, h, annotations)
        coco_path = os.path.join(output_dir, "annotations_coco.json")
        with open(coco_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False, indent=2)
        outputs.append({"path": coco_path, "kind": "json"})
    if args.format in ("geojson", "both"):
        gj = annotations_to_geojson(annotations, bbox, w, h)
        geo_path = os.path.join(output_dir, "annotations.geojson")
        with open(geo_path, "w", encoding="utf-8") as f:
            json.dump(gj, f, ensure_ascii=False, indent=2)
        outputs.append({"path": geo_path, "kind": "vector", "crs_epsg": 4326,
                        "bbox_wgs84": bbox})

    ent_path = os.path.join(output_dir, "uncertainty.tif")
    write_geotiff(ent_path, entropy.astype(np.float32), bbox)
    outputs.append({"path": ent_path, "kind": "raster", "crs_epsg": 4326,
                    "bbox_wgs84": bbox, "band_count": 1})

    n_review = sum(1 for a in annotations if a.get("review"))
    qa: Dict[str, Any] = {
        "source": source_note,
        "method": args.method,
        "n_annotations": int(len(annotations)),
        "n_review": int(n_review),
        "mean_uncertainty": float(np.mean([a["uncertainty"] for a in annotations]))
        if annotations else 0.0,
        "valid_pixel_count": int(valid_mask.sum()) if valid_mask is not None else int(image.size),
    }
    if synth_info is not None:
        qa["synthetic_n_targets"] = synth_info["n_targets"]
    if args.method == "cnn":
        qa["backend"] = f"torch unet (base={unet_meta.get('base', 8)})"
        qa["device"] = str(device)
        qa["model_holdout_iou"] = unet_meta.get("holdout_iou_otsu")
        qa["weights"] = unet_meta.get("weights", "n/a")

    man_path = write_manifest(output_dir, args, outputs, qa, started_at, 0, bbox)

    if not args.quiet:
        print(f"[{SKILL_NAME}] source: {source_note}  method: {args.method}")
        print(f"[{SKILL_NAME}] pre-labels: {len(annotations)}  sent for review: {n_review}")
        for o in outputs:
            print(f"[{SKILL_NAME}] output: {o['path']}")
        if man_path:
            print(f"[{SKILL_NAME}] manifest: {man_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=SKILL_NAME,
        description="AI training-data annotation (pre-labeling + active learning, offline equivalent).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                   help="geographic extent [minLon minLat maxLon maxLat]")
    p.add_argument("--input", help="input GeoTIFF (first band used)")
    p.add_argument("--method", default="cnn", choices=["cnn", "otsu"],
                   help="pre-labeling method (default: cnn — torch U-Net on GPU; "
                        "otsu is the interpretable numpy baseline)")
    p.add_argument("--threshold", type=float, default=None,
                   help="pre-label threshold; default: Otsu auto. For --method cnn, "
                        "must be in (0, 1) (target probability).")
    p.add_argument("--min-area", type=int, default=4, help="drop candidates smaller than this (px)")
    p.add_argument("--n-review", type=int, default=2, help="number of uncertain samples for review")
    p.add_argument("--format", default="both", choices=["coco", "geojson", "both"],
                   help="annotation export format (default: both)")
    p.add_argument("--synthetic", action="store_true", help="generate synthetic data (offline)")
    p.add_argument("--output-dir", default="./output", help="output directory")
    p.add_argument("--quiet", action="store_true", help="suppress console output")
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return process(args)
    except GeoSkillError as exc:
        print(f"[{SKILL_NAME}] ERROR [{exc.kind}] {exc.message}", file=sys.stderr)
        return exc.code
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[{SKILL_NAME}] ERROR {exc}", file=sys.stderr)
        return to_exit_code(exc)


if __name__ == "__main__":
    sys.exit(main())
