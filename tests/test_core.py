"""Core algorithm tests for ai-training-data-annotation."""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from conftest import mod


class TestOtsu:
    def test_bimodal(self):
        rng = np.random.default_rng(0)
        v = np.concatenate([rng.normal(10, 1, 4000), rng.normal(100, 1, 4000)])
        t = mod.otsu_threshold(v)
        assert 30 < t < 80

    def test_empty(self):
        assert mod.otsu_threshold(np.array([np.nan])) == 0.0


class TestPrelabel:
    def test_counts_targets(self):
        img = np.zeros((60, 60))
        img[5:15, 5:15] = 100.0
        img[40:50, 40:50] = 100.0
        anns = mod.prelabel(img, thresh=50.0)
        assert len(anns) == 2

    def test_bbox_valid(self):
        img = np.zeros((40, 40))
        img[10:20, 12:24] = 100.0  # 10 行 x 12 列
        anns = mod.prelabel(img, thresh=50.0)
        assert len(anns) == 1
        a = anns[0]
        assert a["bbox_px"] == [12, 10, 24, 20]
        assert a["area_px"] == 120
        assert 0.0 <= a["confidence"] <= 1.0

    def test_auto_threshold(self):
        img = np.zeros((50, 50))
        img[20:30, 20:30] = 200.0
        anns = mod.prelabel(img, thresh=None)  # Otsu
        assert len(anns) >= 1

    def test_min_area_filter(self):
        img = np.zeros((30, 30))
        img[0, 0] = 100.0        # 面积 1
        img[15:20, 15:20] = 100.0  # 面积 25
        anns = mod.prelabel(img, thresh=50.0, min_area=5)
        assert len(anns) == 1
        assert anns[0]["area_px"] == 25

    def test_rejects_3d(self):
        with pytest.raises(mod.ValidationError):
            mod.prelabel(np.zeros((2, 8, 8)))

    def test_sorted_by_area_desc(self):
        img = np.zeros((40, 40))
        img[0:2, 0:2] = 100.0      # 4
        img[20:30, 20:30] = 100.0  # 100
        anns = mod.prelabel(img, thresh=50.0)
        assert anns[0]["area_px"] == 100


class TestPixelEntropy:
    def test_uniform_max_entropy(self):
        probs = np.full((4, 4, 2), 0.5)
        ent = mod.pixel_entropy(probs)
        np.testing.assert_allclose(ent, 1.0, atol=1e-9)  # 2 类最大熵 = 1 bit

    def test_onehot_zero_entropy(self):
        probs = np.zeros((4, 4, 3))
        probs[..., 0] = 1.0
        ent = mod.pixel_entropy(probs)
        np.testing.assert_allclose(ent, 0.0, atol=1e-9)

    def test_normalizes_unnormalized(self):
        probs = np.full((2, 2, 2), 2.0)  # 未归一，和为 4
        ent = mod.pixel_entropy(probs)
        np.testing.assert_allclose(ent, 1.0, atol=1e-9)

    def test_rejects_2d(self):
        with pytest.raises(mod.ValidationError):
            mod.pixel_entropy(np.zeros((4, 4)))


class TestAnnotationUncertainty:
    def test_mean_entropy_in_box(self):
        ent = np.zeros((10, 10))
        ent[2:4, 2:4] = 1.0  # 4 个像元熵=1
        u = mod.annotation_uncertainty(ent, [2, 2, 4, 4])
        assert u == pytest.approx(1.0)

    def test_empty_box_zero(self):
        ent = np.ones((10, 10))
        assert mod.annotation_uncertainty(ent, [5, 5, 5, 5]) == 0.0

    def test_clipped_to_bounds(self):
        ent = np.ones((10, 10))
        u = mod.annotation_uncertainty(ent, [-5, -5, 3, 3])
        assert u == pytest.approx(1.0)


class TestSelectForReview:
    def test_picks_most_uncertain(self):
        ent = np.zeros((20, 20))
        ent[0:4, 0:4] = 0.1      # 低不确定性
        ent[10:14, 10:14] = 0.9  # 高不确定性
        anns = [
            {"bbox_px": [0, 0, 4, 4], "area_px": 16},
            {"bbox_px": [10, 10, 14, 14], "area_px": 16},
        ]
        out = mod.select_for_review(anns, ent, k=1)
        reviews = [a for a in out if a["review"]]
        assert len(reviews) == 1
        assert reviews[0]["bbox_px"] == [10, 10, 14, 14]
        assert reviews[0]["uncertainty"] == pytest.approx(0.9)

    def test_k_zero_no_review(self):
        ent = np.ones((10, 10))
        anns = [{"bbox_px": [0, 0, 2, 2], "area_px": 4}]
        out = mod.select_for_review(anns, ent, k=0)
        assert not any(a["review"] for a in out)

    def test_negative_k_raises(self):
        with pytest.raises(mod.UsageError):
            mod.select_for_review([], np.ones((4, 4)), k=-1)


class TestBuildCoco:
    def test_structure_and_conversion(self):
        anns = [{"bbox_px": [10, 20, 30, 50], "area_px": 600,
                 "confidence": 0.8, "uncertainty": 0.3, "review": True}]
        coco = mod.build_coco(100, 100, anns)
        assert set(coco.keys()) >= {"images", "annotations", "categories"}
        assert coco["images"][0]["width"] == 100
        a = coco["annotations"][0]
        assert a["bbox"] == [10, 20, 20, 30]  # [x, y, w, h]
        assert a["area"] == 600
        assert a["category_id"] == coco["categories"][0]["id"]
        assert a["review"] is True

    def test_unique_ids(self):
        anns = [{"bbox_px": [0, 0, 5, 5], "area_px": 25} for _ in range(3)]
        coco = mod.build_coco(50, 50, anns)
        ids = [a["id"] for a in coco["annotations"]]
        assert len(set(ids)) == 3


class TestAnnotationsToGeoJSON:
    def test_structure(self):
        anns = [{"bbox_px": [0, 0, 10, 10], "area_px": 100,
                 "confidence": 0.9, "uncertainty": 0.2, "review": False}]
        gj = mod.annotations_to_geojson(anns, [116, 39, 117, 40], 100, 100)
        assert gj["type"] == "FeatureCollection"
        feat = gj["features"][0]
        assert feat["geometry"]["type"] == "Polygon"
        ring = feat["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1]
        assert feat["properties"]["confidence"] == pytest.approx(0.9)


class TestSynthetic:
    def test_prelabel_finds_targets(self):
        img, probs, info = mod.generate_synthetic([116, 39, 117, 40], n_targets=4, seed=5)
        anns = mod.prelabel(img, thresh=None)
        assert 3 <= len(anns) <= 6
        assert probs.shape[2] == 2

    def test_boundary_most_uncertain(self):
        """目标边界处的熵应高于目标内部。"""
        _, probs, _ = mod.generate_synthetic([116, 39, 117, 40], n_targets=1, seed=9)
        ent = mod.pixel_entropy(probs)
        # 边界像元 margin≈±1 -> 熵≈0.84 bit；内部/背景熵接近 0
        assert ent.max() > 0.8
        assert ent.min() < 0.1


class TestGeoTiffIO:
    def test_roundtrip(self, tmp_path):
        arr = np.random.uniform(0, 1, (16, 16)).astype(np.float32)
        bbox = [116.0, 39.0, 117.0, 40.0]
        path = str(tmp_path / "x.tif")
        mod.write_geotiff(path, arr, bbox)
        back, rb = mod.read_geotiff(path)
        np.testing.assert_allclose(back[0], arr, atol=1e-5)
        np.testing.assert_allclose(rb, bbox, atol=1e-6)

    def test_read_missing_raises(self):
        with pytest.raises(mod.UsageError):
            mod.read_geotiff("/nonexistent/x.tif")
