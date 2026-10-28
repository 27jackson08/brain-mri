import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common  # noqa: E402
from prepare_data import convert_labels, foreground_bbox, make_split, normalize_nonzero  # noqa: E402
from transforms import CONFIGS, build_eval_transform, build_train_transform  # noqa: E402


def test_convert_labels_follows_dataset_json_semantics():
    label = np.array([[[0, 1, 2, 3]]], dtype=np.uint8)
    regions = convert_labels(label)
    assert regions.shape == (3, 1, 1, 4)
    assert regions[0].ravel().tolist() == [0, 0, 1, 1]  # TC = non-enhancing + enhancing
    assert regions[1].ravel().tolist() == [0, 1, 1, 1]  # WT = edema + TC
    assert regions[2].ravel().tolist() == [0, 0, 0, 1]  # ET = enhancing only


def test_normalize_nonzero_zscores_brain_and_keeps_background_zero():
    image = np.zeros((1, 4, 4, 4), dtype=np.float32)
    image[0, 1:3, 1:3, 1:3] = np.arange(8, dtype=np.float32).reshape(2, 2, 2) + 10
    out = normalize_nonzero(image)
    brain = out[0, 1:3, 1:3, 1:3]
    assert brain.mean() == pytest.approx(0.0, abs=1e-6)
    assert brain.std() == pytest.approx(1.0, abs=1e-5)
    assert out[0, 0, 0, 0] == 0.0
    assert image[0, 1, 1, 1] == 10.0  # input untouched


def test_foreground_bbox_is_tight():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:5, 3:9, 0:1] = True
    assert foreground_bbox(mask) == (slice(2, 5), slice(3, 9), slice(0, 1))
    with pytest.raises(ValueError):
        foreground_bbox(np.zeros((2, 2, 2), dtype=bool))


def test_split_is_deterministic_and_disjoint():
    ids = [f"BRATS_{i:03d}" for i in range(50)]
    a = make_split(ids, 20, 5, 10, seed=0)
    b = make_split(list(reversed(ids)), 20, 5, 10, seed=0)
    assert a == b
    assert len(set(a["train"]) & set(a["val"])) == 0
    assert len(set(a["train"]) & set(a["test"])) == 0
    assert len(a["unused"]) == 15
    with pytest.raises(ValueError):
        make_split(ids, 40, 10, 10, seed=0)


@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_train_transform_produces_patches(tmp_path, monkeypatch, config):
    monkeypatch.setattr(common, "PREPROCESSED_DIR", tmp_path)
    import transforms

    monkeypatch.setattr(transforms, "case_paths", lambda cid: (tmp_path / f"{cid}_image.npy", tmp_path / f"{cid}_label.npy"))
    rng = np.random.default_rng(0)
    image = rng.standard_normal((4, 100, 110, 90)).astype(np.float16)
    label = np.zeros((3, 100, 110, 90), dtype=np.uint8)
    label[:, 40:60, 40:60, 40:60] = 1
    np.save(tmp_path / "X_image.npy", image)
    np.save(tmp_path / "X_label.npy", label)

    out = build_train_transform(config, samples_per_case=2)({"case_id": "X"})
    assert len(out) == 2
    for sample in out:
        assert tuple(sample["image"].shape) == (4, *common.PATCH_SIZE)
        assert tuple(sample["label"].shape) == (3, *common.PATCH_SIZE)
        assert set(torch.unique(sample["label"]).tolist()) <= {0.0, 1.0}

    full = build_eval_transform()({"case_id": "X"})
    assert tuple(full["image"].shape) == (4, 100, 110, 90)
