import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from degradations import add_gaussian_noise, apply_degradation, reduce_resolution  # noqa: E402


@pytest.fixture
def image():
    torch.manual_seed(0)
    return torch.randn(4, 20, 24, 18)


def test_noise_is_deterministic_per_seed_and_has_requested_scale(image):
    a = add_gaussian_noise(image, sigma=0.2, seed=7)
    b = add_gaussian_noise(image, sigma=0.2, seed=7)
    c = add_gaussian_noise(image, sigma=0.2, seed=8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert (a - image).std().item() == pytest.approx(0.2, rel=0.05)


def test_noise_does_not_mutate_input(image):
    original = image.clone()
    add_gaussian_noise(image, sigma=0.5, seed=1)
    assert torch.equal(image, original)


def test_zero_sigma_returns_copy(image):
    out = add_gaussian_noise(image, sigma=0.0, seed=1)
    assert torch.equal(out, image) and out.data_ptr() != image.data_ptr()


def test_negative_sigma_rejected(image):
    with pytest.raises(ValueError):
        add_gaussian_noise(image, sigma=-0.1, seed=1)


def test_lowres_preserves_shape_and_smooths(image):
    out = reduce_resolution(image, factor=0.5)
    assert out.shape == image.shape
    # Averaging then interpolating removes high-frequency content.
    assert out.std() < image.std() * 0.8
    assert torch.equal(reduce_resolution(image, factor=1.0), image)


def test_lowres_rejects_bad_factor(image):
    with pytest.raises(ValueError):
        reduce_resolution(image, factor=0.0)
    with pytest.raises(ValueError):
        reduce_resolution(image, factor=1.5)


def test_apply_degradation_dispatch(image):
    assert torch.equal(apply_degradation(image, "clean", 1.0, seed=0), image)
    assert apply_degradation(image, "noise", 0.1, seed=0).shape == image.shape
    assert apply_degradation(image, "lowres", 0.5, seed=0).shape == image.shape
    with pytest.raises(ValueError):
        apply_degradation(image, "blur", 1.0, seed=0)
