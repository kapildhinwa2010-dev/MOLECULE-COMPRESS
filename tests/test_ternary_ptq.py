import math

import torch
from torch import nn

import ternary_ptq as tp


def test_values_are_ternary():
    w = torch.randn(100, 100)
    t = tp.ternarize_tensor(w)
    vals = set(t.unique().tolist())
    assert vals.issubset({-1.0, 0.0, 1.0})


def test_zero_tensor_is_safe():
    w = torch.zeros(32, 32)
    t = tp.ternarize_tensor(w)
    assert torch.all(t == 0)
    assert not torch.isnan(t).any()


def test_ternarize_linear_inplace():
    lin = nn.Linear(64, 32)
    hit = tp.ternarize_(lin)
    assert hit == 1
    assert lin.weight.shape == (32, 64)
    vals = set(lin.weight.float().unique().tolist())
    assert vals.issubset({-1.0, 0.0, 1.0})
    # bias untouched
    assert lin.bias is None or torch.isfinite(lin.bias).all()


def test_size_math():
    m = nn.Linear(1000, 1000, bias=False)  # exactly 1e6 weights, nothing else
    s = tp.size_report(m)
    assert s["linear_params"] == 1_000_000
    assert s["total_params"] == 1_000_000
    # 5 packed trits per byte -> 200_000 bytes for the linear weights
    assert s["ternary_bytes"] == 200_000
    # int8 linear weights -> 1e6 bytes
    assert s["int8_bytes"] == 1_000_000
    assert s["bf16_bytes"] == 2_000_000


def test_demo_report_fields():
    rep = tp.demo_report()
    assert rep["layers_ternarized"] == 2
    assert rep["unique_values"] == [-1.0, 0.0, 1.0]
    assert rep["sizes"]["linear_params"] == 64 * 128 + 128 * 64


def test_absmean_scale_positive():
    w = torch.randn(50, 50) * 0.001
    s = tp.absmean_scale(w)
    assert s.item() > 0


def test_version():
    assert tp.__version__ == "0.1.0"
