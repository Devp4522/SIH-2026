"""YAML config loading with dotted-key command-line overrides.

    cfg = load_config("configs/edsr_baseline.yaml", overrides=["train.epochs=5", "model.n_feats=32"])
    cfg.train.epochs  -> 5
"""
from __future__ import annotations

import copy
import os
import re

import yaml


class Cfg(dict):
    """dict with attribute access (cfg.train.lr) that stays YAML/JSON-serialisable."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    def get_path(self, dotted, default=None):
        node = self
        for k in dotted.split("."):
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


def _wrap(obj):
    if isinstance(obj, dict):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def to_plain(obj):
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain(v) for v in obj]
    return obj


def _deep_merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_overrides(cfg: dict, overrides):
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must look like key.sub=value, got {ov!r}")
        key, raw = ov.split("=", 1)
        value = yaml.safe_load(raw)  # "5" -> 5, "1e-4" -> 1e-4 (str), "[1,2]" -> list, "true" -> True
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _coerce_numbers(obj):
    """PyYAML reads '1e-4' as a string; turn numeric-looking strings into floats."""
    if isinstance(obj, dict):
        return {k: _coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numbers(v) for v in obj]
    if isinstance(obj, str) and _NUM_RE.match(obj.strip()):
        return float(obj)
    return obj


def load_config(path: str, overrides=None) -> Cfg:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base_path = raw.pop("_base_", None)
    if base_path:
        base = to_plain(load_config(os.path.join(os.path.dirname(path), base_path)))
        raw = _deep_merge(base, raw)
    raw = apply_overrides(raw, overrides)
    return _wrap(_coerce_numbers(raw))


def save_config(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(to_plain(cfg), f, sort_keys=False)
