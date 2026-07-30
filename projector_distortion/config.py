# projector_distortion/config.py
"""
YAML config loading and the precedence rules.

Resolution order, lowest to highest:
    1. the bundled configs/*.yaml
    2. a file passed via --config
    3. explicit CLI flags

Paths inside a config are resolved relative to the project root (the directory holding
`weights/` and `data/`), not the current working directory, so `python demo.py` works
from anywhere.
"""

import os
from typing import Any, Dict, Optional

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(_PKG_DIR, "configs")
#: Repo root when running from a checkout; falls back to the package dir once installed.
PROJECT_ROOT = os.path.dirname(_PKG_DIR)


def load_yaml(path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("reading configs needs PyYAML: pip install PyYAML") from e
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursive dict merge; `override` wins. Neither input is mutated."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(name: str, extra: Optional[str] = None) -> Dict[str, Any]:
    """
    Load `configs/<name>.yaml`, then merge `extra` (a path) on top when given.

    `name` is 'restoration' or 'detection'.
    """
    cfg = load_yaml(os.path.join(CONFIG_DIR, f"{name}.yaml"))
    if extra:
        if not os.path.exists(extra):
            raise FileNotFoundError(f"config not found: {extra}")
        cfg = deep_merge(cfg, load_yaml(extra))
    return cfg


def resolve_path(path, root: Optional[str] = None) -> Optional[str]:
    """Make a config-relative path absolute against the project root."""
    if not path:
        return None
    path = str(path)
    if os.path.isabs(path):
        return path
    candidate = os.path.join(root or PROJECT_ROOT, path)
    return candidate if os.path.exists(candidate) else os.path.abspath(path)


def pick(cli_value, cfg: Dict, *keys, default=None):
    """
    CLI wins when it is not None, otherwise walk `keys` through `cfg`.

    pick(args.conf, det_cfg, "detector", "conf", default=0.25)
    """
    if cli_value is not None:
        return cli_value
    node: Any = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return default if node is None else node
