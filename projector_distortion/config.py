"""
YAML config loading and precedence.

configs/*.yaml holds the settings, and the handful of CLI flags that remain override
it. Paths inside a config resolve against the project root, not the working
directory, so `python demo.py` works from anywhere.
"""

import os
from typing import Any, Dict, Optional

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(_PKG_DIR, "configs")
PROJECT_ROOT = os.path.dirname(_PKG_DIR)


def load_yaml(path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("reading configs needs PyYAML: pip install PyYAML") from e
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(name: str) -> Dict[str, Any]:
    """Load `configs/<name>.yaml` ('restoration' | 'detection')."""
    return load_yaml(os.path.join(CONFIG_DIR, f"{name}.yaml"))


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
