# placeholder
# src/config.py
import yaml
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def seed(self):
        return self.raw.get("seed", 42)

    @property
    def device(self):
        return self.raw.get("device", "cuda")

    def get(self, *keys, default=None):
        d = self.raw
        for k in keys:
            d = d.get(k, {})
        return d if d != {} else default


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Config(cfg)
