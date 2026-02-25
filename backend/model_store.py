import json
import os
import time
import uuid
from typing import Dict, Optional


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def build_model_version(prefix: str = "transition") -> str:
    stamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    token = uuid.uuid4().hex[:8]
    return f"{prefix}_{stamp}_{token}"


def save_model_artifact(model_dir: str, payload: Dict) -> str:
    ensure_dir(model_dir)
    version = str(payload.get("version") or build_model_version())
    payload = dict(payload)
    payload["version"] = version
    path = os.path.join(model_dir, f"{version}.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return path


def load_model_artifact(path: str) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
