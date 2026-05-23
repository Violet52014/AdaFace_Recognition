"""
AdaFace 特征提取：从 backend/models 下的权重加载模型，对对齐后人脸图输出 512 维 L2 归一化向量。

使用前请安装 backend/AdaFace/requirements.txt，并将 .ckpt / .pth 放入 backend/models/
或通过环境变量 ADAFACE_MODEL_PATH 指定文件路径。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ..config import ADAFACE_ROOT, Config

_model = None
_device_str: Optional[str] = None


def _ensure_adaface_on_path() -> None:
    root = str(ADAFACE_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _pick_device() -> str:
    import os

    import torch

    d = os.environ.get("ADAFACE_DEVICE")
    if d:
        return d
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_backbone_weights(model, path: Path, map_location) -> None:
    import torch

    raw = torch.load(path, map_location=map_location)
    if isinstance(raw, dict) and "state_dict" in raw:
        statedict = raw["state_dict"]
    elif isinstance(raw, dict):
        statedict = raw
    else:
        raise ValueError("无法解析权重文件，请使用 Lightning ckpt 或 state_dict")

    if any(k.startswith("model.") for k in statedict.keys()):
        model_sd = {k[6:]: v for k, v in statedict.items() if k.startswith("model.")}
    else:
        model_sd = statedict

    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if not model_sd:
        raise ValueError(f"权重中没有与 backbone 匹配的参数: {path}")


def get_model():
    """懒加载单例模型。"""
    global _model, _device_str
    if _model is not None:
        return _model, _device_str

    import torch

    path = Config.ADAFACE_MODEL_PATH
    if not path or not Path(path).is_file():
        raise FileNotFoundError(
            "未找到 AdaFace 权重：请将 .pth/.ckpt 放入 backend/models/，"
            "或设置环境变量 ADAFACE_MODEL_PATH=/绝对路径/xxx.ckpt"
        )

    _ensure_adaface_on_path()
    import net  # noqa: WPS433  # AdaFace 仓库内模块

    arch = Config.ADAFACE_ARCH
    _device_str = _pick_device()
    device = torch.device(_device_str)

    model = net.build_model(arch)
    _load_backbone_weights(model, Path(path), map_location=device)
    model = model.to(device)
    model.eval()
    _model = model
    return _model, _device_str


def _pil_to_tensor_bgr_norm(pil_rgb: Image.Image):
    """与 AdaFace inference.py to_input 一致：RGB PIL -> BGR 归一化张量。"""
    import torch

    np_img = np.array(pil_rgb.convert("RGB"))
    brg_img = ((np_img[:, :, ::-1] / 255.0) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2, 0, 1)], dtype=torch.float32)
    return tensor


def _normalize_mtcnn_bbox(box_info: tuple) -> dict:
    """MTCNN 像素框 (x1,y1,x2,y2,w_img,h_img) -> 归一化 x,y,w,h。"""
    x1, y1, x2, y2, w_img, h_img = box_info
    if w_img <= 0 or h_img <= 0:
        return {}
    w = max(float(x2) - float(x1), 1.0)
    h = max(float(y2) - float(y1), 1.0)
    return {
        "x": round(max(0.0, float(x1) / w_img), 4),
        "y": round(max(0.0, float(y1) / h_img), 4),
        "w": round(min(1.0, w / w_img), 4),
        "h": round(min(1.0, h / h_img), 4),
    }


def extract_embedding_from_bgr(
    image_bgr: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[dict]]:
    """
    从 BGR 图提取 512 维特征（已 L2 归一化）。
    返回 (embedding, error_message, bbox)；bbox 为 MTCNN 归一化框，失败时为 None。
    """
    import torch

    try:
        model, dev = get_model()
    except Exception as e:
        return None, str(e), None

    _ensure_adaface_on_path()
    from face_alignment import align  # noqa: WPS433

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    try:
        aligned, box_info = align.get_aligned_face_with_bbox(None, rgb_pil_image=pil)
    except Exception as e:
        return None, f"对齐失败: {e}", None

    if aligned is None:
        return None, "未检测到人脸或对齐失败", None

    bbox = _normalize_mtcnn_bbox(box_info) if box_info else None

    device = torch.device(dev)
    inp = _pil_to_tensor_bgr_norm(aligned).to(device)
    with torch.no_grad():
        feat, _norm = model(inp)
    vec = feat.cpu().numpy().astype(np.float32).reshape(-1)
    return vec, None, bbox


def is_adaface_available() -> bool:
    p = Config.ADAFACE_MODEL_PATH
    return bool(p and Path(p).is_file())


def parse_stored_embedding(text: Optional[str]) -> Optional[np.ndarray]:
    if not text or not text.strip():
        return None
    try:
        arr = json.loads(text)
        return np.asarray(arr, dtype=np.float32).reshape(-1)
    except json.JSONDecodeError:
        return None
