from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def imread_bgr_unicode(path) -> Optional[np.ndarray]:
    """
    以 BGR ndarray 读取图片。Windows 上 cv2.imread 对含中文等非 ANSI 路径会失败，
    使用 read_bytes + imdecode 可正确读取。
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = p.read_bytes()
    except OSError:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def decode_base64_image(image_base64: str) -> np.ndarray:
    if not image_base64:
        raise ValueError("image 不能为空")

    image_bytes = base64.b64decode(image_base64)
    return decode_image_bytes(image_bytes)


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """从原始图片字节解码为 BGR ndarray（与 Base64 解码后一致）。"""
    if not image_bytes:
        raise ValueError("图片数据为空")
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
