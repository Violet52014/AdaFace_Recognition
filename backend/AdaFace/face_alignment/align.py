import sys
import os

from face_alignment import mtcnn
import argparse
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime

try:
    import torch

    _default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
except Exception:
    _default_device = "cpu"
_mtcnn_device = os.environ.get("ADAFACE_DEVICE", _default_device)
mtcnn_model = mtcnn.MTCNN(device=_mtcnn_device, crop_size=(112, 112))

def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def get_aligned_face(image_path, rgb_pil_image=None):
    face, _box_info = get_aligned_face_with_bbox(image_path, rgb_pil_image)
    return face


def get_aligned_face_with_bbox(image_path, rgb_pil_image=None):
    """
    返回对齐后人脸图，以及 MTCNN 检测框（像素坐标 x1,y1,x2,y2 + 原图宽高）。
    用于前端画框时在后端归一化；未检测到人脸时返回 (None, None)。
    """
    if rgb_pil_image is None:
        img = Image.open(image_path).convert('RGB')
    else:
        assert isinstance(rgb_pil_image, Image.Image), 'Face alignment module requires PIL image or path to the image'
        img = rgb_pil_image
    try:
        bboxes, faces = mtcnn_model.align_multi(img, limit=1)
        if not faces:
            return None, None
        face = faces[0]
        bb = bboxes[0]
        x1, y1, x2, y2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        w_img, h_img = img.size
        box_info = (x1, y1, x2, y2, w_img, h_img)
        return face, box_info
    except Exception as e:
        print('Face detection Failed due to error.')
        print(e)
        return None, None


