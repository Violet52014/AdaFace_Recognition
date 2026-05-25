from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from ..config import Config
from ..models import FaceProfile, Template, parse_template_vectors
from .adaface_infer import extract_embedding_from_bgr, is_adaface_available, parse_stored_embedding
from .image_utils import decode_base64_image, decode_image_bytes


@dataclass
class FaceRecognitionResult:
    recognized: bool
    name: str
    message: str
    class_name: Optional[str] = None
    avatar: Optional[str] = None
    bbox: Optional[dict] = None


def _load_profiles_from_db():
    return FaceProfile.query.order_by(FaceProfile.id.asc()).all()


def detect_faces_with_boxes(image_bgr: np.ndarray) -> list:
    """OpenCV Haar：保留函数便于对照实验，当前识别主流程已不再调用。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    return [] if faces is None else faces.tolist()


def _placeholder_embed(_image_bgr: np.ndarray):
    return np.random.rand(128)


def _placeholder_match(feature: np.ndarray, _profiles):
    if feature.mean() > 0.5:
        return 0, 0.78
    return None, 0.32


def _match_with_gallery(
    query: np.ndarray, profiles: list
) -> Tuple[Optional[FaceProfile], float]:
    """query 与库中向量均为 L2 归一化时，点积即余弦相似度。"""
    best_p = None
    best_score = -1.0
    for p in profiles:
        vec = parse_stored_embedding(p.feature_vector)
        if vec is None or vec.size == 0:
            continue
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        q = query / (np.linalg.norm(query) + 1e-8)
        score = float(np.dot(q, vec))
        if score > best_score:
            best_score = score
            best_p = p
    return best_p, best_score


def recognize_face_from_image_bytes(image_bytes: bytes) -> FaceRecognitionResult:
    image_bgr = decode_image_bytes(image_bytes)
    return recognize_face_from_bgr(image_bgr)


def recognize_face_from_base64(image_base64: str) -> FaceRecognitionResult:
    image_bgr = decode_base64_image(image_base64)
    return recognize_face_from_bgr(image_bgr)


def recognize_face_from_bgr(image_bgr: np.ndarray) -> FaceRecognitionResult:
    profiles = _load_profiles_from_db()

    if is_adaface_available():
        emb, err, bbox = extract_embedding_from_bgr(image_bgr)
        if err or emb is None:
            return FaceRecognitionResult(
                recognized=False,
                name="未识别",
                message=err or "特征提取失败",
                bbox=bbox,
            )

        with_vectors = [p for p in profiles if p.feature_vector]
        if not with_vectors:
            return FaceRecognitionResult(
                recognized=False,
                name="未识别",
                message="人脸库为空：请先在服务器用数据集脚本写入特征（见 backend/scripts/build_face_gallery.py）",
                bbox=bbox,
            )

        from ..config_loader import AppConfig
        # 配置文件路径相对仓库根；run.py 启动时 cwd=backend，因此 ../config.yaml
        cfg = AppConfig.load(Path(__file__).resolve().parents[3] / "config.yaml")
        threshold = cfg.recognition.match_threshold
        best_p, score = match_with_templates(
            emb,
            profiles,  # 含全部 profile，由 match_with_templates 内部过滤
            strategy=cfg.recognition.production_strategy,
        )

        if best_p is None or score < threshold:
            return FaceRecognitionResult(
                recognized=False,
                name="未识别",
                message=f"未识别到库中人员（最高相似度 {score:.3f}，阈值 {threshold}）",
                bbox=bbox,
            )

        return FaceRecognitionResult(
            recognized=True,
            name=best_p.name,
            class_name=best_p.class_name,
            avatar=best_p.avatar,
            message=f"识别成功，欢迎 {best_p.name}（相似度 {score:.3f}）",
            bbox=bbox,
        )

    # 未配置模型：沿用占位逻辑便于联调（无 AdaFace 时无 MTCNN bbox）
    if not profiles:
        profiles = [
            FaceProfile(name="张三", class_name="高三(1)班", avatar="/images/default-avatar.png"),
            FaceProfile(name="李四", class_name="高三(2)班", avatar="/images/default-avatar.png"),
        ]

    feature = _placeholder_embed(image_bgr)
    matched_index, score = _placeholder_match(feature, profiles)

    if matched_index is None or matched_index >= len(profiles):
        return FaceRecognitionResult(
            recognized=False,
            name="未识别",
            message=f"未识别到本班同学(置信度 {score:.2f})，请重试",
            bbox=None,
        )

    person = profiles[matched_index]
    return FaceRecognitionResult(
        recognized=True,
        name=person.name,
        class_name=person.class_name,
        avatar=person.avatar,
        message=f"识别成功，欢迎 {person.name} (置信度 {score:.2f})",
        bbox=None,
    )


def match_with_templates(
    query: np.ndarray,
    profiles: list,
    strategy: str,
) -> Tuple[Optional[FaceProfile], float]:
    """对每个 profile 取该 strategy 的 Template，计算 max-cosine；返回最佳 profile + 分数。

    无任何 profile 拥有该策略模板时返回 (None, -1.0)。
    query 与模板均假定已 L2 归一化（点积 = 余弦）。
    """
    q = query / (np.linalg.norm(query) + 1e-8)
    best_profile: Optional[FaceProfile] = None
    best_score = -1.0
    for p in profiles:
        tpl = next((t for t in p.templates if t.strategy == strategy), None)
        if tpl is None:
            continue
        mat = parse_template_vectors(tpl.vector_json)
        if mat.shape[0] == 0:
            continue
        # 模板再保险 L2 一次
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        mat_l2 = mat / norms
        score = float((mat_l2 @ q).max())
        if score > best_score:
            best_score = score
            best_profile = p
    return best_profile, best_score
