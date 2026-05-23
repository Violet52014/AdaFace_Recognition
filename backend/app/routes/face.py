from flask import Blueprint, request

from ..extensions import db
from ..models import RecognitionRecord
from ..services.face_service import recognize_face_from_base64, recognize_face_from_image_bytes
from ..utils.response import api_error, api_success


face_bp = Blueprint("face", __name__)


@face_bp.post("/face/recognize")
def recognize_face():
    result = None

    # 1) multipart/form-data：字段名 image 或 file（小程序 wx.uploadFile 常用 image）
    if request.files:
        upload = request.files.get("image") or request.files.get("file")
        if upload is not None:
            raw = upload.read()
            if not raw:
                return api_error(message="上传文件为空")
            try:
                result = recognize_face_from_image_bytes(raw)
            except Exception as err:
                return api_error(message=f"图片处理失败: {err}")

    # 2) application/json：{ "image": "<base64>" }
    if result is None:
        payload = request.get_json(silent=True) or {}
        image_base64 = payload.get("image", "")
        if not image_base64:
            return api_error(message="缺少图片：请使用 multipart 字段 image/file，或 JSON 字段 image(Base64)")
        try:
            result = recognize_face_from_base64(image_base64)
        except Exception as err:
            return api_error(message=f"图片处理失败: {err}")

    record = RecognitionRecord(
        name=result.name,
        status="success" if result.recognized else "error",
        class_name=result.class_name,
        avatar=result.avatar,
        description=result.message,
    )
    db.session.add(record)
    db.session.commit()

    return api_success(
        data={
            "recognized": result.recognized,
            "name": result.name,
            "message": result.message,
            "class": result.class_name or "",
            "avatar": result.avatar or "",
            "results": [
                {
                    "recognized": result.recognized,
                    "name": result.name,
                    "score": None,
                    "bbox": result.bbox,
                    "label": result.name,
                }
            ]
            if result.bbox
            else [],
            "count": 1 if result.bbox else 0,
        }
    )
