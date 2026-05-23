from flask import Blueprint, request

from ..extensions import db
from ..models import RecognitionRecord
from ..utils.response import api_error, api_success


records_bp = Blueprint("records", __name__)


@records_bp.get("/records/list")
def list_records():
    try:
        page = max(int(request.args.get("page", 1)), 1)
        size = max(min(int(request.args.get("size", 20)), 100), 1)
    except ValueError:
        return api_error(message="page/size 参数格式错误")

    query = RecognitionRecord.query.order_by(RecognitionRecord.recognized_at.desc())
    total = query.count()
    items = query.offset((page - 1) * size).limit(size).all()

    records = [item.to_api_dict() for item in items]
    has_more = page * size < total
    return api_success(data={"records": records, "hasMore": has_more, "total": total})


@records_bp.post("/records/clear")
def clear_records():
    deleted = RecognitionRecord.query.delete()
    db.session.commit()
    return api_success(data={"deleted": deleted}, message="清空成功")
