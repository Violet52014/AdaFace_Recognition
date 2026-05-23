from datetime import datetime, timedelta

from flask import Blueprint

from ..models import RecognitionRecord
from ..utils.response import api_success


stats_bp = Blueprint("stats", __name__)


@stats_bp.get("/stats/today")
def today_stats():
    now = datetime.now()
    day_start = datetime(now.year, now.month, now.day)
    day_end = day_start + timedelta(days=1)

    query = RecognitionRecord.query.filter(
        RecognitionRecord.recognized_at >= day_start,
        RecognitionRecord.recognized_at < day_end,
    )
    total = query.count()
    success = query.filter(RecognitionRecord.status == "success").count()
    rate = round((success / total * 100), 2) if total else 0

    return api_success(data={"total": total, "success": success, "rate": rate})
