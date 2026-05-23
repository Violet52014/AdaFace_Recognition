from flask import jsonify


def api_success(data=None, message="success", http_status=200):
    payload = {"code": 0, "message": message, "data": data if data is not None else {}}
    return jsonify(payload), http_status


def api_error(message="error", code=1, data=None, http_status=200):
    payload = {"code": code, "message": message, "data": data if data is not None else {}}
    return jsonify(payload), http_status
