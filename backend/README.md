# 人脸识别项目后端（Flask）

该后端已对齐你当前小程序前端调用的 4 个接口：

- `POST /api/face/recognize`
- `GET /api/records/list`
- `GET /api/stats/today`
- `POST /api/records/clear`

## 技术栈

- Web 框架：Flask
- 数据库：SQLite（开发）
- ORM：SQLAlchemy
- 图像处理：OpenCV、Pillow
- 深度学习：已预留占位函数（后续可替换）

## 快速启动

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
python run.py
```


"D:\homework_ex\深度学习视觉实战\人脸识别项目\backend\requirements.txt''


默认监听：`http://127.0.0.1:5000`
健康检查：`GET /api/health`

## 前端联调

将前端 `utils/api.js` 中的 `BASE_URL` 改为：

```js
const BASE_URL = 'http://127.0.0.1:5000/api'
```

如果在真机调试，需要改成你的电脑局域网 IP 并配置小程序合法域名策略。

## 目录结构

```text
backend/
  app/
    routes/
      face.py
      records.py
      stats.py
    services/
      image_utils.py
      face_service.py
    utils/
      response.py
    config.py
    extensions.py
    models.py
    __init__.py
  requirements.txt
  run.py
```

## 接口返回规范

统一格式：

```json
{
  "code": 0,
  "message": "success",
  "data": {}
}
```

错误时 `code != 0`，`message` 描述错误原因。

## 深度学习接口预留点

文件：`app/services/face_service.py`

- `detect_faces(...)`：当前使用 OpenCV HaarCascade，占位可替换为 DNN/YOLO/MTCNN。
- `_placeholder_embed(...)`：当前返回随机向量，占位可替换为 ArcFace/FaceNet 等 embedding。
- `_placeholder_match(...)`：当前为简单阈值逻辑，占位可替换为余弦相似度 + 向量库检索。

你只需要替换这些函数，其他 API 与数据库逻辑可保持不变。
