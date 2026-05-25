# AdaFace_Recognition

基于 [AdaFace](https://github.com/mk-minchul/AdaFace) 的开放集人脸识别系统：**Flask 后端 + 微信小程序前端**，支持动态增删人员，附带 5 策略多模板消融评估体系（ROC / EER / TAR\@FAR）。

> 本科期末项目，fork 自 [Violet52014/AdaFace_Recognition](https://github.com/Violet52014/AdaFace_Recognition)。

---

## 仓库结构

仓库一根目录里同时放了**小程序前端**（微信工具链强制根目录）和 **Python 后端**两套东西，初看会有点乱。按"谁负责"分组之后是这样：

```
AdaFace_Recognition/
│
├── 🟦 微信小程序前端（上游 fork 自带，微信开发者工具直接打开根目录）
│   ├── app.js / app.json / app.wxss     # 小程序入口 + 全局配置
│   ├── pages/                           # 页面：拍照识别 + 记录
│   │   ├── index/                       #   首页：拍照 → 调 /api/face/recognize
│   │   └── logs/                        #   识别记录页
│   ├── images/                          # 小程序图标资源
│   ├── utils/api.js                     # BASE_URL 配置
│   ├── project.config.json              # 微信工具项目配置
│   ├── project.private.config.json      # 个人本地配置
│   └── sitemap.json                     # 微信收录配置
│
├── 🟩 Python 后端
│   └── backend/
│       ├── AdaFace/                     # 上游：AdaFace 模型源码 + MTCNN 对齐（**禁止改动**）
│       ├── app/                         # Flask 应用代码
│       │   ├── __init__.py              #   create_app + Blueprint 装配
│       │   ├── config.py                #   上游：环境变量驱动的旧配置
│       │   ├── extensions.py            #   SQLAlchemy / CORS 实例
│       │   ├── models.py                #   ORM：FaceProfile / Template / RecognitionRecord
│       │   ├── routes/                  #   HTTP 端点（face / records / stats）
│       │   ├── services/                #   上游：face_service / adaface_infer / image_utils
│       │   ├── strategies/              #   5 个 TemplateStrategy 实现
│       │   ├── evaluation/              #   离线评估：data_split / metrics / pair_generator / runner
│       │   ├── application/             #   用例编排层（注册 / 识别）
│       │   └── utils/response.py
│       ├── scripts/                     # 一次性脚本（建库 / 迁移 / LFW 子集）
│       ├── tests/                       # 测试金字塔（unit / integration）
│       ├── models/                      # AdaFace 权重 .pth（gitignore，不入仓）
│       ├── dataset/                     # 私照 + LFW 缓存（gitignore，不入仓）
│       ├── instance/                    # SQLite 数据库（gitignore，不入仓）
│       ├── run.py                       # Flask 启动入口
│       └── requirements.txt             # uv 不可用时的 fallback 依赖列表
│
├── 🟨 依赖与配置（仓库根）
│   ├── pyproject.toml                   # uv 项目声明
│   ├── uv.lock                          # uv 版本锁
│   ├── .venv/                           # uv 创建的虚拟环境（gitignore）
│   ├── config.yaml                      # 业务配置：阈值 / 模型路径 / 数据集路径
│   └── .gitignore
│
├── 🟪 文档与产物
│   ├── docs/superpowers/                # spec（设计文档）+ plan（实施计划）
│   └── reports/                         # 评估输出：ablation.csv 等（gitignore）
│
└── README.md / CLAUDE.md
```

为什么前端要放根目录？因为**微信开发者工具的项目根 = 小程序根**，`project.config.json` 和 `app.json` 必须在根。这是上游就这么设计的，挪不了。

---

## 快速开始

### 1. 后端

```bash
# 一次性：装 uv（包管理器）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 在仓库根目录执行：
uv sync                            # 创建 .venv 并按 uv.lock 装齐依赖
uv run python backend/run.py       # 启动 Flask，默认 http://127.0.0.1:5000

# 健康检查
curl http://127.0.0.1:5000/api/health
```

> 装 uv 失败时的 fallback：`pip install -r backend/requirements.txt`（不锁版本，不保证一致）。

### 2. 模型权重

AdaFace 预训练权重 `.pth` 体积大，**不入仓**。需自行下载放到 `backend/models/`，启动时会自动选其中一个 `.ckpt/.pth/.pt`。
也可通过环境变量 `ADAFACE_MODEL_PATH=/abs/path/to/weight.ckpt` 指定绝对路径覆盖。

### 3. 数据集

`backend/dataset/<人名>/img_xxx.jpg` 放 35 个目标人脸，每人 ≥ 20 张。
`backend/dataset/lfw/<姓名>/...` 放外部 impostor（用 `backend/scripts/prepare_lfw_subset.py` 从 LFW 抽）。

### 4. 离线建库 + 评估

```bash
# 建库（5 策略 Template 写入 SQLite）
uv run python backend/scripts/build_face_gallery.py \
    --dataset backend/dataset/ --config config.yaml

# 5 策略消融评估（输出 ROC / EER / TAR@FAR=1e-3 / AUC）
uv run python -m app.evaluation.run_ablation \
    --dataset backend/dataset --config config.yaml
```

### 5. 跑测试

```bash
uv run pytest                       # 全部（默认排除 slow 集成测试）
uv run pytest backend/tests/unit/   # 仅单元测试
uv run pytest -m slow               # 仅集成测试（需权重 + 数据集）
```

### 6. 前端

用**微信开发者工具**打开仓库根目录即可。`utils/api.js` 里的 `BASE_URL` 默认指向 `http://127.0.0.1:5000/api`，真机调试需改为电脑局域网 IP。

---

## 后端 API 契约

| 方法 | 路径 | 用途 |
|---|---|---|
| `POST` | `/api/face/recognize` | 提交一张人脸图，返回最相似 person + 相似度 |
| `GET`  | `/api/records/list`   | 历史识别记录（分页） |
| `GET`  | `/api/stats/today`    | 今日识别统计 |
| `POST` | `/api/records/clear`  | 清空记录 |
| `GET`  | `/api/health`         | 健康检查 |

返回统一格式：`{ "code": 0, "message": "success", "data": {...} }`。

---

## 学术目标

5 个模板生成策略对比：

| 策略 | 描述 |
|---|---|
| `random_one` | 随机选一张作模板 |
| `mean_all` | 全部向量求均值 |
| `manual_three` | 手工分 frontal/left/right 三个姿态各取均值 |
| `kmeans_k3` | KMeans 聚 3 类，每类中心做模板 |
| `all_vectors` | 不压缩，保留全部向量 |

匹配方式：max-cosine over templates，对 35 人内部 + LFW 外部 impostor 出 ROC，报 EER / TAR\@FAR=1e-3 / AUC。

---

## 文档

- 设计文档（spec）：`docs/superpowers/specs/`
- 实施计划（plan）：`docs/superpowers/plans/`
- AI 协作指令：`CLAUDE.md`（项目级指令，入仓共享）
