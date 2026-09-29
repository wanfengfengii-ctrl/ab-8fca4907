# 编钟残音 · 联合基频分析（bianzhong-analyzer）

博物馆修复师从一组带有环境杂峰的编钟残音中确认哪些峰属于同一口钟。本服务对
6–18 条按频率升序录入的十进制峰值，**联合**选择严格递增且不重复的泛音序号与
可剔除杂峰，使同一基频 f₀ 同时落入所有采用峰的容差区间，并按以下顺序择优：

1. 采用峰数最多；
2. 最大相对误差（|fᵢ − nᵢ·f₀| / fᵢ）最小；
3. 相对误差平方和最小；
4. 按录入顺序最早的序号方案（泛音序号元组字典序，再按采用峰下标元组）。

不会先逐峰就近归属泛音再拼接结论：搜索在每个分支维护所有已采用峰可行 f₀
窗口的交集，泛音序号与剔除决策在同一棵分支限界搜索树中共同决定。

## 运行（Docker Compose）

宿主机发布端口可通过 `WEB_PORT` 配置（默认 8080）：

```bash
WEB_PORT=9090 docker compose up --build
# 浏览器打开 http://localhost:9090
```

健康检查：`GET /health` → `{"status":"ok"}`（容器内 8000 端口带 HEALTHCHECK，
Compose 以可配置宿主机端口发布）。

## 一次性 verify 服务

`verify` 服务在 `web` 健康后启动，执行**代码测试**与 **API 冒烟**（健康端点、
一份可复原残音、一份无解残音），以退出码报告结果后自行退出：

```bash
docker compose up --build verify
echo "退出码: $?"        # 0 表示构建、测试、冒烟全部通过
docker compose logs verify
```

## 页面用法

- 录入 6–18 行峰值（频率、各自允许偏差），设置候选基频闭区间、最高泛音序号、
  最多可剔除杂峰数；
- “提交分析”后查看：采用峰 / 被剔除峰、对应泛音序号、统一基频、逐峰绝对与
  相对误差、最大相对误差与平方和；
- 无解时**原始峰值原样保留**，并按峰值顺序高亮最早无法纳入任何剩余联合解释
  的证据峰（此前所有前缀均存在联合解释，到该峰为止的前缀联合搜索失败）。

## API

`POST /api/analyze`

```json
{
  "peaks": [
    {"frequency": 200.4, "tolerance": 1.5},
    {"frequency": 299.6, "tolerance": 1.5},
    {"frequency": 401.1, "tolerance": 1.5},
    {"frequency": 555.2, "tolerance": 1.5},
    {"frequency": 600.3, "tolerance": 1.5},
    {"frequency": 799.7, "tolerance": 1.5},
    {"frequency": 900.5, "tolerance": 1.5}
  ],
  "f0_min": 80, "f0_max": 130,
  "max_harmonic": 12, "max_rejected": 1
}
```

响应包含 `feasible`、统一 `f0`、逐峰 `peaks`（含 `adopted`/`harmonic`/
`expected`/`abs_error`/`rel_error`）、汇总误差；无解时 `f0` 为 `null`，
`first_unexplainable_index` 指向最早证据峰，所有原始频率原样返回。

## 本地开发（无 Docker）

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
python -m pytest tests -q
SMOKE_BASE_URL=http://127.0.0.1:8000 python scripts/smoke_api.py
```

## 算法要点

- 对峰 i 的候选序号 n，其可行 f₀ 窗口为
  [(fᵢ−tᵢ)/n, (fᵢ+tᵢ)/n] ∩ [f0_min, f0_max]；
- DFS 按录入顺序决定每峰「采用（选哪个序号）/剔除」，携带窗口交集，
  严格递增序号、剔除数上限、可达性剪枝与记忆化；
- 每个可行序号元组的最优 f₀ 从误差函数有限断点（窗口端点、误差 V 形曲线
  折点与对臂交点、SSE 唯一驻点）中精确选取，再按四级判优全局比较。
