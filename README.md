# AI 驱动的竞品分析 Agent 协作系统

> 4 个专职 Agent + LangGraph DAG + Doubao 语义推理 + 闭环质检 + 全链路可观测。
> 端到端：**数据采集 → Agent 编排 → 知识存储 → FastAPI 后端 → Next.js 前端**。

---

## 1. 一图看懂

```mermaid
flowchart TD
    A[Next.js 前端<br/>(http://127.0.0.1:3000)<br/>任务创建 | DAG 可视化 | 回放动画 | Trace 实时流 | 决策路径展示] -->|/api/* (rewrite)| B[FastAPI 后端<br/>(http://127.0.0.1:8000)<br/>POST /api/tasks | GET /api/tasks/{id}/{trace|replay|stream}]
    B --> C[Orchestrator (LangGraph StateGraph)<br/>collect → analyze → qa_analyst → write → qa_writer → END]
    C --> D[Providers: <br/>LLM(Doubao|Mock) | Search(Mock|可换 Tavily)<br/>Storage: SQLite traces/messages/tasks | data/output/<br/>Reliability: 重试 | 自一致性 | 分片 | 自评估 | 决策路径 trace]

    %% 条件打回（≤3 轮，含改善判定）
    C -->|条件打回（≤3 轮）| C
2. 当前能力清单（对齐评分维度）
多 Agent 协作与输出可信度（35%）
 4 个角色边界清晰：Collector / Analyst / Writer / QA，docstring 明确 “不做的事”
 LangGraph DAG，含 conditional edges 与闭环
 结构化消息传递：AgentMessage 强类型，intent / agent 都是 Literal 枚举，禁止裸字符串
 真闭环伪闭环：Agent 失败 → 路由 Collector/Analyst → orchestrator 做 _check_improvement 改善判定
 Schema 强约束：所有 citations 字段 min_length=1，无引用直接拒收
 信息溯源完整：每个 atomic claim 携带 source_url + snippet + fetched_at + confidence
技术深度与工程完整度（25%）
 端到端 5 段链路全通：采集 → 编排 → SQLite 存储 → FastAPI → Next.js
 可观测性达标：trace 表记录 prompt / 输入 / 输出 /tokens/latency /status/ 决策原因
 上下文管理：证据 > SHARD_THRESHOLD 时按 topic 分片，分别推 SWOT 后合并
 错误恢复：
网络 / 限流 / 超时异常 → 指数退避重试（最多 3 次，含抖动）
LLM 全失败 → 自动降级规则版 SWOT
QA 闭环 ≥3 轮 → 标记 needs_human
 幻觉抑制：
引用强制（Schema 拒收无 citation 的结论）
LLM 引用的 URL 必须在原始证据列表中，否则丢弃
 自一致性校验：同 prompt N 次采样投票，少数派被丢弃
 Agent 自评估：completeness/citation_density/reasoning_quality < 0.6 → 视为失败
 前瞻性思考：
防伪闭环改善判定（diff 旧 / 新产物，覆盖率 < 80% 强制再次打回）
Agent 自评估三维打分
LLM 失败可降级到规则版（不中断主流程）
业务价值与产品体验（20%）
 配置即可换竞品 / 换行业（CLI 或前端表单）
 报告 + 溯源 + 决策路径 全部可视化
 闭环触发次数 / 角色调用分布 /token 消耗 在前端实时可见
 三种切换：--llm mock（演示稳定）/ --llm doubao（真实推理）
代码质量与文档（10%）
 模块化目录、Pydantic 类型完整、关键逻辑注释
 README + architecture.md + agent-protocol.md
 15 个 pytest 全通过（Schema / 闭环 / 消息 / 重试 / 分片 / 自评估 / 决策路径）
合规、材料与答辩（10%）
 MVP 用本地 Mock 数据集，URL 来自产品公开页面（已声明）
 .env + .gitignore 隔离凭据
 SearchProvider 抽象预留 Tavily 接入位（合规友好）
3. 快速开始
后端
bash
运行
cd backend
python3 -m venv .env   # 或 conda create -n competitor python=3.12
.env/bin/pip install -r requirements.txt

# 配置凭据
cp .env.example .env
# 编辑 .env，填入 DOUBAO_API_KEY / DOUBAO_ENDPOINT

# 启动 FastAPI
.env/bin/python -m app.api.server
# → http://127.0.0.1:8000（访问 /docs 查看 API）
前端
bash
运行
cd frontend
pnpm install --registry https://registry.npmmirror.com
pnpm dev
# → http://127.0.0.1:3000
CLI（无需前端，演示更稳）
bash
运行
cd backend
# 默认走 Doubao（已加载 .env）
.env/bin/python -m app.cli run \
  --industry "企业协同办公" \
  --competitors "飞书,钉钉,企业微信"

# 临时切 Mock（不消耗 token、不依赖网络）
.env/bin/python -m app.cli run --llm mock --industry ... --competitors ...

# 查 trace（含决策原因）
.env/bin/python -m app.cli trace --task-id <uuid>

# 回放动态路径
.env/bin/python -m app.cli replay --task-id <uuid>

# 导出系统静态 DAG
.env/bin/python -m app.cli graph --format mermaid
4. 演示效果（典型一次 run）
Trace（含决策路径）
text
r0  qa       ok   0ms  issues=3, pass=False        ← 缺 personas
r0  orchestrator ok   0ms → collect (high severity → 打回 Collector)
r1  collector ok   0ms  evidence_count=25          ← 二次采集补齐
r1  analyst ...  llm_used=True
r1  orchestrator ok   0ms  {pass:true, coverage:1.0, note:improved} → 改善判定
r1  qa       ok   0ms  issues=0, pass=True
r0  orchestrator ok   0ms → write（QA 通过，进入 Writer）
r0  writer   ok   0ms  markdown_len=3212, refs=24
r0  qa       ok   0ms  issues=0, pass=True
r0  orchestrator ok   0ms → end（QA 通过，报告完成）
前端
text
左        中                    右
启动表单  系统 DAG（mermaid）    trace 实时流
任务列表  实际执行路径回放（带闭环） 决策路径高亮
          Markdown 报告           token/延迟统计
5. 项目结构
text
competitor-agent/
├─ backend/
│  ├─ app/
│  │  ├─ agents/              # 4 个 Agent（含 retry/分片/自一致性/自评估）
│  │  │  ├─ base.py           # call_llm 含指数退避重试
│  │  │  ├─ analyst.py        # 分片 + 自一致性投票 + 自评估
│  │  │  ├─ collector.py / writer.py / qa.py
│  │  ├─ schemas/             # Pydantic + AgentMessage 协议
│  │  ├─ orchestrator/
│  │  │  ├─ graph.py          # LangGraph DAG + 决策原因落 trace
│  │  │  └─ visualize.py       # mermaid / ascii 导出
│  │  ├─ providers/          # LLM (Mock/Doubao) + Search (Mock)
│  │  ├─ storage/             # SQLite tasks/messages/traces
│  │  ├─ api/server.py        # FastAPI: REST + SSE
│  │  └─ cli.py               # CLI 入口 + .env 自动加载
│  ├─ data/mock/              # 离线证据（公开来源声明）
│  ├─ data/output/            # 报告产物
│  ├─ tests/                  # 15 个 pytest
│  ├─ .env / .env.example     # 凭据（git ignored）
│  └─ requirements.txt
├─ frontend/
│  ├─ app/                    # Next.js 14 App Router
│  ├─ components/Mermaid.tsx  # CDN 加载 mermaid（规避 webpack 兼容问题）
│  ├─ next.config.js          # /api/* rewrite 到 :8000
│  └─ package.json
└─ docs/
   ├─ architecture.md
   └─ agent-protocol.md
6. 接入真实搜索（可选）
SearchProvider 已抽象，新建 app/providers/tavily.py：
python
运行
from tavily import TavilyClient
from .search import SearchProvider

class TavilySearchProvider(SearchProvider):
    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key=api_key)
    def search(self, competitor, industry):
        res = self.client.search(f"{competitor} {industry} 功能 定价 评价", max_results=10)
        return [RawEvidence(...) for r in res["results"]]
CLI/API 注入位仅 1 行即可切换。
7. 后续路线图
 Tavily 真实搜索接入（接口已预留）
 Langfuse 集成（trace 已就绪，按其 API 二次上报即可）
 人工介入面板（needs_human 状态下编辑产物再跑链）
 动态 Schema 演化（领域人员通过 UI 改 schema 不重启）