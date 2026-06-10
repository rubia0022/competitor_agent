"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Mermaid } from "@/components/Mermaid";

// --- Types ---
type TaskSummary = {
  task_id: string;
  industry: string;
  competitors: string[];
  status: string;
  rounds: number;
  created_at: string | null;
  finished_at: string | null;
};

type TraceRow = {
  id: number;
  task_id: string;
  agent: string;
  round_no: number;
  prompt: string;
  input_payload: string;
  output_payload: string;
  tokens_in: number;
  tokens_out: number;
  latency_ms: number;
  status: string;
  error_msg: string | null;
  created_at: string;
};

type ExpandedTraceSet = Set<number>;

type ReplayData = {
  stats: {
    total_messages: number;
    rounds: number[];
    loop_triggered: number;
    by_agent: Record<string, number>;
    by_intent: Record<string, number>;
  };
  mermaid: string;
  ascii: string;
};

type ReportData = {
  task_id: string;
  industry: string;
  competitors: string[];
  status: string;
  rounds: number;
  report: {
    title: string;
    markdown: string;
    competitors: any[];
  } | null;
};

// --- Helpers ---
const fmt = (n: number) => (n > 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`);
const statusChip = (s: string) => {
  if (s === "done") return "chip chip-ok";
  if (s === "running") return "chip chip-info";
  if (s === "needs_human") return "chip chip-warn";
  return "chip chip-err";
};
const agentColor = (a: string) => {
  const m: Record<string, string> = {
    collector: "chip-info",
    analyst: "chip-warn",
    writer: "chip-ok",
    qa: "chip-muted",
    orchestrator: "chip-err",
  };
  return `chip ${m[a] || "chip-muted"}`;
};

// --- Main Page ---
export default function Home() {
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [staticGraph, setStaticGraph] = useState<string>("");
  const [replay, setReplay] = useState<ReplayData | null>(null);
  const [report, setReport] = useState<ReportData | null>(null);
  const [traces, setTraces] = useState<TraceRow[]>([]);
  const [creating, setCreating] = useState(false);
  const [expandedTraces, setExpandedTraces] = useState<ExpandedTraceSet>(
    new Set()
  );
  const [schemaJson, setSchemaJson] = useState<string>("");
  const [schemaOpen, setSchemaOpen] = useState(false);
  const [schemaSaving, setSchemaSaving] = useState(false);
  const [taskState, setTaskState] = useState<any>(null);
  const [editState, setEditState] = useState<string>("");
  const [resuming, setResuming] = useState(false);
  const [metrics, setMetrics] = useState<any>(null);

  const toggleTrace = useCallback((id: number) => {
    setExpandedTraces((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }, []);

  // Form state
  const [industry, setIndustry] = useState("企业协同办公");
  const [competitorsRaw, setCompetitorsRaw] = useState("飞书,钉钉,企业微信");
  const [llmKind, setLlmKind] = useState<"mock" | "doubao">("mock");

  // Load task list
  const refreshTasks = useCallback(async () => {
    const r = await fetch("/api/tasks?limit=20");
    if (r.ok) setTasks(await r.json());
  }, []);

  useEffect(() => {
    refreshTasks();
    fetch("/api/graph")
      .then((r) => r.json())
      .then((d) => setStaticGraph(d.mermaid));
    fetch("/api/schema")
      .then((r) => r.json())
      .then((d) => setSchemaJson(JSON.stringify(d, null, 2)));
    fetch("/api/metrics")
      .then((r) => r.json())
      .then((d) => setMetrics(d));
    const t = setInterval(refreshTasks, 3000);
    return () => clearInterval(t);
  }, [refreshTasks]);

  const saveSchema = async () => {
    try {
      JSON.parse(schemaJson);
    } catch {
      alert("JSON 格式错误");
      return;
    }
    setSchemaSaving(true);
    await fetch("/api/schema", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: schemaJson,
    });
    setSchemaSaving(false);
  };

  // --- needs_human 状态时加载保存的 state ---
  useEffect(() => {
    if (report?.status === "needs_human" && selectedId) {
      fetch(`/api/tasks/${selectedId}/state`)
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => {
          if (d) {
            setTaskState(d);
            setEditState(
              JSON.stringify(d.state?.competitors || [], null, 2)
            );
          }
        });
    } else {
      setTaskState(null);
      setEditState("");
    }
  }, [selectedId, report?.status]);

  const handleResume = async () => {
    if (!selectedId) return;
    let competitors;
    try {
      competitors = JSON.parse(editState);
    } catch {
      alert("JSON 格式错误，请检查后重试");
      return;
    }
    setResuming(true);
    try {
      await fetch(`/api/tasks/${selectedId}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ competitors }),
      });
      await refreshTasks();
      await loadTask(selectedId);
    } catch {
      alert("恢复失败");
    }
    setResuming(false);
  };

  // 选中任务 → 拉 report + replay + trace
  const loadTask = useCallback(async (id: string) => {
    setSelectedId(id);
    const [r1, r2, r3] = await Promise.all([
      fetch(`/api/tasks/${id}`).then((r) => r.json()),
      fetch(`/api/tasks/${id}/replay`).then((r) => (r.ok ? r.json() : null)),
      fetch(`/api/tasks/${id}/trace`).then((r) => (r.ok ? r.json() : [])),
    ]);
    setReport(r1);
    setReplay(r2);
    setTraces(r3 || []);
  }, []);

  // SSE 实时流 (任务进行中)
  useEffect(() => {
    if (!selectedId) return;
    if (report?.status && report.status !== "running") return;
    const es = new EventSource(`/api/tasks/${selectedId}/stream`);
    es.addEventListener("trace", (e: MessageEvent) => {
      try {
        const t = JSON.parse(e.data);
        setTraces((prev) => [...prev, t]);
      } catch {}
    });
    es.addEventListener("done", async () => {
      es.close();
      await loadTask(selectedId);
    });
    es.onerror = () => es.close();
    return () => es.close();
  }, [selectedId, report?.status, loadTask]);

  // 创建任务
  const submit = async () => {
    if (!industry || !competitorsRaw) return;
    setCreating(true);
    const r = await fetch("/api/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        industry,
        competitors: competitorsRaw
          .split(/[,，]/)
          .map((s) => s.trim())
          .filter(Boolean),
        llm: llmKind,
      }),
    });
    const data = await r.json();
    const sid = data.session_id;
    const tick = async () => {
      const sr = await fetch(`/api/sessions/${sid}`);
      const sd = await sr.json();
      if (sd.task_id) {
        setCreating(false);
        await refreshTasks();
        await loadTask(sd.task_id);
      } else {
        setTimeout(tick, 1500);
      }
    };
    setTimeout(tick, 1500);
  };

  // trace 实时滚动统计
  const traceStats = useMemo(() => {
    const byAgent: Record<string, { count: number; ms: number; tokens: number }> = {};
    let totalErr = 0;
    for (const t of traces) {
      const a = (byAgent[t.agent] = byAgent[t.agent] || {
        count: 0,
        ms: 0,
        tokens: 0,
      });
      a.count += 1;
      a.ms += t.latency_ms || 0;
      a.tokens += (t.tokens_in || 0) + (t.tokens_out || 0);
      if (t.status === "err") totalErr += 1;
    }
    return { byAgent, totalErr };
  }, [traces]);

  return (
    <main className="min-h-screen p-4 grid grid-cols-12 gap-4">
      {/* Header */}
      <header className="col-span-12 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">
            🤖 AI 竞品分析 Agent 协作系统
          </h1>
          <p className="text-xs opacity-60 mt-1">
            LangGraph DAG · Doubao · 多 Agent 协作 · 闭环质检 · 全程可追溯
          </p>
        </div>
        <a
          href="http://127.0.0.1:8000/docs"
          target="_blank"
          className="chip chip-info"
        >
          API Docs
        </a>
      </header>

      {/* Dashboard: 运营指标仪表盘 */}
      {metrics && (
        <details className="col-span-12 glass p-4">
          <summary className="text-sm font-semibold cursor-pointer opacity-80">
            📊 运营指标仪表盘
          </summary>
          <div className="mt-3 space-y-3">
            <div className="grid grid-cols-5 gap-2 text-xs">
              {[
                { label: "总任务", value: metrics.total_tasks, unit: "个" },
                {
                  label: "完成率",
                  value: `${(metrics.done_rate * 100).toFixed(1)}%`,
                },
                {
                  label: "人工修正率",
                  value: `${(metrics.needs_human_rate * 100).toFixed(1)}%`,
                },
                {
                  label: "闭环触发率",
                  value: `${(metrics.loop_trigger_rate * 100).toFixed(1)}%`,
                },
                { label: "平均轮次", value: metrics.avg_rounds, unit: "轮" },
              ].map((m) => (
                <div key={m.label} className="bg-black/20 rounded-lg p-3 text-center">
                  <div className="opacity-50 mb-1">{m.label}</div>
                  <div className="text-lg font-semibold">
                    {m.value}
                    {m.unit && (
                      <span className="text-[10px] opacity-40 ml-0.5">
                        {m.unit}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
            <div className="grid grid-cols-4 gap-2 text-xs">
              {[
                {
                  label: "总 Token 消耗",
                  value:
                    metrics.total_tokens > 1000
                      ? `${(metrics.total_tokens / 1000).toFixed(1)}k`
                      : metrics.total_tokens,
                },
                {
                  label: "平均延迟",
                  value: `${(metrics.avg_latency_ms / 1000).toFixed(1)}s`,
                },
                {
                  label: "平均自评估分",
                  value:
                    metrics.avg_self_eval != null
                      ? `${(metrics.avg_self_eval * 100).toFixed(0) + "%"}`
                      : "N/A",
                },
                {
                  label: "引用覆盖率",
                  value:
                    metrics.avg_improvement_coverage != null
                      ? (metrics.avg_improvement_coverage * 100).toFixed(0) + "%"
                      : "N/A",
                },
              ].map((m) => (
                <div key={m.label} className="bg-black/20 rounded-lg p-3 text-center">
                  <div className="opacity-50 mb-1">{m.label}</div>
                  <div className="text-lg font-semibold">{m.value}</div>
                </div>
              ))}
            </div>
            <div className="bg-black/20 rounded-lg p-3">
              <div className="text-xs opacity-50 mb-2">Agent 调用分布</div>
              <div className="space-y-1.5">
                {(() => {
                  const dist = metrics.agent_distribution as Record<
                    string,
                    number
                  >;
                  const maxVal = Math.max(...Object.values(dist), 1);
                  return Object.entries(dist)
                    .sort(([, a], [, b]) => b - a)
                    .map(([agent, count]) => (
                      <div
                        key={agent}
                        className="flex items-center gap-2 text-xs"
                      >
                        <span className={agentColor(agent)}>{agent}</span>
                        <div className="flex-1 bg-black/30 rounded-full h-4 overflow-hidden">
                          <div
                            className="h-full rounded-full bg-indigo-400/50"
                            style={{
                              width: `${(count / maxVal) * 100}%`,
                            }}
                          />
                        </div>
                        <span className="w-12 opacity-50">{count}</span>
                      </div>
                    ));
                })()}
              </div>
            </div>
          </div>
        </details>
      )}

      {/* Schema config (动态演化) */}
      <section>
        <details open={schemaOpen} onToggle={(e) => setSchemaOpen((e.target as HTMLDetailsElement).open)}>
          <summary className="text-sm font-semibold mb-2 opacity-80 cursor-pointer">
            ⚙️ Schema 配置 (动态演化)
          </summary>
          <div className="space-y-2 text-xs">
            <textarea
              className="w-full h-40 px-2 py-1 rounded bg-black/30 border border-white/10 font-mono text-[10px] leading-relaxed"
              value={schemaJson}
              onChange={(e) => setSchemaJson(e.target.value)}
            />
            <button
              onClick={saveSchema}
              disabled={schemaSaving}
              className="w-full py-1 rounded bg-emerald-500/80 hover:bg-emerald-500 disabled:opacity-40 transition text-xs font-medium"
            >
              {schemaSaving ? "保存中…" : "💾 保存配置"}
            </button>
            <p className="opacity-40 text-[10px]">
              修改后立即生效，无需重启服务
            </p>
          </div>
        </details>
      </section>

      {/* Left: Task list + New */}
      <aside className="col-span-3 glass p-4 space-y-4 scroll-y" style={{ maxHeight: "calc(100vh - 96px)" }}>
        <section>
          <h2 className="text-sm font-semibold mb-2 opacity-80">新建分析任务</h2>
          <div className="space-y-2 text-sm">
            <input
              className="w-full px-2 py-1 rounded bg-black/30 border border-white/10"
              value={industry}
              onChange={(e) => setIndustry(e.target.value)}
              placeholder="行业"
            />
            <input
              className="w-full px-2 py-1 rounded bg-black/30 border border-white/10"
              value={competitorsRaw}
              onChange={(e) => setCompetitorsRaw(e.target.value)}
              placeholder="竞品 (逗号分隔)"
            />
            <div className="flex gap-2 items-center">
              <label className="text-xs opacity-60">LLM</label>
              <select
                value={llmKind}
                onChange={(e) => setLlmKind(e.target.value as any)}
                className="flex-1 px-2 py-1 rounded bg-black/30 border border-white/10 text-xs"
              >
                <option value="doubao">Doubao (真模型, 慢)</option>
                <option value="mock">Mock (演示规则, 快)</option>
              </select>
            </div>
            <button
              onClick={submit}
              disabled={creating}
              className="w-full py-1.5 rounded bg-indigo-500/80 hover:bg-indigo-500 disabled:opacity-40 transition text-sm font-medium"
            >
              {creating ? "创建中…" : "🚀 启动分析"}
            </button>
          </div>
        </section>

        <section>
          <h2 className="text-sm font-semibold mb-2 opacity-80">
            最近任务 ({tasks.length})
          </h2>
          <ul className="space-y-1.5">
            {tasks.map((t) => (
              <li
                key={t.task_id}
                onClick={() => loadTask(t.task_id)}
                className={`p-2 rounded cursor-pointer text-xs transition ${
                  selectedId === t.task_id
                    ? "bg-indigo-500/30 border border-indigo-400/40"
                    : "bg-black/20 hover:bg-black/30 border border-transparent"
                }`}
              >
                <div className="flex justify-between items-center mb-1">
                  <span className={statusChip(t.status)}>{t.status}</span>
                  <span className="opacity-50 text-[10px]">r.{t.rounds}</span>
                </div>
                <div className="font-medium truncate">{t.industry}</div>
                <div className="opacity-60 truncate">
                  {t.competitors.join(" · ")}
                </div>
                <div className="opacity-30 text-[10px] mt-0.5">
                  {t.task_id.slice(0, 8)}
                </div>
              </li>
            ))}
            {tasks.length === 0 && (
              <li className="text-xs opacity-40 p-4 text-center">
                还没有任务，点上面"启动分析"创建
              </li>
            )}
          </ul>
        </section>
      </aside>

      {/* Center: Static DAG + Report */}
      <section className="col-span-6 space-y-4 scroll-y" style={{ maxHeight: "calc(100vh - 96px)" }}>
        <details className="glass p-2 rounded-lg">
          <summary className="text-[11px] font-medium opacity-60 cursor-pointer">
            🧩 系统 DAG (点击展开)
          </summary>
          <div className="mt-1">
            {staticGraph ? (
              <Mermaid chart={staticGraph} />
            ) : (
              <div className="opacity-40 text-xs">加载中…</div>
            )}
          </div>
        </details>

        {replay && (
          <div className="glass p-4">
            <h2 className="text-sm font-semibold mb-2 opacity-80">
              🔁 实际执行路径回放
              {replay.stats.loop_triggered > 0 && (
                <span className="chip chip-warn ml-2">
                  闭环触发 {replay.stats.loop_triggered} 次
                </span>
              )}
            </h2>
            <Mermaid chart={replay.mermaid} />
            <div className="mt-3 grid grid-cols-4 gap-2 text-xs">
              {Object.entries(replay.stats.by_agent).map(([a], c) => (
                <div key={a} className="bg-black/20 rounded p-2">
                  <div className={agentColor(a)}>{a}</div>
                  <div className="text-lg font-semibold mt-1">{c}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {report?.report && (
          <div className="glass p-4">
            <h2 className="text-sm font-semibold mb-2 opacity-80">
              📄 报告产物
            </h2>
            <details open>
              <summary className="cursor-pointer text-xs opacity-60 mb-2">
                {report.report.title} ({report.report.markdown.length} chars)
              </summary>
              <pre className="text-xs whitespace-pre-wrap bg-black/30 p-3 rounded leading-relaxed">
                {report.report.markdown}
              </pre>
            </details>
          </div>
        )}

        {report?.status === "needs_human" && taskState && (
          <div className="glass p-4 border border-yellow-400/30">
            <h2 className="text-sm font-semibold mb-2 text-yellow-200/90">
              ⚠️ 需要人工介入
            </h2>
            <p className="text-xs opacity-60 mb-3">
              QA 质检在最大轮次后仍未通过 (失败节点: {taskState.failed_node})。请编辑数据后点击恢复。
            </p>
            {taskState.state?.qa_analyst7?.issues?.length > 0 && (
              <div className="mb-3">
                <h3 className="text-xs font-medium mb-1 opacity-80">质检问题:</h3>
                {taskState.state.qa_analyst.issues.map((issue: any, i: number) => (
                  <div
                    key={i}
                    className="text-[11px] bg-red-900/20 border border-red-400/20 rounded p-1.5 mb-1"
                  >
                    <span className="chip chip-err text-[10px]">
                      {issue.severity}
                    </span>
                    <span className="ml-2 opacity-80">
                      {issue.field_path}: {issue.reason}
                    </span>
                  </div>
                ))}
              </div>
            )}
            <div className="mb-3">
              <h3 className="text-xs font-medium mb-1 opacity-80">
                竞品数据 (可编辑 JSON)
              </h3>
              <textarea
                className="w-full h-56 px-2 py-1.5 rounded bg-black/30 border border-white/10 font-mono text-[10px] leading-relaxed"
                value={editState}
                onChange={(e) => setEditState(e.target.value)}
              />
            </div>
            <button
              onClick={handleResume}
              disabled={resuming}
              className="w-full py-1.5 rounded bg-yellow-500/80 hover:bg-yellow-500 disabled:opacity-40 transition text-sm font-medium"
            >
              {resuming ? "恢复中…" : "▶️ 恢复执行"}
            </button>
          </div>
        )}
      </section>

      {/* Right: trace + decisions */}
      <aside className="col-span-3 glass p-4 scroll-y space-y-3" style={{ maxHeight: "calc(100vh - 96px)" }}>
        <h2 className="text-sm font-semibold mb-2 opacity-80">
          🧵 Trace 实时流 ({traces.length})
          {traceStats.totalErr > 0 && (
            <span className="chip chip-err ml-2">{traceStats.totalErr}</span>
          )}
        </h2>
        <div className="grid grid-cols-2 gap-1 mb-3 text-[10px]">
          {Object.entries(traceStats.byAgent).map(([a, s]) => (
            <div key={a} className="bg-black/20 rounded px-2 py-1">
              <div className={agentColor(a)}>{a}</div>
              <div className="text-lg font-semibold mt-1">{s.count}</div>
            </div>
          ))}
        </div>
        <ul className="space-y-1">
          {traces.map((t) => {
            const isDecision = t.agent === "orchestrator";
            const isExpanded = expandedTraces.has(t.id);
            let decisionInfo: { decision?: string; reason?: string } = {};
            if (isDecision && t.output_payload) {
              try {
                decisionInfo = JSON.parse(t.output_payload);
              } catch {}
            }
            const rowBg =
              t.status === "err"
                ? "border-red-400/30 bg-red-900/20 hover:bg-red-900/30"
                : isDecision
                ? "border-indigo-400/30 bg-indigo-900/15 hover:bg-indigo-900/25"
                : "border-white/5 bg-black/20 hover:bg-black/30";
            return (
              <li key={t.id} className="mt-1">
                <div
                  onClick={() => toggleTrace(t.id)}
                  className={`text-[11px] p-2 rounded border cursor-pointer transition ${rowBg}`}
                >
                  <div className="flex justify-between items-center">
                    <span className={agentColor(t.agent)}>{t.agent}</span>
                    <span className="opacity-50 flex items-center gap-1">
                      r.{t.round_no} · {t.latency_ms}ms
                      <span className="opacity-60">{isExpanded ? "▼" : "▶"}</span>
                    </span>
                  </div>
                  {isDecision && (
                    <div className="mt-1">
                      <div className="text-indigo-300 font-medium">
                        {decisionInfo.decision}
                      </div>
                      <div className="opacity-60"> · {decisionInfo.reason}</div>
                    </div>
                  )}
                  <div
                    className={`opacity-70 mt-1 ${isExpanded ? "" : "line-clamp-2"}`}
                  >
                    {t.output_payload || "（无输出）"}
                  </div>
                </div>
                {isExpanded ? (
                  <div className="mt-1">
                    {t.prompt && (
                      <div className="opacity-60 mb-0.5">prompt</div>
                    )}
                    <pre className="whitespace-pre-wrap break-words bg-black/30 p-2 rounded opacity-80 max-h-48 overflow-y-auto scroll-y">
                      {t.prompt}
                    </pre>
                    {t.input_payload && (
                      <div className="opacity-60 mb-0.5">input</div>
                    )}
                    <pre className="whitespace-pre-wrap break-words bg-black/30 p-2 rounded opacity-80 max-h-48 overflow-y-auto scroll-y">
                      {t.input_payload}
                    </pre>
                    {t.output_payload && (
                      <div className="opacity-60 mb-0.5">output</div>
                    )}
                    <pre className="whitespace-pre-wrap break-words bg-black/30 p-2 rounded opacity-80 max-h-48 overflow-y-auto scroll-y">
                      {t.output_payload}
                    </pre>
                    {t.error_msg && (
                      <div className="text-red-300/70 mb-0.5">error</div>
                    )}
                    <pre className="whitespace-pre-wrap break-words bg-red-900/20 p-2 rounded text-red-200/80">
                      {t.error_msg}
                    </pre>
                  </div>
                ) : null}
              </li>
            );
          })}
          {traces.length === 0 && (
            <li className="text-xs opacity-40 text-center p-4">
              选择左侧任务查看 trace
            </li>
          )}
        </ul>
      </aside>
    </main>
  );
}
