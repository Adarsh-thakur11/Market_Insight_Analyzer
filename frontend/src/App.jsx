import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

function cx(...classes) {
  return classes.filter(Boolean).join(" ");
}

function Chip({ children }) {
  return (
    <span className="inline-flex items-center rounded-full border border-neutral-700 bg-neutral-800 px-2.5 py-1 text-xs font-medium text-neutral-200">
      {children}
    </span>
  );
}

function ResultCard({ r }) {
  return (
    <div className="rounded-2xl border border-neutral-800 bg-neutral-900 p-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <div className="truncate text-base font-semibold text-white">
            {r.title || "(no title)"}
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            <Chip>theme: {r.theme}</Chip>
            <Chip>keyword: {r.keyword}</Chip>
            <Chip>date: {String(r.created_at).slice(0, 10)}</Chip>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Chip>score: {Number(r.score).toFixed(3)}</Chip>
          <a
            href={r.url}
            target="_blank"
            rel="noreferrer"
            className="rounded-xl border border-neutral-700 bg-neutral-800 px-3 py-2 text-xs font-semibold text-neutral-100 hover:bg-neutral-700"
          >
            Open
          </a>
        </div>
      </div>

      <p className="mt-4 text-base leading-7 text-neutral-200">
        {r.preview}
        {r.preview?.length >= 680 ? "…" : ""}
      </p>
    </div>
  );
}

function MessageBubble({ role, children }) {
  const isUser = role === "user";
  return (
    <div className={cx("flex w-full", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cx(
          "max-w-3xl rounded-3xl px-6 py-4 text-base leading-7",
          isUser
            ? "bg-white text-black"
            : "bg-neutral-900 text-white border border-neutral-800"
        )}
      >
        {children}
      </div>
    </div>
  );
}

export default function App() {
  const [themes, setThemes] = useState([]);
  const [themeFilter, setThemeFilter] = useState("");
  const [topK, setTopK] = useState(8);
  const [days, setDays] = useState(30);
  const [requirePain, setRequirePain] = useState(true);

  const [input, setInput] = useState("What are people complaining about?");
  const [loading, setLoading] = useState(false);

  const [messages, setMessages] = useState([
    {
      id: crypto.randomUUID(),
      role: "assistant",
      type: "text",
      content:
        "Ask me about market pain points, emerging themes, or complaints. I’ll search your RAG index and return the most relevant posts.",
    },
  ]);

  // ✅ NEW: pipeline running state
  const [pipelineRunning, setPipelineRunning] = useState(false);

  const endRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/themes`)
      .then((r) => r.json())
      .then(setThemes)
      .catch(() => {});
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const timeOptions = useMemo(
    () => [
      { label: "Last 7 days", value: 7 },
      { label: "Last 14 days", value: 14 },
      { label: "Last 30 days", value: 30 },
      { label: "All time", value: "all" },
    ],
    []
  );

  async function runSearch(userText) {
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: userText,
          top_k: Number(topK),
          days: days === "all" ? null : Number(days),
          theme_filter: themeFilter || null,
          require_pain: requirePain,
        }),
      });

      const data = await res.json();
      const results = data.results ?? [];

      const summary =
        results.length === 0
          ? "I couldn’t find matches with these filters. Try turning off “Require pain language” or widening the time window."
          : `Found ${results.length} relevant posts. Here are the top matches from your index:`;

      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          type: "results",
          summary,
          results,
        },
      ]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          type: "text",
          content: "Backend not reachable. Is FastAPI running on 127.0.0.1:8000?",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function onSend() {
    const text = input.trim();
    if (!text || loading) return;

    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "user", type: "text", content: text },
    ]);
    setInput("");
    await runSearch(text);
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  }

  // ✅ NEW: secure update function
  async function runPipelineWithAuth() {
  if (pipelineRunning) return;

  const password = window.prompt("Enter admin password:");
  if (!password) return;

  try {
    setPipelineRunning(true);

    const res = await fetch(`${API_BASE}/pipeline/run`, {
      method: "POST",
      headers: { "x-api-key": password },
    });

    const data = await res.json();

    if (!data.ok) {
      throw new Error("Failed to start pipeline");
    }

    const jobId = data.job_id;

    // Poll the backend every 2 seconds
    const interval = setInterval(async () => {
      const statusRes = await fetch(`${API_BASE}/pipeline/status/${jobId}`);
      const statusData = await statusRes.json();

      // Check if job is done
      if (statusData.ok && statusData.status !== "running") {
        clearInterval(interval);
        setPipelineRunning(false);

        if (statusData.status === "success") {
          alert("Pipeline finished successfully ✅");
        } else {
          alert("Pipeline finished with ERROR ❌\n" + statusData.log);
        }
      }
    }, 2000);

  } catch (err) {
    setPipelineRunning(false);
    alert("Error starting pipeline: " + err.message);
  }
  }

  const controlClass =
    "rounded-xl border border-neutral-800 bg-neutral-900 px-3 py-2 text-xs text-neutral-100 shadow-sm outline-none focus:border-neutral-600 focus:ring-4 focus:ring-neutral-900";

  return (
    <div className="min-h-screen bg-black">
      {/* Top bar */}
      <div className="sticky top-0 z-10 border-b border-neutral-800 bg-black/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
              Market Insight Analyser
            </div>
          </div>

          {/* Controls */}
          <div className="flex flex-wrap items-center gap-2">

            {/* ✅ NEW UPDATE BUTTON */}
            <button
              onClick={runPipelineWithAuth}
              disabled={pipelineRunning}
              className="rounded-xl border border-neutral-700 bg-neutral-800 px-3 py-2 text-xs font-semibold text-neutral-100 hover:bg-neutral-700 disabled:opacity-50"
            >
              {pipelineRunning ? "Updating…" : "Update Data"}
            </button>

            <select
              value={days}
              onChange={(e) => setDays(e.target.value)}
              className={controlClass}
              title="Time window"
            >
              {timeOptions.map((opt) => (
                <option key={String(opt.value)} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>

            <select
              value={themeFilter}
              onChange={(e) => setThemeFilter(e.target.value)}
              className={cx(controlClass, "w-44")}
              title="Theme"
            >
              <option value="">(Any theme)</option>
              {themes.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>

            <input
              type="number"
              min={3}
              max={20}
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
              className={cx(controlClass, "w-20")}
              title="Top K"
            />

            <label className="flex items-center gap-2 rounded-xl border border-neutral-800 bg-neutral-900 px-3 py-2 text-xs text-neutral-100 shadow-sm">
              <input
                type="checkbox"
                checked={requirePain}
                onChange={(e) => setRequirePain(e.target.checked)}
                className="h-4 w-4 rounded border-neutral-600"
              />
              Require pain
            </label>
          </div>
        </div>
      </div>

      {/* Chat area */}
      <div className="mx-auto max-w-6xl px-4 pt-8 pb-40">
        <div className="mx-auto max-w-3xl space-y-4">
          {messages.map((m) => {
            if (m.type === "text") {
              return (
                <MessageBubble key={m.id} role={m.role}>
                  {m.content}
                </MessageBubble>
              );
            }

            return (
              <div key={m.id} className="space-y-3">
                <MessageBubble role="assistant">
                  <div className="text-base text-neutral-100">{m.summary}</div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <Chip>Top K: {topK}</Chip>
                    <Chip>
                      Window: {days === "all" ? "All time" : `${days} days`}
                    </Chip>
                    <Chip>Theme: {themeFilter || "Any"}</Chip>
                    <Chip>Pain: {requirePain ? "On" : "Off"}</Chip>
                  </div>
                </MessageBubble>

                <div className="grid gap-3">
                  {m.results.map((r, idx) => (
                    <ResultCard key={idx} r={r} />
                  ))}
                </div>
              </div>
            );
          })}

          {loading && (
            <MessageBubble role="assistant">
              <div className="flex items-center gap-3">
                <div className="h-2 w-2 animate-bounce rounded-full bg-neutral-400" />
                <div
                  className="h-2 w-2 animate-bounce rounded-full bg-neutral-400"
                  style={{ animationDelay: "120ms" }}
                />
                <div
                  className="h-2 w-2 animate-bounce rounded-full bg-neutral-400"
                  style={{ animationDelay: "240ms" }}
                />
                <span className="text-base text-neutral-300">Searching…</span>
              </div>
            </MessageBubble>
          )}

          <div ref={endRef} />
        </div>
      </div>

      {/* Composer (ChatGPT-style) */}
      <div className="fixed bottom-0 left-0 right-0 border-t border-neutral-800 bg-black">
        <div className="mx-auto max-w-4xl px-6 py-6">
          <div className="flex items-end gap-4 rounded-3xl border border-neutral-700 bg-neutral-900 px-5 py-4 shadow-lg">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              placeholder="Message Market Insight Analyser..."
              className="flex-1 resize-none bg-transparent text-base text-white outline-none placeholder:text-neutral-500"
            />

            <button
              onClick={onSend}
              disabled={loading || !input.trim()}
              className="rounded-2xl bg-white px-5 py-2 text-sm font-semibold text-black transition hover:bg-neutral-200 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Send
            </button>
          </div>

          <div className="mt-3 text-center text-xs text-neutral-500">
            Press Enter to send • Shift+Enter for new line
          </div>
        </div>
      </div>
    </div>
  );
}