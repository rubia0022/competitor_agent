"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type MermaidProps = {
  chart: string;
};

declare global {
  interface Window {
    mermaid?: {
      initialize: (cfg: any) => void;
      run: (cfg: { nodes: NodeListOf<Element> }) => Promise<void>;
    };
  }
}

let mermaidScriptPromise: Promise<void> | null = null;

function loadMermaidScript(): Promise<void> {
  if (mermaidScriptPromise) return mermaidScriptPromise;
  mermaidScriptPromise = new Promise((resolve, reject) => {
    if (typeof window === "undefined") return resolve();
    if (window.mermaid) return resolve();
    const existing = document.querySelector<HTMLScriptElement>(
      'script[data-mermaid="1"]'
    );
    if (existing) {
      existing.addEventListener("load", () => resolve());
      existing.addEventListener("error", () => reject(new Error("mermaid load error")));
      return;
    }
    const s = document.createElement("script");
    s.dataset.mermaid = "1";
    s.src = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js";
    s.async = true;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("mermaid load error"));
    document.head.appendChild(s);
  });
  return mermaidScriptPromise;
}

export function Mermaid({ chart }: MermaidProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [ready, setReady] = useState(false);

  const normalized = useMemo(() => chart.trim(), [chart]);

  useEffect(() => {
    let canceled = false;
    loadMermaidScript()
      .then(() => {
        if (canceled) return;
        if (window.mermaid) {
          window.mermaid.initialize({
            startOnLoad: false,
            securityLevel: "loose",
            theme: "dark",
          });
        }
        setReady(true);
      })
      .catch(() => setReady(false));
    return () => {
      canceled = true;
    };
  }, []);

  useEffect(() => {
    if (!ready) return;
    if (!ref.current) return;
    if (!window.mermaid) return;
    window
      .mermaid!.run({ nodes: ref.current.querySelectorAll(".mermaid") })
      .catch(() => {});
  }, [ready, normalized]);

  if (!normalized) return null;

  return (
    <div ref={ref}>
      <div className="mermaid">{normalized}</div>
    </div>
  );
}

