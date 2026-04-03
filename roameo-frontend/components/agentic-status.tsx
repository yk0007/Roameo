"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { AgentTraceEvent } from "@/lib/types";

/* ─────────────────────────────────────────────
   Minimal shimmer text
───────────────────────────────────────────── */
function ShimmerText({
  children,
  className = "",
}: {
  children: string;
  className?: string;
}) {
  return (
    <span
      className={`inline-block bg-clip-text text-transparent ${className}`}
      style={{
        backgroundImage:
          "linear-gradient(90deg, #94a3b8 0%, #64748b 40%, #cbd5e1 60%, #94a3b8 100%)",
        backgroundSize: "200% 100%",
        animation: "shimmer 2s linear infinite",
      }}
    >
      {children}
    </span>
  );
}

/* ─────────────────────────────────────────────
   Stage → display config
   Maps planning state stages and trace agent names
   to human-readable labels + icon animations.
───────────────────────────────────────────── */

type AnimKind =
  | "pulse_dot"      // simple pulsing dot — generic thinking
  | "blink_cursor"   // blinking text cursor — typing / drafting
  | "spin_ring"      // spinning ring — network / fetch
  | "scan_lines"     // horizontal scan lines — reading / extraction
  | "wave_bars"      // bouncing bars — analysis / compute
  | "orbit"          // orbiting dots — planning / coordination
  | "writing_pen"    // pen stroke animation — writing output
  | "magnify"        // magnify pulse — searching / discovery
  ;

interface StageConfig {
  label: string;
  sub: string;
  anim: AnimKind;
  color: string; // tailwind text color class
}

function classifyTrace(trace: AgentTraceEvent | null): StageConfig {
  if (!trace) {
    return {
      label: "Thinking",
      sub: "Working through your request…",
      anim: "pulse_dot",
      color: "text-slate-500",
    };
  }

  const agent = trace.agent.toLowerCase();
  const label = trace.label.toLowerCase();

  // Tool / extraction calls
  if (agent.includes("tool") || label.includes("tool") || label.includes("fetch") || label.includes("lookup")) {
    return {
      label: trace.label || "Fetching data",
      sub: trace.detail || "Calling live data source…",
      anim: "spin_ring",
      color: "text-blue-500",
    };
  }

  // Search / discovery
  if (
    agent.includes("search") ||
    agent.includes("discovery") ||
    agent.includes("research") ||
    label.includes("search") ||
    label.includes("discover") ||
    label.includes("research") ||
    label.includes("finding")
  ) {
    return {
      label: trace.label || "Searching",
      sub: trace.detail || "Scanning destinations and places…",
      anim: "magnify",
      color: "text-violet-500",
    };
  }

  // Reading / extraction
  if (
    label.includes("read") ||
    label.includes("extract") ||
    label.includes("parse") ||
    label.includes("context") ||
    label.includes("pulling")
  ) {
    return {
      label: trace.label || "Reading context",
      sub: trace.detail || "Extracting relevant information…",
      anim: "scan_lines",
      color: "text-amber-500",
    };
  }

  // Planning / building
  if (
    agent.includes("planner") ||
    agent.includes("planning") ||
    label.includes("plan") ||
    label.includes("build") ||
    label.includes("itinerary") ||
    label.includes("schedul") ||
    label.includes("coordinat") ||
    label.includes("layout")
  ) {
    return {
      label: trace.label || "Building your plan",
      sub: trace.detail || "Coordinating the itinerary…",
      anim: "orbit",
      color: "text-emerald-500",
    };
  }

  // Writing / drafting output
  if (
    label.includes("writ") ||
    label.includes("draft") ||
    label.includes("craft") ||
    label.includes("generat") ||
    label.includes("compil") ||
    agent.includes("writer") ||
    agent.includes("narrator")
  ) {
    return {
      label: trace.label || "Crafting response",
      sub: trace.detail || "Writing a personalised answer…",
      anim: "writing_pen",
      color: "text-rose-500",
    };
  }

  // Analysis / evaluation
  if (
    label.includes("analys") ||
    label.includes("evaluat") ||
    label.includes("calculat") ||
    label.includes("scor") ||
    label.includes("rank")
  ) {
    return {
      label: trace.label || "Analysing options",
      sub: trace.detail || "Comparing and ranking results…",
      anim: "wave_bars",
      color: "text-cyan-500",
    };
  }

  // Subagent / worker
  if (agent.includes("worker") || agent.includes("subagent") || agent.includes("sub_agent")) {
    return {
      label: trace.label || "Running subagent",
      sub: trace.detail || "Delegating to a specialised agent…",
      anim: "orbit",
      color: "text-indigo-500",
    };
  }

  // Default: resolver / thinking
  return {
    label: trace.label || "Reasoning",
    sub: trace.detail || "Thinking through the next step…",
    anim: "pulse_dot",
    color: "text-slate-500",
  };
}

/* ─────────────────────────────────────────────
   Individual icon animations
───────────────────────────────────────────── */

function PulseDot({ color }: { color: string }) {
  return (
    <div className="relative flex h-5 w-5 items-center justify-center">
      <motion.div
        className={`h-2.5 w-2.5 rounded-full bg-current ${color}`}
        animate={{ scale: [1, 1.5, 1], opacity: [0.6, 1, 0.6] }}
        transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className={`absolute h-2.5 w-2.5 rounded-full bg-current ${color} opacity-30`}
        animate={{ scale: [1, 2.4, 1], opacity: [0.3, 0, 0.3] }}
        transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
      />
    </div>
  );
}

function BlinkCursor({ color }: { color: string }) {
  return (
    <div className="flex h-5 items-end pb-0.5">
      <motion.div
        className={`h-3.5 w-0.5 rounded-full bg-current ${color}`}
        animate={{ opacity: [1, 0, 1] }}
        transition={{ duration: 0.9, repeat: Infinity, ease: "linear" }}
      />
    </div>
  );
}

function SpinRing({ color }: { color: string }) {
  return (
    <div className="relative h-5 w-5">
      <motion.div
        className={`h-5 w-5 rounded-full border-2 border-current border-t-transparent ${color}`}
        animate={{ rotate: 360 }}
        transition={{ duration: 0.8, repeat: Infinity, ease: "linear" }}
      />
    </div>
  );
}

function ScanLines({ color }: { color: string }) {
  return (
    <div className={`flex h-5 w-5 flex-col justify-around overflow-hidden ${color}`}>
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          className="h-0.5 w-full rounded-full bg-current"
          animate={{ scaleX: [0, 1, 0], opacity: [0, 1, 0] }}
          transition={{
            duration: 1.2,
            repeat: Infinity,
            delay: i * 0.25,
            ease: "easeInOut",
          }}
          style={{ transformOrigin: "left" }}
        />
      ))}
    </div>
  );
}

function WaveBars({ color }: { color: string }) {
  return (
    <div className={`flex h-5 w-5 items-end justify-center gap-0.5 ${color}`}>
      {[0, 1, 2, 3].map((i) => (
        <motion.div
          key={i}
          className="w-1 rounded-t-full bg-current"
          animate={{ scaleY: [0.3, 1, 0.3] }}
          transition={{
            duration: 0.8,
            repeat: Infinity,
            delay: i * 0.12,
            ease: "easeInOut",
          }}
          style={{ height: "100%", transformOrigin: "bottom" }}
        />
      ))}
    </div>
  );
}

function OrbitDots({ color }: { color: string }) {
  return (
    <div className="relative h-5 w-5">
      <motion.div
        className="absolute inset-0"
        animate={{ rotate: 360 }}
        transition={{ duration: 1.6, repeat: Infinity, ease: "linear" }}
      >
        <div className={`absolute top-0 left-1/2 -translate-x-1/2 h-1.5 w-1.5 rounded-full bg-current ${color}`} />
      </motion.div>
      <motion.div
        className="absolute inset-0"
        animate={{ rotate: -360 }}
        transition={{ duration: 2.2, repeat: Infinity, ease: "linear" }}
      >
        <div className={`absolute bottom-0 left-1/2 -translate-x-1/2 h-1 w-1 rounded-full bg-current opacity-60 ${color}`} />
      </motion.div>
      <div className={`absolute inset-0 m-auto h-1.5 w-1.5 rounded-full bg-current opacity-40 ${color}`} />
    </div>
  );
}

function WritingPen({ color }: { color: string }) {
  return (
    <div className={`relative h-5 w-5 ${color}`}>
      <motion.svg viewBox="0 0 20 20" fill="none" className="h-5 w-5">
        <motion.path
          d="M3 14 L10 4 L14 8 L6 16 L3 17 Z"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: [0, 1, 0] }}
          transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.line
          x1="3"
          y1="17"
          x2="3"
          y2="17"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          animate={{ x2: [3, 7] }}
          transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
        />
      </motion.svg>
    </div>
  );
}

function MagnifyPulse({ color }: { color: string }) {
  return (
    <div className={`relative h-5 w-5 ${color}`}>
      <motion.svg viewBox="0 0 20 20" fill="none" className="h-5 w-5">
        <motion.circle
          cx="8" cy="8" r="5"
          stroke="currentColor"
          strokeWidth="1.8"
          fill="none"
          animate={{ r: [4, 5.5, 4] }}
          transition={{ duration: 1.3, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.line
          x1="12" y1="12" x2="17" y2="17"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 1.3, repeat: Infinity, ease: "easeInOut" }}
        />
      </motion.svg>
    </div>
  );
}

function AnimIcon({ kind, color }: { kind: AnimKind; color: string }) {
  switch (kind) {
    case "pulse_dot":   return <PulseDot color={color} />;
    case "blink_cursor": return <BlinkCursor color={color} />;
    case "spin_ring":   return <SpinRing color={color} />;
    case "scan_lines":  return <ScanLines color={color} />;
    case "wave_bars":   return <WaveBars color={color} />;
    case "orbit":       return <OrbitDots color={color} />;
    case "writing_pen": return <WritingPen color={color} />;
    case "magnify":     return <MagnifyPulse color={color} />;
  }
}

/* ─────────────────────────────────────────────
   Main component
───────────────────────────────────────────── */

export interface AgenticStatusProps {
  /** legacy mode prop — kept for backwards compat */
  mode?: "general" | "planning" | "worker";
  title?: string;
  detail?: string;
  steps?: Array<{
    label: string;
    detail?: string;
    state?: "queued" | "running" | "done" | "completed";
  }>;
  /** live trace events from the current turn */
  traces?: AgentTraceEvent[];
  /** current turn id — used to filter traces */
  turnId?: string;
}

export function AgenticStatus({
  mode = "general",
  title,
  detail,
  steps,
  traces = [],
  turnId,
}: AgenticStatusProps) {
  // Filter to traces for this turn if we have a turnId,
  // otherwise show all currently-running ones
  const turnTraces = turnId
    ? traces.filter((t) => t.turnId === turnId)
    : traces;

  // Pick the most recent running trace
  const runningTrace =
    [...turnTraces]
      .reverse()
      .find((t) => t.status === "running") ?? turnTraces.at(-1) ?? null;

  const config = classifyTrace(runningTrace);

  // If caller provides explicit title/detail, use those
  const resolvedLabel = title || config.label;
  const resolvedSub = detail || config.sub;

  // Legacy steps prop
  const activeSteps = (steps || []).slice(0, 5);

  return (
    <div className="space-y-2.5 py-1">
      {/* ── Primary animated row ── */}
      <AnimatePresence mode="wait">
        <motion.div
          key={resolvedLabel}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.25 }}
          className="flex items-center gap-2.5"
        >
          <AnimIcon kind={config.anim} color={config.color} />
          <ShimmerText className="text-[15px] font-medium">
            {resolvedLabel}
          </ShimmerText>
        </motion.div>
      </AnimatePresence>

      {/* ── Sub-label ── */}
      <AnimatePresence mode="wait">
        <motion.div
          key={resolvedSub}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3, delay: 0.05 }}
          className="pl-7 text-[13px] leading-5 text-slate-400"
        >
          {resolvedSub}
        </motion.div>
      </AnimatePresence>

      {/* ── Live trace timeline ── */}
      {turnTraces.length > 0 && (
        <div className="mt-2 space-y-1.5 pl-7">
          {turnTraces.slice(-5).map((trace) => {
            const isRunning = trace.status === "running";
            const isDone =
              trace.status === "completed" || trace.status === "failed";
            return (
              <motion.div
                key={trace.id}
                initial={{ opacity: 0, x: -6 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2 }}
                className="flex items-center gap-2"
              >
                <motion.div
                  className={`h-1.5 w-1.5 flex-shrink-0 rounded-full ${
                    isRunning
                      ? "bg-slate-700"
                      : isDone
                        ? "bg-slate-300"
                        : "bg-slate-200"
                  }`}
                  animate={
                    isRunning
                      ? { scale: [1, 1.6, 1], opacity: [0.7, 1, 0.7] }
                      : undefined
                  }
                  transition={
                    isRunning
                      ? { duration: 1, repeat: Infinity, ease: "easeInOut" }
                      : undefined
                  }
                />
                <span
                  className={`text-[12px] leading-4 ${
                    isRunning
                      ? "font-medium text-slate-700"
                      : isDone
                        ? "text-slate-400 line-through decoration-slate-300"
                        : "text-slate-500"
                  }`}
                >
                  {trace.label}
                </span>
              </motion.div>
            );
          })}
        </div>
      )}

      {/* ── Legacy steps (backwards compat) ── */}
      {activeSteps.length > 0 && turnTraces.length === 0 && (
        <div className="space-y-2 pl-7">
          {activeSteps.map((step, i) => {
            const isRunning = step.state === "running";
            const isDone = step.state === "done" || step.state === "completed";
            return (
              <div key={`${step.label}-${i}`} className="flex items-start gap-3">
                <motion.div
                  className={`mt-1.5 h-2 w-2 rounded-full ${
                    isRunning
                      ? "bg-slate-900"
                      : isDone
                        ? "bg-slate-300"
                        : "bg-slate-200"
                  }`}
                  animate={
                    isRunning
                      ? { scale: [1, 1.6, 1], opacity: [0.5, 1, 0.5] }
                      : undefined
                  }
                  transition={
                    isRunning
                      ? { duration: 1.2, repeat: Infinity, ease: "easeInOut" }
                      : undefined
                  }
                />
                <div>
                  <div
                    className={`text-[14px] leading-6 ${
                      isRunning
                        ? "font-medium text-slate-900"
                        : isDone
                          ? "text-slate-400"
                          : "text-slate-500"
                    }`}
                  >
                    {step.label}
                  </div>
                  {step.detail && (
                    <div className="text-[12px] leading-5 text-slate-400">
                      {step.detail}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
