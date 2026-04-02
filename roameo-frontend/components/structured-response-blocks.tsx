"use client";

import { CalendarDays, LoaderCircle, MapPin, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { ChatMessage, POI } from "@/lib/types";

interface StructuredResponseBlocksProps {
  message: ChatMessage;
  pois?: POI[];
  onQuickAction?: (prompt: string) => void;
}

function findPoiMap(pois?: POI[]) {
  return new Map((pois || []).map((poi) => [poi.id, poi]));
}

export function StructuredResponseBlocks({
  message,
  pois,
  onQuickAction
}: StructuredResponseBlocksProps) {
  const blocks = message.meta?.responseBlocks;
  if (!blocks?.length) {
    return null;
  }

  const poiMap = findPoiMap(pois);

  return (
    <div className="space-y-4">
      {blocks.map((block, index) => {
        if (block.type === "lead") {
          return (
            <div key={`${block.type}-${index}`} className="text-[15px] leading-7 text-slate-700">
              {block.text}
            </div>
          );
        }

        if (block.type === "planning_status") {
          return (
            <div
              key={`${block.type}-${index}`}
              className="flex items-center gap-3 rounded-2xl border border-white/40 bg-white/65 px-4 py-3 text-sm text-slate-600 shadow-[0_12px_32px_rgba(15,23,42,0.08)] backdrop-blur-xl"
            >
              <LoaderCircle className="h-4 w-4 animate-spin text-slate-500" />
              <div>
                <div className="font-medium text-slate-900">{block.label}</div>
                {block.detail ? <div>{block.detail}</div> : null}
              </div>
            </div>
          );
        }

        if (block.type === "clarifying_questions") {
          return (
            <div key={`${block.type}-${index}`} className="space-y-3">
              {block.title ? (
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                  {block.title}
                </div>
              ) : null}
              <div className="flex flex-wrap gap-2">
                {block.questions.map((question) => (
                  <button
                    key={question}
                    type="button"
                    onClick={() => onQuickAction?.(question)}
                    className="rounded-full border border-slate-200 bg-white/80 px-4 py-2 text-left text-sm text-slate-700 transition-colors hover:border-slate-300 hover:bg-white"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          );
        }

        if (block.type === "recommendation_cards") {
          const recommendations = block.poiIds
            .map((poiId) => poiMap.get(poiId))
            .filter((poi): poi is POI => Boolean(poi))
            .slice(0, 4);
          if (!recommendations.length) {
            return null;
          }

          return (
            <div key={`${block.type}-${index}`} className="space-y-3">
              {block.title ? (
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                  {block.title}
                </div>
              ) : null}
              <div className="grid gap-3 sm:grid-cols-2">
                {recommendations.map((poi) => (
                  <div
                    key={poi.id}
                    className="overflow-hidden rounded-[22px] border border-white/35 bg-white/75 shadow-[0_18px_40px_rgba(15,23,42,0.08)] backdrop-blur-xl"
                  >
                    <div
                      className="h-28 bg-cover bg-center"
                      style={{
                        backgroundImage: poi.photoUrl
                          ? `linear-gradient(180deg, rgba(15,23,42,0.04), rgba(15,23,42,0.38)), url("${poi.photoUrl}")`
                          : "linear-gradient(135deg, #dbeafe, #eff6ff)"
                      }}
                    />
                    <div className="space-y-1 px-4 py-3">
                      <div className="text-sm font-semibold text-slate-900">{poi.name}</div>
                      <div className="flex items-center gap-2 text-xs text-slate-500">
                        <MapPin className="h-3.5 w-3.5" />
                        <span className="line-clamp-1">{poi.address || "Location available on the map"}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        }

        if (block.type === "itinerary_summary") {
          return (
            <div key={`${block.type}-${index}`} className="space-y-3">
              {block.title ? (
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                  {block.title}
                </div>
              ) : null}
              <div className="space-y-2">
                {block.days.map((day) => (
                  <div
                    key={day.day}
                    className="rounded-[20px] border border-slate-200/80 bg-white/80 px-4 py-3"
                  >
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
                      <CalendarDays className="h-3.5 w-3.5" />
                      Day {day.day}
                    </div>
                    <div className="mt-1 text-base font-semibold text-slate-900">{day.title}</div>
                    {day.summary ? (
                      <div className="mt-1 text-sm leading-6 text-slate-600">{day.summary}</div>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>
          );
        }

        if (block.type === "quick_actions") {
          return (
            <div key={`${block.type}-${index}`} className="space-y-3">
              {block.title ? (
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                  {block.title}
                </div>
              ) : null}
              <div className="flex flex-wrap gap-2">
                {block.actions.map((action) => (
                  <Button
                    key={action.label}
                    type="button"
                    variant="outline"
                    size="sm"
                    className="rounded-full border-slate-200 bg-white/75 text-slate-700 hover:bg-white"
                    onClick={() => onQuickAction?.(action.prompt)}
                  >
                    <Sparkles className="mr-1.5 h-3.5 w-3.5" />
                    {action.label}
                  </Button>
                ))}
              </div>
            </div>
          );
        }

        return null;
      })}
    </div>
  );
}
