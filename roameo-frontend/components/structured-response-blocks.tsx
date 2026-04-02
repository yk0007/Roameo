"use client";

import {
  CalendarDays,
  Clock3,
  LoaderCircle,
  MapPin,
  Sparkles
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { CompactPoiCard } from "./poi-card";
import type { ChatMessage, POI } from "@/lib/types";

interface StructuredResponseBlocksProps {
  message: ChatMessage;
  pois?: POI[];
  savedIds?: Set<string>;
  itineraryPoiIds?: Set<string>;
  onQuickAction?: (prompt: string) => void;
  onToggleSave?: (poi: POI, nextSaved: boolean) => void;
  onAddPoi?: (poi: POI) => void;
  onReplan?: (poi: POI) => void;
}

function findPoiMap(pois?: POI[]) {
  return new Map((pois || []).map((poi) => [poi.id, poi]));
}

function cardRow(
  poiIds: string[],
  poiMap: Map<string, POI>,
  savedIds?: Set<string>,
  itineraryPoiIds?: Set<string>,
  onToggleSave?: (poi: POI, nextSaved: boolean) => void,
  onAddPoi?: (poi: POI) => void,
  onReplan?: (poi: POI) => void
) {
  const items = poiIds
    .map((poiId) => poiMap.get(poiId))
    .filter((poi): poi is POI => Boolean(poi))
    .slice(0, 4);

  if (!items.length) {
    return null;
  }

  return (
    <div className="flex gap-3 overflow-x-auto pb-2">
      {items.map((poi) => (
        <CompactPoiCard
          key={poi.id}
          poi={poi}
          isSaved={savedIds?.has(poi.id) ?? false}
          isItineraryItem={itineraryPoiIds?.has(poi.id) ?? false}
          onToggleSave={(currentPoi, nextSaved) =>
            onToggleSave?.(currentPoi, nextSaved)
          }
          onAddPoi={(currentPoi) => onAddPoi?.(currentPoi)}
          onReplan={(currentPoi) => onReplan?.(currentPoi)}
        />
      ))}
    </div>
  );
}

export function StructuredResponseBlocks({
  message,
  pois,
  savedIds,
  itineraryPoiIds,
  onQuickAction,
  onToggleSave,
  onAddPoi,
  onReplan
}: StructuredResponseBlocksProps) {
  const blocks = message.meta?.responseBlocks;
  if (!blocks?.length) {
    return null;
  }

  const poiMap = findPoiMap(pois);

  return (
    <div className="space-y-4">
      {blocks.map((block, index) => {
        if (block.type === "trip_intro") {
          return (
            <div
              key={`${block.type}-${index}`}
              className="rounded-[24px] border border-white/55 bg-white/72 px-5 py-4 shadow-[0_12px_36px_rgba(15,23,42,0.08)]"
            >
              {block.eyebrow ? (
                <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.22em] text-orange-500">
                  {block.eyebrow}
                </div>
              ) : null}
              <div className="text-[20px] font-semibold italic leading-tight text-slate-900">
                {block.moodEmoji ? `${block.moodEmoji} ` : ""}
                {block.title}
              </div>
              <div className="mt-2 text-[15px] leading-7 text-slate-600">
                {block.body}
              </div>
            </div>
          );
        }

        if (block.type === "lead") {
          return (
            <div
              key={`${block.type}-${index}`}
              className="text-[15px] leading-7 text-slate-700"
            >
              {block.text}
            </div>
          );
        }

        if (block.type === "planning_status") {
          return (
            <div
              key={`${block.type}-${index}`}
              className="flex items-center gap-3 rounded-2xl border border-orange-100 bg-orange-50/80 px-4 py-3 text-sm text-slate-600"
            >
              <LoaderCircle className="h-4 w-4 animate-spin text-orange-500" />
              <div>
                <div className="font-medium text-slate-900">{block.label}</div>
                <div className="text-xs uppercase tracking-[0.18em] text-orange-500">
                  {block.stage.replace(/_/g, " ")}
                </div>
                {block.detail ? <div className="mt-1">{block.detail}</div> : null}
              </div>
            </div>
          );
        }

        if (block.type === "itinerary_template") {
          return (
            <div key={`${block.type}-${index}`} className="space-y-4">
              <div className="rounded-[26px] border border-orange-100 bg-gradient-to-br from-orange-50 via-white to-amber-50 px-5 py-5 shadow-[0_18px_44px_rgba(251,146,60,0.14)]">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className="text-[22px] font-semibold italic leading-tight text-slate-900">
                      {block.title}
                    </div>
                    {block.subtitle ? (
                      <div className="mt-2 text-sm leading-6 text-slate-600">
                        {block.subtitle}
                      </div>
                    ) : null}
                  </div>
                  {block.budgetLabel ? (
                    <div className="rounded-full border border-orange-200 bg-white/90 px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-orange-600">
                      {block.budgetLabel}
                    </div>
                  ) : null}
                </div>
              </div>

              <div className="space-y-4">
                {block.days.map((day) => (
                  <div
                    key={day.day}
                    className="rounded-[24px] border border-orange-100 bg-white/82 px-5 py-4 shadow-[0_12px_34px_rgba(251,146,60,0.10)]"
                  >
                    <div className="flex items-start gap-3 border-b border-orange-100 pb-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-orange-500 text-sm font-semibold text-white shadow-[0_10px_20px_rgba(249,115,22,0.35)]">
                        {day.day}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-orange-500">
                          <CalendarDays className="h-3.5 w-3.5" />
                          <span>{day.destination}</span>
                          {day.date ? <span>{day.date}</span> : null}
                        </div>
                        <div className="mt-1 text-lg font-semibold italic text-slate-900">
                          {day.title}
                        </div>
                        {day.summary ? (
                          <div className="mt-1 text-sm leading-6 text-slate-600">
                            {day.summary}
                          </div>
                        ) : null}
                      </div>
                    </div>

                    <div className="mt-4 space-y-4">
                      {day.periods.map((period) => (
                        <div key={`${day.day}-${period.key}`} className="space-y-2">
                          <div className="rounded-xl border-l-[6px] border-orange-400 bg-orange-50 px-3 py-2 text-sm font-semibold text-slate-900">
                            {period.emoji ? `${period.emoji} ` : ""}
                            {period.label}
                          </div>
                          <div className="space-y-2 pl-1">
                            {period.entries.map((entry, entryIndex) => {
                              const poi = entry.poiId ? poiMap.get(entry.poiId) : undefined;
                              return (
                                <div
                                  key={`${day.day}-${period.key}-${entryIndex}`}
                                  className="flex gap-3 rounded-2xl bg-white/80 px-3 py-2"
                                >
                                  <div className="mt-1 h-2 w-2 rounded-full bg-orange-300" />
                                  <div className="min-w-0 flex-1">
                                    <div className="flex flex-wrap items-center gap-2">
                                      {entry.timeLabel ? (
                                        <span className="inline-flex items-center gap-1 rounded-full bg-orange-50 px-2 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-orange-600">
                                          <Clock3 className="h-3 w-3" />
                                          {entry.timeLabel}
                                        </span>
                                      ) : null}
                                      <span className="font-semibold text-slate-900">
                                        {entry.title}
                                      </span>
                                    </div>
                                    {entry.description ? (
                                      <div className="mt-1 text-sm leading-6 text-slate-600">
                                        {entry.description}
                                      </div>
                                    ) : null}
                                    {poi?.address ? (
                                      <div className="mt-1 flex items-center gap-1 text-xs text-slate-500">
                                        <MapPin className="h-3.5 w-3.5" />
                                        <span className="line-clamp-1">{poi.address}</span>
                                      </div>
                                    ) : null}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      ))}
                    </div>

                    {day.footer ? (
                      <div className="mt-4 rounded-xl bg-orange-50 px-3 py-2 text-sm text-slate-600">
                        {day.footer}
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>
          );
        }

        if (block.type === "place_card_row") {
          const row = cardRow(
            block.poiIds,
            poiMap,
            savedIds,
            itineraryPoiIds,
            onToggleSave,
            onAddPoi,
            onReplan
          );
          if (!row) {
            return null;
          }

          return (
            <div key={`${block.type}-${index}`} className="space-y-3">
              {block.title ? (
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                  {block.title}
                </div>
              ) : null}
              {row}
            </div>
          );
        }

        if (block.type === "recommendation_cards") {
          const row = cardRow(
            block.poiIds,
            poiMap,
            savedIds,
            itineraryPoiIds,
            onToggleSave,
            onAddPoi,
            onReplan
          );
          if (!row) {
            return null;
          }

          return (
            <div key={`${block.type}-${index}`} className="space-y-3">
              {block.title ? (
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                  {block.title}
                </div>
              ) : null}
              {row}
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
                    className="rounded-full border border-orange-200 bg-orange-50 px-4 py-2 text-left text-sm text-slate-700 transition-colors hover:border-orange-300 hover:bg-orange-100"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          );
        }

        if (block.type === "assistant_prompt_chips") {
          return (
            <div key={`${block.type}-${index}`} className="space-y-3">
              {block.title ? (
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                  {block.title}
                </div>
              ) : null}
              <div className="flex flex-wrap gap-2">
                {block.prompts.map((action) => (
                  <Button
                    key={action.label}
                    type="button"
                    variant="outline"
                    size="sm"
                    className="rounded-full border-orange-200 bg-white/80 text-slate-700 hover:bg-orange-50"
                    onClick={() => onQuickAction?.(action.prompt)}
                  >
                    <Sparkles className="mr-1.5 h-3.5 w-3.5 text-orange-500" />
                    {action.label}
                  </Button>
                ))}
              </div>
            </div>
          );
        }

        if (block.type === "itinerary_summary") {
          return (
            <div key={`${block.type}-${index}`} className="space-y-2">
              {block.title ? (
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                  {block.title}
                </div>
              ) : null}
              {block.days.map((day) => (
                <div
                  key={day.day}
                  className="rounded-[18px] border border-slate-200/80 bg-white/80 px-4 py-3"
                >
                  <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
                    Day {day.day}
                  </div>
                  <div className="mt-1 text-base font-semibold text-slate-900">
                    {day.title}
                  </div>
                  {day.summary ? (
                    <div className="mt-1 text-sm leading-6 text-slate-600">
                      {day.summary}
                    </div>
                  ) : null}
                </div>
              ))}
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
