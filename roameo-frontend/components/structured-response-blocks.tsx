"use client";

import {
  LoaderCircle,
  MapPin,
  Sparkles,
  Star
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { resolvePoiImageUrl } from "@/lib/poi-image-url";
import { SearchCard } from "./search-card";
import { PoiTypeIcon } from "./poi-type-icon";
import { AgenticStatus } from "./agentic-status";
import type { ChatMessage, POI } from "@/lib/types";

interface StructuredResponseBlocksProps {
  message: ChatMessage;
  pois?: POI[];
  savedIds?: Set<string>;
  itineraryPoiIds?: Set<string>;
  onQuickAction?: (prompt: string) => void;
  onSlotAction?: (action: { field: string; value: string | number }, prompt: string) => void;
  onToggleSave?: (poi: POI, nextSaved: boolean) => void;
  onAddPoi?: (poi: POI) => void;
  onReplan?: (poi: POI) => void;
  hideProgressBlocks?: boolean;
}

function findPoiMap(pois?: POI[]) {
  return new Map((pois || []).map((poi) => [poi.id, poi]));
}

/** Strip markdown formatting characters so plain-text fields don't show raw symbols. */
function strip(text?: string | null): string {
  if (!text) return "";
  return text
    .replace(/\*\*(.*?)\*\*/g, "$1")   // **bold**
    .replace(/__(.*?)__/g, "$1")        // __bold__
    .replace(/\*(.*?)\*/g, "$1")        // *italic*
    .replace(/_(.*?)_/g, "$1")          // _italic_
    .replace(/`(.*?)`/g, "$1")          // `code`
    .replace(/~~(.*?)~~/g, "$1")        // ~~strike~~
    .replace(/#+\s*/g, "")              // ## headings
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1"); // [link](url)
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
  const items = Array.from(new Set(poiIds))
    .map((poiId) => poiMap.get(poiId))
    .filter((poi): poi is POI => Boolean(poi))
    .slice(0, 4);

  if (!items.length) {
    return null;
  }

  return (
    // Each card is fixed at 188px wide — same SearchCard design, scaled down for chat column
    <div className="flex gap-3 overflow-x-auto pb-2 [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
      {items.map((poi) => (
        <div key={poi.id} className="w-[188px] flex-shrink-0">
          <SearchCard
            poi={poi}
            isSaved={savedIds?.has(poi.id) ?? false}
            isItineraryItem={itineraryPoiIds?.has(poi.id) ?? false}
            compact
            onToggleSave={(currentPoi, nextSaved) =>
              onToggleSave?.(currentPoi, nextSaved)
            }
            onAddPoi={(currentPoi) => onAddPoi?.(currentPoi)}
            onReplan={(currentPoi) => onReplan?.(currentPoi)}
          />
        </div>
      ))}
    </div>
  );
}

function poiTypeLabel(poi: POI) {
  switch (poi.type) {
    case "restaurant":
      return "Restaurant";
    case "stay":
      return "Stay";
    case "destination":
      return "Destination";
    case "transit":
      return "Transit";
    default:
      return "Attraction";
  }
}

export function StructuredResponseBlocks({
  message,
  pois,
  savedIds,
  itineraryPoiIds,
  onQuickAction,
  onToggleSave,
  onAddPoi,
  onReplan,
  hideProgressBlocks,
  onSlotAction
}: StructuredResponseBlocksProps) {
  const blocks = message.meta?.responseBlocks;
  if (!blocks?.length) {
    return null;
  }

  const poiMap = findPoiMap(pois);
  const hasItinerary = hideProgressBlocks || blocks.some((b) => b.type === "itinerary_template");

  return (
    <div className="space-y-4">
      {blocks.map((block, index) => {
        // Hide progress/status blocks once the itinerary is ready
        if (hasItinerary && (block.type === "worker_progress" || block.type === "planning_status")) {
          return null;
        }

        if (block.type === "trip_intro") {
          return (
            <div
              key={`${block.type}-${index}`}
              className="py-1"
            >
              {block.eyebrow ? (
                <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">
                  {block.eyebrow}
                </div>
              ) : null}
              <div className="text-[20px] font-semibold italic leading-tight text-slate-900">
                {block.moodEmoji ? `${block.moodEmoji} ` : ""}
                {block.title}
              </div>
              <div className="mt-2 text-[15px] leading-7 text-slate-600">
                {strip(block.body)}
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
              {strip(block.text)}
            </div>
          );
        }

        if (block.type === "capabilities_overview") {
          return (
            <div
              key={`${block.type}-${index}`}
              className="py-1"
            >
              <div className="text-[20px] font-semibold leading-tight text-slate-900">
                {block.title}
              </div>
              {block.intro ? (
                <div className="mt-2 text-[15px] leading-7 text-slate-600">
                  {strip(block.intro)}
                </div>
              ) : null}
              <div className="mt-4 space-y-3">
                {block.sections.map((section) => (
                  <div key={section.title} className="rounded-2xl bg-slate-50/90 px-4 py-3">
                    <div className="font-semibold text-slate-900">{section.title}</div>
                    <div className="mt-1 text-sm leading-6 text-slate-600">
                      {strip(section.body)}
                    </div>
                  </div>
                ))}
              </div>
              {block.examples.length ? (
                <div className="mt-4">
                  <div className="text-sm font-semibold text-slate-900">
                    {block.examplesTitle || "Example help"}
                  </div>
                  <ul className="mt-2 space-y-2 pl-5 text-sm leading-6 text-slate-600">
                    {block.examples.map((example) => (
                      <li key={example}>{example}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          );
        }

        if (block.type === "planning_status") {
          return (
            <div
              key={`${block.type}-${index}`}
              className="flex items-center gap-3 py-2 text-sm text-slate-600"
            >
              <LoaderCircle className="h-4 w-4 animate-spin text-slate-400" />
              <div>
                <div className="font-medium text-slate-900">{block.label}</div>
                <div className="text-xs uppercase tracking-[0.18em] text-slate-400">
                  {block.stage.replace(/_/g, " ")}
                </div>
                {block.detail ? <div className="mt-1">{block.detail}</div> : null}
              </div>
            </div>
          );
        }

        if (block.type === "worker_progress") {
          return (
            <div
              key={`${block.type}-${index}`}
              className="px-1 py-1"
            >
              <AgenticStatus mode="worker" title={block.title} steps={block.steps} />
            </div>
          );
        }

        if (block.type === "date_advisory") {
          return (
            <div
              key={`${block.type}-${index}`}
              className="py-2"
            >
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                Timing guidance
              </div>
              <div className="mt-2 text-[18px] font-semibold text-slate-900">
                {block.title}
              </div>
              <div className="mt-2 text-sm leading-6 text-slate-600">{strip(block.summary)}</div>
              <div className="mt-4 space-y-3">
                {block.advisories.map((item) => (
                  <div key={`${item.kind}-${item.title}`} className="rounded-2xl bg-white/85 px-4 py-3">
                    <div className="text-sm font-semibold text-slate-900">{strip(item.title)}</div>
                    <div className="mt-1 text-sm leading-6 text-slate-600">{strip(item.detail)}</div>
                  </div>
                ))}
              </div>
            </div>
          );
        }

        if (block.type === "event_window_summary") {
          return (
            <div
              key={`${block.type}-${index}`}
              className="py-2"
            >
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                Date context
              </div>
              <div className="mt-2 text-[18px] font-semibold text-slate-900">{block.title}</div>
              {block.summary ? (
                <div className="mt-2 text-sm leading-6 text-slate-600">{strip(block.summary)}</div>
              ) : null}
              <div className="mt-4 space-y-3">
                {block.items.map((item) => (
                  <div key={`${item.title}-${item.sourceLabel || ""}`} className="rounded-2xl bg-slate-50/90 px-4 py-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <div className="font-semibold text-slate-900">{strip(item.title)}</div>
                      {item.sourceLabel ? (
                        <div className="rounded-full bg-white px-2 py-0.5 text-[11px] uppercase tracking-[0.14em] text-slate-400">
                          {item.sourceLabel}
                        </div>
                      ) : null}
                    </div>
                    <div className="mt-1 text-sm leading-6 text-slate-600">{strip(item.detail)}</div>
                  </div>
                ))}
              </div>
            </div>
          );
        }

        if (block.type === "itinerary_template") {
          return (
            <div key={`${block.type}-${index}`} className="space-y-6">
              {/* Title block */}
              <div>
                <div className="text-[22px] font-bold leading-tight tracking-[-0.02em] text-slate-900">
                  {block.title}
                </div>
                {block.subtitle ? (
                  <div className="mt-2 text-[15px] italic leading-7 text-slate-500">
                    {block.subtitle}
                  </div>
                ) : null}
                {block.budgetLabel ? (
                  <div className="mt-2 text-[13px] font-medium text-slate-400">
                    {block.budgetLabel}
                  </div>
                ) : null}
              </div>

              <hr className="border-slate-200" />

              {/* Days */}
              {block.days.map((day, dayIndex) => (
                <div key={day.day} className="space-y-4">
                  {/* Day header */}
                  <div>
                    <div className="text-[20px] font-bold leading-tight text-slate-900">
                      Day {day.day} – {day.title}
                    </div>
                    {day.summary ? (
                      <div className="mt-1.5 text-[15px] italic leading-7 text-slate-500">
                        {strip(day.summary)}
                      </div>
                    ) : null}
                  </div>

                  {/* Periods */}
                  {day.periods.map((period) => (
                    <div key={`${day.day}-${period.key}`} className="space-y-3">
                      <div className="text-[16px] font-semibold text-slate-900">
                        {period.emoji ? `${period.emoji} ` : ""}{period.label}:
                      </div>
                      <div className="space-y-3 text-[15px] leading-7 text-slate-700">
                        {period.entries.map((entry, entryIndex) => {
                          const poi = entry.poiId ? poiMap.get(entry.poiId) : undefined;
                          return (
                            <div key={`${day.day}-${period.key}-${entryIndex}`}>
                              {entry.description ? (
                                <p>
                                  {entry.title ? (
                                    <>→ <span className="font-bold text-slate-900">{strip(entry.title)}</span>{` — ${strip(entry.description)}`}</>
                                  ) : (
                                    strip(entry.description)
                                  )}
                                </p>
                              ) : (
                                <p>→ <span className="font-bold text-slate-900">{strip(entry.title)}</span></p>
                              )}
                              {poi?.address ? (
                                <div className="mt-1 flex items-center gap-1.5 text-[13px] text-slate-400">
                                  <MapPin className="h-3 w-3" />
                                  <span className="line-clamp-1">{poi.address}</span>
                                </div>
                              ) : null}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ))}

                  {day.footer ? (
                    <p className="text-[15px] leading-7 text-slate-600">
                      {strip(day.footer)}
                    </p>
                  ) : null}

                  {/* Divider between days, not after the last */}
                  {dayIndex < block.days.length - 1 ? (
                    <hr className="border-slate-200" />
                  ) : null}
                </div>
              ))}
            </div>
          );
        }

        if (block.type === "stay_recommendation_list") {
          const bestPoi = poiMap.get(block.bestOption.poiId);
          const bestPoiImage = resolvePoiImageUrl(bestPoi?.photoUrl);
          const alternativeRow = cardRow(
            block.alternatives.map((item) => item.poiId),
            poiMap,
            savedIds,
            itineraryPoiIds,
            onToggleSave,
            onAddPoi,
            onReplan
          );

          return (
            <div key={`${block.type}-${index}`} className="space-y-4">
              <div className="py-2">
                <div className="text-[20px] font-semibold text-slate-900">{block.title}</div>
                {block.intro ? (
                  <div className="mt-2 text-sm leading-6 text-slate-600">{strip(block.intro)}</div>
                ) : null}
                <div className="mt-5 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                  Best option
                </div>
                <div className="mt-3 flex gap-4 rounded-[24px] border border-slate-200/80 bg-slate-50/70 p-4">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <div className="text-xl font-semibold text-slate-900">
                        {bestPoi?.name || block.bestOption.title}
                      </div>
                      {typeof bestPoi?.rating === "number" ? (
                        <div className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2 py-1 text-xs font-semibold text-amber-700">
                          <Star className="h-3.5 w-3.5 fill-current" />
                          {bestPoi.rating.toFixed(1)}
                        </div>
                      ) : null}
                    </div>
                    {block.bestOption.rateLabel ? (
                      <div className="mt-2 text-sm font-medium text-slate-700">
                        {block.bestOption.rateLabel}
                      </div>
                    ) : null}
                    <div className="mt-2 text-[15px] leading-7 text-slate-600">
                      {strip(block.bestOption.body)}
                    </div>
                    {block.bestOption.caveat ? (
                      <div className="mt-3 text-sm leading-6 text-slate-500">
                        {strip(block.bestOption.caveat)}
                      </div>
                    ) : null}
                  </div>
                  {bestPoiImage ? (
                    <div className="h-40 w-40 shrink-0 overflow-hidden rounded-[20px] bg-slate-100">
                      <img
                        src={bestPoiImage}
                        alt={bestPoi?.name || block.bestOption.title}
                        className="h-full w-full object-cover"
                      />
                    </div>
                  ) : null}
                </div>
                {block.bookingDisclaimer ? (
                  <div className="mt-4 py-2 text-sm leading-6 text-slate-500 italic">
                    {block.bookingDisclaimer}
                  </div>
                ) : null}
              </div>
              {alternativeRow ? (
                <div className="space-y-3">
                  <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                    {block.alternativesTitle || "Other options that could work"}
                  </div>
                  {alternativeRow}
                </div>
              ) : null}
              {block.notFit.length ? (
                <div className="py-2">
                  <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                    {block.notFitTitle || "Not a good fit"}
                  </div>
                  <div className="mt-3 space-y-2">
                    {block.notFit.map((item) => (
                      <div key={item.label} className="text-sm leading-6 text-slate-600">
                        <span className="font-semibold text-slate-900">{item.label}:</span>{" "}
                        {item.reason}
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
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
              <div className="-mx-1">{row}</div>
            </div>
          );
        }

        if (block.type === "featured_poi") {
          const poi = poiMap.get(block.poiId);
          if (!poi) {
            return null;
          }

          return (
            <div key={`${block.type}-${index}`} className="space-y-3">
              {block.title ? (
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                  {block.title}
                </div>
              ) : null}
              {block.body ? (
                <div className="text-sm leading-6 text-slate-600">{strip(block.body)}</div>
              ) : null}
              <div className="-mx-1 overflow-x-auto pb-2 [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
                <div className="w-[188px]">
                  <SearchCard
                    poi={poi}
                    isSaved={savedIds?.has(poi.id) ?? false}
                    isItineraryItem={itineraryPoiIds?.has(poi.id) ?? false}
                    compact
                    onToggleSave={(currentPoi: POI, nextSaved: boolean) =>
                      onToggleSave?.(currentPoi, nextSaved)
                    }
                    onAddPoi={(currentPoi: POI) => onAddPoi?.(currentPoi)}
                    onReplan={(currentPoi: POI) => onReplan?.(currentPoi)}
                  />
                </div>
              </div>
            </div>
          );
        }

        if (block.type === "poi_story_list") {
          return (
            <div key={`${block.type}-${index}`} className="space-y-4">
              {block.title ? (
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                  {block.title}
                </div>
              ) : null}
              {block.intro ? (
                <div className="text-sm leading-6 text-slate-600">{strip(block.intro)}</div>
              ) : null}
              <div className="space-y-4">
                {block.items.map((item) => {
                  const poi = poiMap.get(item.poiId);
                  if (!poi) {
                    return null;
                  }

                  return (
                    <div
                      key={item.poiId}
                      className="flex gap-4 py-3"
                    >
                      {resolvePoiImageUrl(poi.photoUrl) ? (
                        <div className="h-32 w-32 shrink-0 overflow-hidden rounded-[20px] bg-slate-100">
                          <img
                            src={resolvePoiImageUrl(poi.photoUrl)}
                            alt={poi.name}
                            className="h-full w-full object-cover"
                          />
                        </div>
                      ) : null}
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <div className="text-lg font-semibold text-slate-900">
                            {item.title || poi.name}
                          </div>
                          {typeof poi.rating === "number" ? (
                            <div className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2 py-1 text-xs font-semibold text-amber-700">
                              <Star className="h-3.5 w-3.5 fill-current" />
                              {poi.rating.toFixed(1)}
                            </div>
                          ) : null}
                        </div>
                        <div className="mt-1 flex flex-wrap items-center gap-2 text-xs uppercase tracking-[0.16em] text-slate-400">
                          <span>{item.badge || poiTypeLabel(poi)}</span>
                        </div>
                        <div className="mt-3 text-[15px] leading-7 text-slate-600">
                          {strip(item.body)}
                        </div>
                        {poi.address ? (
                          <div className="mt-2 flex items-center gap-1 text-xs text-slate-500">
                            <PoiTypeIcon poi={poi} className="h-3.5 w-3.5" />
                            <span className="line-clamp-1">{poi.address}</span>
                          </div>
                        ) : null}
                      </div>
                    </div>
                  );
                })}
              </div>
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
              <div className="-mx-1">{row}</div>
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
                    className="rounded-full border border-slate-200 bg-white px-4 py-2 text-left text-sm text-slate-700 transition-colors hover:border-slate-300 hover:bg-slate-50"
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
                    className="rounded-full border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                    onClick={() => action.slotAction ? onSlotAction?.(action.slotAction, action.prompt) : onQuickAction?.(action.prompt)}
                  >
                    <Sparkles className="mr-1.5 h-3.5 w-3.5 text-slate-400" />
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
                  className="py-2"
                >
                  <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
                    Day {day.day}
                  </div>
                  <div className="mt-1 text-base font-semibold text-slate-900">
                    {day.title}
                  </div>
                  {day.summary ? (
                    <div className="mt-1 text-sm leading-6 text-slate-600">
                      {strip(day.summary)}
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
