"use client";

import React, { type ReactNode, isValidElement, cloneElement } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Send, Plus, ArrowDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { sendChat } from "@/lib/api";
import { InlinePlanningStatus } from "./inline-planning-status";
import { StructuredResponseBlocks } from "./structured-response-blocks";
import { TypingIndicator } from "./typing-indicator";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { CachedImage } from "./cached-image";
import { CompactPoiCard } from "./poi-card";
import type { ChatMessage, POI, SessionPlanningState } from "@/lib/types";

function normalizeRenderableMessageContent(content: string) {
  return content
    .replace(/,?\[object Object\],?/g, "")
    .replace(/\[object Object\]/g, "")
    .replace(/object Object/g, "")
    .trim();
}

function shouldRenderMessage(message: ChatMessage) {
  if (message.role !== "user" && message.role !== "assistant") {
    return false;
  }

  if (message.role === "assistant" && message.meta?.responseBlocks?.length) {
    return true;
  }

  return normalizeRenderableMessageContent(String(message.content || "")).length > 0;
}

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onSendMessage: (content: string) => void;
  activeView: "chat" | "search" | "saved";
  onViewChange: (view: "chat" | "search" | "saved") => void;
  isRightPanelVisible?: boolean;
  activeRightView?: "map" | "itinerary";
  setActiveRightView?: (view: "map" | "itinerary") => void;
  setIsRightPanelVisible?: (visible: boolean) => void;
  pois?: POI[];
  isTyping?: boolean;
  detectedIntent?: "PLAN_TRIP" | "DESTINATION_SEARCH" | "CHAT" | null;
  planningActive?: boolean;
  planningState?: SessionPlanningState;
  savedIds?: Set<string>;
  itineraryPoiIds?: Set<string>;
  onToggleSave?: (poi: POI, nextSaved: boolean) => void;
  onAddPoi?: (poi: POI) => void;
  onReplan?: (poi: POI) => void;
  onPopulateInput?: (text: string) => void;
  inputValue?: string;
  onInputChange?: (value: string) => void;
}

export function ChatInterface({
  messages,
  onSendMessage,
  activeView,
  onViewChange,
  isRightPanelVisible,
  activeRightView,
  setActiveRightView,
  setIsRightPanelVisible,
  pois,
  isTyping,
  detectedIntent,
  planningActive,
  planningState,
  savedIds,
  itineraryPoiIds,
  onToggleSave,
  onAddPoi,
  onReplan,
  onPopulateInput,
  inputValue: externalInputValue,
  onInputChange,
}: ChatInterfaceProps) {
  const [inputValue, setInputValue] = useState(externalInputValue || "");
  const [showSuggestion, setShowSuggestion] = useState(false);
  const [lastSuggestionAt, setLastSuggestionAt] = useState<number>(0);
  const SUGGESTION_COOLDOWN_MS = 60_000; // do not show more than once per minute
  const SUGGESTION_PROBABILITY = 0.65; // ~65% of eligible assistant turns
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  const [userJustSent, setUserJustSent] = useState(false);
  const [responseType, setResponseType] = useState<
    "general" | "planning" | null
  >(null);
  const lastAssistantIdRef = useRef<string | null>(null);
  const typedMessageIdsRef = useRef<Set<string>>(new Set());
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Sync external input value with internal state
  useEffect(() => {
    if (externalInputValue !== undefined && externalInputValue !== inputValue) {
      setInputValue(externalInputValue);
    }
  }, [externalInputValue]);

  // Handle input changes
  const handleInputChange = (value: string) => {
    setInputValue(value);
    onInputChange?.(value);
  };

  const safeMessages = messages || [];
  const visibleMessages = useMemo(
    () => safeMessages.filter(shouldRenderMessage),
    [safeMessages]
  );

  // Debug logging for ChatInterface messages
  useEffect(() => {
    console.log(
      "[chat-interface] Messages prop updated, count:",
      safeMessages.length,
    );
    if (visibleMessages.length > 0) {
      const lastMessage = visibleMessages[visibleMessages.length - 1];
      console.log("[chat-interface] Last message:", {
        id: lastMessage.id,
        role: lastMessage.role,
        contentLength: lastMessage.content?.length,
      });
    }
  }, [safeMessages.length, visibleMessages]);

  // Force re-render when messages change to ensure assistant messages are visible
  useEffect(() => {
    if (visibleMessages.length > 0) {
      const lastMessage = visibleMessages[visibleMessages.length - 1];
      if (lastMessage.role === "assistant") {
        console.log(
          "[chat-interface] Assistant message detected - ensuring visibility",
        );
        // Scroll to show the new message
        setTimeout(() => {
          scrollToBottom("auto");
        }, 50);
      }
    }
  }, [visibleMessages]);
  const lastMessage = useMemo(() => {
    return visibleMessages.length > 0
      ? visibleMessages[visibleMessages.length - 1]
      : undefined;
  }, [visibleMessages]);

  const lastAssistant = useMemo(() => {
    for (let i = visibleMessages.length - 1; i >= 0; i--) {
      if (visibleMessages[i].role === "assistant") return visibleMessages[i];
    }
    return undefined;
  }, [visibleMessages]);
  const lastAssistantHasStructuredBlocks = Boolean(lastAssistant?.meta?.responseBlocks?.length);

  // Decide when to show the suggestion chip: only after a substantive assistant reply,
  // only once per assistant turn, and only on the Chat view, and not when user just sent a message
  useEffect(() => {
    if (!lastAssistant) return;
    if (activeView !== "chat") return;
    if (userJustSent) return; // Hide suggestions when user just sent a message
    const isNewAssistantTurn = lastAssistantIdRef.current !== lastAssistant.id;
    if (isNewAssistantTurn) {
      lastAssistantIdRef.current = lastAssistant.id;
      const longEnough =
        (lastAssistant.content?.length || 0) > 80 && visibleMessages.length >= 3;
      const cooldownOk = Date.now() - lastSuggestionAt > SUGGESTION_COOLDOWN_MS;
      const chance = Math.random() < SUGGESTION_PROBABILITY;
      const shouldShow =
        !lastAssistantHasStructuredBlocks && longEnough && cooldownOk && chance;
      setShowSuggestion(shouldShow);
      if (shouldShow) setLastSuggestionAt(Date.now());
      setUserJustSent(false); // Reset after assistant responds
    }
  }, [
    lastAssistant,
    visibleMessages.length,
    activeView,
    userJustSent,
    lastSuggestionAt,
    lastAssistantHasStructuredBlocks,
  ]);

  const scrollToBottom = (behavior: "smooth" | "auto" = "smooth") => {
    messagesEndRef.current?.scrollIntoView({ behavior });
  };

  useEffect(() => {
    // Auto-scroll on new messages
    scrollToBottom("auto");
  }, [visibleMessages, isTyping]);

  // Reset response type when typing stops
  useEffect(() => {
    if (!isTyping) {
      setResponseType(null);
    }
  }, [isTyping]);

  // Update response type based on server-detected intent
  // Handle typing indicator visibility based on detected intent
  useEffect(() => {
    if (isTyping || planningActive) {
      if (
        detectedIntent === "PLAN_TRIP" ||
        planningActive ||
        detectedIntent === "DESTINATION_SEARCH"
      ) {
        console.log(
          "[chat-interface] Setting response type to planning for intent:",
          detectedIntent,
          "planningActive:",
          planningActive,
        );
        setResponseType("planning");
      } else if (detectedIntent) {
        console.log(
          "[chat-interface] Setting response type to general for intent:",
          detectedIntent,
        );
        setResponseType("general");
      } else {
        // If typing but no intent detected yet, default to general typing indicator
        console.log(
          "[chat-interface] No intent detected yet, defaulting to general typing",
        );
        setResponseType("general");
      }
    } else {
      setResponseType(null);
    }
  }, [detectedIntent, isTyping, planningActive]);

  // Ensure typing indicator is hidden when messages arrive
  useEffect(() => {
    if (visibleMessages.length > 0) {
      const lastMessage = visibleMessages[visibleMessages.length - 1];
      if (lastMessage.role === "assistant") {
        console.log(
          "[chat-interface] Assistant message detected - forcing typing state clear",
        );
        // This is a safety net to ensure messages are always visible
        // The parent component should handle this, but this provides redundancy

        // If we're passed an onTypingChange callback, use it to notify parent
        if (typeof onSendMessage === "function") {
          // This is a hack to force parent state update
          setTimeout(() => {
            console.log("[chat-interface] Forcing parent state notification");
          }, 0);
        }
      }
    }
  }, [visibleMessages, isTyping, onSendMessage]);

  const handleScroll = () => {
    const container = scrollContainerRef.current;
    if (container) {
      const isScrolledUp =
        container.scrollHeight - container.scrollTop >
        container.clientHeight + 150;
      setShowScrollToBottom(isScrolledUp);
    }
  };

  // Shared strong renderer that turns bold names into POI hover cards when matched
  const StrongWithPoiHover: React.FC<any> = ({ children, ..._props }) => {
    const flat = React.Children.toArray(children)
      .map((c: any) => (typeof c === "string" ? c : ""))
      .join("")
      .trim();
    const matchingPoi = flat
      ? pois?.find(
          (p: any) =>
            p.name.toLowerCase().includes(flat.toLowerCase()) ||
            flat.toLowerCase().includes(p.name.toLowerCase()),
        )
      : undefined;
    if (matchingPoi && onPopulateInput) {
      const isSaved = savedIds?.has?.(matchingPoi.id || "") ?? false;
      return (
        <HoverCard>
          <HoverCardTrigger asChild>
            <strong
              className="font-semibold text-gray-900 cursor-pointer hover:underline"
              onClick={() => {
                const loc = matchingPoi.address
                  ? ` at ${matchingPoi.address}`
                  : "";
                const coord =
                  matchingPoi.lat && matchingPoi.lng
                    ? ` (coords: ${matchingPoi.lat}, ${matchingPoi.lng})`
                    : "";
                const msg = `Please add ${matchingPoi.name}${loc}${coord} to my itinerary in an appropriate slot. If needed, adjust nearby activities accordingly.`;
                onPopulateInput(msg);
              }}
            >
              {children}
            </strong>
          </HoverCardTrigger>
          <HoverCardContent className="w-80 p-0" side="top" align="start">
            <CompactPoiCard
              poi={matchingPoi}
              isSaved={isSaved}
              isItineraryItem={false}
              onToggleSave={(poi, next) => onToggleSave?.(poi, next)}
              onAddPoi={(poi) => {
                const loc = poi.address ? ` at ${poi.address}` : "";
                const coord =
                  poi.lat && poi.lng ? ` (coords: ${poi.lat}, ${poi.lng})` : "";
                const msg = `Please add ${poi.name}${loc}${coord} to my itinerary in an appropriate slot. If needed, adjust nearby activities accordingly.`;
                onPopulateInput?.(msg);
              }}
              onReplan={(poi) => onReplan?.(poi)}
            />
          </HoverCardContent>
        </HoverCard>
      );
    }
    return (
      <strong className="font-semibold text-gray-900">
        {renderWithPoiHover(children, true)}
      </strong>
    );
  };

  const renderWithPoiHover = (
    children: React.ReactNode,
    isAssistant: boolean = false,
  ): React.ReactNode => {
    const processNode = (node: React.ReactNode): React.ReactNode => {
      if (typeof node === "string") {
        // Split by POI markers (📍, 📌, pushpin U+1F4CD) with optional VS16 and process each part
        const parts = node.split(
          /((?:📍|📌|\u{1F4CD})(?:\uFE0F)?\s*[\w\s.'-]+\b)/gu,
        );

        return parts.map((part: string, index: number) => {
          if (/^(?:📍|📌|\u{1F4CD})(?:\uFE0F)?/u.test(part)) {
            const pinMatch = part.match(/^(?:📍|📌|\u{1F4CD})(?:\uFE0F)?/u);
            const pin = pinMatch ? pinMatch[0] : "📍";
            const poiName = part
              .replace(/^(?:📍|📌|\u{1F4CD})(?:\uFE0F)?/u, "")
              .trim();
            const matchingPoi = pois?.find(
              (poi: any) =>
                poi.name.toLowerCase().includes(poiName.toLowerCase()) ||
                poiName.toLowerCase().includes(poi.name.toLowerCase()),
            );

            if (matchingPoi && isAssistant && onPopulateInput) {
              const isSaved = savedIds?.has?.(matchingPoi.id || "") ?? false;
              return (
                <HoverCard key={`${poiName}-${index}`}>
                  <HoverCardTrigger asChild>
                    <span
                      className="cursor-pointer text-blue-600 hover:text-blue-800 hover:underline font-semibold"
                      onClick={() => {
                        const loc = matchingPoi.address
                          ? ` at ${matchingPoi.address}`
                          : "";
                        const coord =
                          matchingPoi.lat && matchingPoi.lng
                            ? ` (coords: ${matchingPoi.lat}, ${matchingPoi.lng})`
                            : "";
                        const msg = `Please add ${matchingPoi.name}${loc}${coord} to my itinerary in an appropriate slot. If needed, adjust nearby activities accordingly.`;
                        onPopulateInput(msg);
                      }}
                      title={`Click to add ${matchingPoi.name} to trip`}
                    >
                      {pin} {poiName}
                    </span>
                  </HoverCardTrigger>
                  <HoverCardContent
                    className="w-80 p-0"
                    side="top"
                    align="start"
                  >
                    <CompactPoiCard
                      poi={matchingPoi}
                      isSaved={isSaved}
                      isItineraryItem={false}
                      onToggleSave={(poi, next) => onToggleSave?.(poi, next)}
                      onAddPoi={(poi) => {
                        const loc = poi.address ? ` at ${poi.address}` : "";
                        const coord =
                          poi.lat && poi.lng
                            ? ` (coords: ${poi.lat}, ${poi.lng})`
                            : "";
                        const msg = `Please add ${poi.name}${loc}${coord} to my itinerary in an appropriate slot. If needed, adjust nearby activities accordingly.`;
                        onPopulateInput(msg);
                      }}
                      onReplan={(poi) => onReplan?.(poi)}
                    />
                  </HoverCardContent>
                </HoverCard>
              );
            }
            // Fallback: bold the pin text even if we don't have a matching POI
            return (
              <strong
                key={`${poiName}-${index}`}
                className="font-semibold text-gray-900"
              >
                {pin} {poiName}
              </strong>
            );
          }
          return part;
        });
      }

      if (React.isValidElement(node)) {
        return React.cloneElement(node as React.ReactElement<any>, {
          children: React.Children.map((node as any).props.children, (child) =>
            processNode(child),
          ),
        });
      }

      return node;
    };

    return processNode(children);
  };

  const renderFormattedContent = (content: string) => {
    // Clean the content thoroughly
    let cleanedContent = content
      .replace(/,?\[object Object\],?/g, "")
      .replace(/\[object Object\]/g, "")
      .replace(/object Object/g, "")
      .replace(/,\s*,/g, ",")
      .replace(/^\s*,\s*/, "")
      .replace(/\s*,\s*$/, "")
      .trim();

    // Helper: remove bold around POI names so asterisks don't show and POIs aren't bolded
    const stripPoiBold = (text: string) => text; // deprecated, keep for backward compat but no-op

    // Helper: remove bold around currency ranges so **₹500 - ₹1000** -> ₹500 - ₹1000
    const stripPriceBold = (text: string) =>
      text
        // Specifically handle lines that include Price range: **₹...**
        .replace(/(Price range:\s*)\*\*([^*]+)\*\*/gi, "$1$2")
        // Fallback: any bold that starts with a currency or digit, unbold it
        .replace(/\*\*([₹$€£]\s?[^*]+)\*\*/g, "$1");

    // Debug: print the raw and cleaned markdown so we can compare with the UI rendering
    try {
      // Only log non-trivial assistant payloads
      if (cleanedContent && cleanedContent.length > 0) {
      }
    } catch {}

    if (!cleanedContent) {
      return <div className="text-gray-500 italic">Empty content</div>;
    }

    // Process questions, but do not return early; we'll append them after itinerary rendering
    const questionHeader = "🤔 Got More Questions?";
    let extractedQuestions: string[] = [];
    if (cleanedContent.includes(questionHeader)) {
      const parts = cleanedContent.split(questionHeader);
      cleanedContent = parts[0];
      const questionsText = parts[1] || "";
      extractedQuestions =
        questionsText.match(/"(.*?)"/g)?.map((q) => q.slice(1, -1)) || [];
    }

    // Process content line by line for better control (preserve blank lines for spacing)
    const lines = cleanedContent.split("\n");

    return (
      <div className="space-y-3.5">
        {lines.map((line, index) => {
          const trimmedLine = line.trim();
          // Per-line classification logs
          if (trimmedLine.length) {
          }

          // Main title (# *Title*) — italic, slightly lighter weight
          if (trimmedLine.match(/^#\s*\*.*\*$/)) {
            const title = trimmedLine.replace(/^#\s*\*(.+)\*$/, "$1");
            return (
              <h1
                key={index}
                className="text-[22px] md:text-2xl font-semibold italic mb-2.5 text-gray-900"
              >
                {renderWithPoiHover(title, true)}
              </h1>
            );
          }

          // Day headers (## *Day 1:*) — italic, lighter weight with subtle divider
          if (trimmedLine.match(/^##\s*\*.*\*$/)) {
            const dayTitle = trimmedLine.replace(/^##\s*\*(.+)\*$/, "$1");
            return (
              <h2
                key={index}
                className="text-xl font-medium italic mb-2.5 mt-5 text-gray-900 border-b border-gray-200 pb-2"
              >
                {renderWithPoiHover(dayTitle, true)}
              </h2>
            );
          }

          // Time section headers (### ☀️ Morning) — orange banner with strong left bar
          if (
            trimmedLine.match(
              /^###\s*(☀️|⛵|🌅|🚗|🛍️|🍽️|🏨|✈️)\s*(Morning|Afternoon|Evening|Shopping|Heritage|Departure)/,
            )
          ) {
            const sectionTitle = trimmedLine.replace(/^###\s*/, "");
            return (
              <div key={index} className="mt-5 mb-2">
                <div className="bg-orange-50 border-l-8 border-orange-400 rounded-md px-3 py-1.5">
                  <h3 className="font-semibold text-gray-900 text-base">
                    {renderWithPoiHover(sectionTitle, true)}
                  </h3>
                </div>
              </div>
            );
          }

          // Skip stray '###' lines (sometimes produced by model)
          if (/^###\s*$/.test(trimmedLine)) {
            return <div key={index} className="h-1.5" />;
          }

          // Generic ### headings (e.g., Accommodation & Meals, Estimated Budget)
          if (trimmedLine.startsWith("### ")) {
            const sectionTitle = trimmedLine.replace(/^###\s*/, "");
            return (
              <div key={index} className="mt-5 mb-2">
                <div className="bg-orange-50 border-l-8 border-orange-400 rounded-md px-3 py-2">
                  <h3 className="font-semibold text-gray-900 text-base">
                    {renderWithPoiHover(sectionTitle, true)}
                  </h3>
                </div>
              </div>
            );
          }

          // Meals subheading emitted as a bullet (e.g., "*  Meals:" or "**Meals:**") — render as banner
          {
            const normalized = trimmedLine.replace(/^\*\s*/, "").trim();
            if (/^\**\s*Meals:?\s*\**$/i.test(normalized)) {
              return (
                <div key={index} className="mt-4 mb-1">
                  <div className="bg-orange-50 border-l-8 border-orange-400 rounded-md px-3 py-1.5">
                    <h3 className="font-semibold text-gray-900 text-base">
                      Meals
                    </h3>
                  </div>
                </div>
              );
            }
          }

          // Time entries (* **9:00 AM:** Description)
          if (trimmedLine.match(/^\*\s*\*\*\d{1,2}:\d{2}\s*(AM|PM):\*\*/)) {
            const timeMatch = trimmedLine.match(
              /^\*\s*\*\*(\d{1,2}:\d{2}\s*(AM|PM)):\*\*(.*)/,
            );
            if (timeMatch) {
              const time = timeMatch[1];
              const description = (timeMatch[3] || "").trim();

              return (
                <div key={index} className="flex items-start gap-3 py-1.5 ml-4">
                  <span className="w-2 h-2 bg-gray-300 rounded-full mt-2 flex-shrink-0"></span>
                  <div className="flex-1">
                    <div className="font-bold text-gray-900 mb-1">{time}:</div>
                    <div className="text-gray-600 leading-relaxed">
                      <ReactMarkdown
                        components={{
                          p: ({ children }) => (
                            <span>{renderWithPoiHover(children, true)}</span>
                          ),
                          strong: StrongWithPoiHover,
                        }}
                      >
                        {stripPriceBold(description)}
                      </ReactMarkdown>
                    </div>
                  </div>
                </div>
              );
            }
          }

          // Regular bullet points (* Item)
          if (trimmedLine.match(/^\*\s*\*\*.*\*\*/)) {
            const content = stripPriceBold(trimmedLine.replace(/^\*\s+/, ""));
            return (
              <div key={index} className="flex items-start gap-3 py-0.5 ml-4">
                <span className="w-2 h-2 bg-gray-300 rounded-full mt-2 flex-shrink-0"></span>
                <div className="text-gray-600 leading-relaxed">
                  <ReactMarkdown
                    components={{
                      p: ({ children }) => (
                        <span>{renderWithPoiHover(children, true)}</span>
                      ),
                      strong: StrongWithPoiHover,
                    }}
                  >
                    {content}
                  </ReactMarkdown>
                </div>
              </div>
            );
          }

          // Generic bullets (e.g., "*   📍 Place To Bee: ...")
          if (/^\*\s+/.test(trimmedLine)) {
            const content = stripPriceBold(
              stripPoiBold(trimmedLine.replace(/^\*\s+/, "")),
            );
            return (
              <div key={index} className="flex items-start gap-3 py-0.5 ml-4">
                <span className="w-2 h-2 bg-gray-300 rounded-full mt-2 flex-shrink-0"></span>
                <div className="text-gray-600 leading-relaxed">
                  <ReactMarkdown
                    components={{
                      p: ({ children }) => (
                        <span>{renderWithPoiHover(children, true)}</span>
                      ),
                      strong: StrongWithPoiHover,
                    }}
                  >
                    {content}
                  </ReactMarkdown>
                </div>
              </div>
            );
          }

          // Preserve explicit blank lines for spacing
          if (!trimmedLine.length) {
            return <div key={index} className="h-2" />;
          }

          // Descriptive paragraphs (no markdown prefix)
          if (
            trimmedLine.match(/^[A-Z].*[.!]$/) &&
            !trimmedLine.includes("#")
          ) {
            return (
              <div
                key={index}
                className="mb-4 leading-relaxed italic text-gray-500 text-sm pl-4 border-l-2 border-gray-200"
              >
                <ReactMarkdown
                  components={{
                    p: ({ children }) => (
                      <span>{renderWithPoiHover(children, true)}</span>
                    ),
                    strong: StrongWithPoiHover,
                  }}
                >
                  {stripPriceBold(trimmedLine)}
                </ReactMarkdown>
              </div>
            );
          }

          // Regular paragraphs
          return (
            <div key={index} className="mb-2.5 leading-relaxed text-gray-700">
              <ReactMarkdown
                components={{
                  p: ({ children }) => (
                    <span>{renderWithPoiHover(children, true)}</span>
                  ),
                  strong: StrongWithPoiHover,
                }}
              >
                {stripPriceBold(trimmedLine)}
              </ReactMarkdown>
            </div>
          );
        })}

        {extractedQuestions.length > 0 && (
          <div className="mt-6">
            <h3 className="text-md font-bold mb-2">{questionHeader}</h3>
            <div className="flex flex-col space-y-2 mt-2">
              {extractedQuestions.map((q, i) => (
                <button
                  key={i}
                  className="bg-white/80 border border-zinc-200 rounded-lg px-3 py-2 text-sm text-left hover:bg-zinc-50 transition-colors"
                  onClick={() => onSendMessage(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim()) {
      setUserJustSent(true); // Hide suggestions when user sends a message
      setShowSuggestion(false);

      // Response type will be set by server-detected intent
      onSendMessage(inputValue.trim());
      handleInputChange(""); // Clear both internal and external input
    }
  };

  return (
    <div className="relative flex h-full flex-col bg-transparent">
      {planningState?.status === "unavailable" ? (
        <div className="border-b border-amber-200 bg-amber-50 px-6 py-3 text-sm text-amber-900">
          AI planning is temporarily unavailable. Roameo kept your last accepted trip visible until the provider recovers.
        </div>
      ) : null}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className={`flex-1 overflow-y-auto ${isRightPanelVisible ? "px-6 pb-32 pt-20" : "w-full px-10 pb-32 pt-20 sm:px-16 lg:px-24"}`}
      >
        <div className={`w-full ${isRightPanelVisible ? "max-w-full" : "mx-auto max-w-[760px]"}`}>
          {visibleMessages.map((message, i) => (
            <div
              key={message.id || message.createdAt || i}
              className={`mb-6 flex items-start gap-3 ${
                message.role === "user"
                  ? "justify-end"
                  : "justify-start"
              }`}
            >
              <Avatar
                className={`h-8 w-8 flex-shrink-0 ${message.role === "user" ? "order-2" : "order-1"}`}
              >
                <AvatarFallback
                  className={
                    message.role === "user"
                      ? "bg-[#4f8df7] text-white"
                      : "bg-black text-white flex items-center justify-center"
                  }
                >
                  {message.role === "user" ? (
                    "N"
                  ) : (
                    <div className="w-2 h-2 bg-white rounded-full"></div>
                  )}
                </AvatarFallback>
              </Avatar>
              <div
                className={`${
                  message.role === "user"
                    ? "order-1 max-w-[calc(100%-44px)]"
                    : "order-2 flex-1 max-w-[calc(100%-44px)]"
                }`}
              >
                <div
                  className={`max-w-none rounded-[20px] border ${
                    message.role === "user"
                      ? `ml-auto w-fit min-w-[72px] border-[#d9d9d9] bg-[#e5e5e5] px-[21px] py-[17px] text-[#1a1a1a] shadow-[0_8px_24px_rgba(0,0,0,0.10),0_2px_8px_rgba(0,0,0,0.06)] ${
                          isRightPanelVisible ? "max-w-[400px]" : "max-w-[430px]"
                        }`
                      : `${isRightPanelVisible ? "max-w-[76%]" : "max-w-[72%]"} border-[#e5e5e5] bg-white px-5 py-4 shadow-[0_8px_24px_rgba(0,0,0,0.10),0_2px_8px_rgba(0,0,0,0.06)]`
                  }`}
                >
                  <div className="text-[14.875px] font-normal leading-[27px]">
                    {message.role === "user" ? (
                      <ReactMarkdown>
                        {String(message.content)
                          .replace(/\\[object Object\\]/g, "")
                          .trim()}
                      </ReactMarkdown>
                    ) : message.meta?.responseBlocks?.length ? (
                      <StructuredResponseBlocks
                        message={message}
                        pois={pois}
                        savedIds={savedIds}
                        itineraryPoiIds={itineraryPoiIds}
                        onQuickAction={(prompt) => {
                          onPopulateInput?.(prompt);
                        }}
                        onToggleSave={onToggleSave}
                        onAddPoi={(poi) => onAddPoi?.(poi)}
                        onReplan={onReplan}
                      />
                    ) : (
                      renderFormattedContent(
                        String(message.content).replace(
                          /\\[object Object\\]/g,
                          "",
                        ),
                      )
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}

          {/* Suggested Questions Component */}
          {lastAssistant && showSuggestion && (
            <div className="mt-6 flex justify-start pl-12">
              <div className="text-sm text-gray-600">
                <p className="mb-3 font-medium">
                  Want me to customize your plan? Try:
                </p>
                <div className="flex flex-wrap gap-2">
                  {(() => {
                    const pool: Array<{ label: string; prompt: string }> = [
                      {
                        label: "🚗 Transportation options",
                        prompt:
                          "What's the best way to travel within the city?",
                      },
                      {
                        label: "🍽️ Local cuisine",
                        prompt:
                          "Can you suggest local food specialties I should try?",
                      },
                      {
                        label: "🏛️ Cultural sites",
                        prompt: "What are the must-visit cultural attractions?",
                      },
                      {
                        label: "🎉 Events & festivals",
                        prompt:
                          "Are there any local festivals or events during my visit?",
                      },
                      {
                        label: "⏰ Best visiting times",
                        prompt:
                          "What's the best time to visit popular attractions to avoid crowds?",
                      },
                      {
                        label: "💎 Hidden gems",
                        prompt:
                          "Can you recommend some hidden gems or off-the-beaten-path places?",
                      },
                      {
                        label: "🎿 Adventure activities",
                        prompt:
                          "Could you add adventure activities to the plan?",
                      },
                      {
                        label: "👨‍👩‍👧‍👦 Family-friendly tips",
                        prompt:
                          "What are some family-friendly tips and places?",
                      },
                      {
                        label: "🥘 Must-try dishes",
                        prompt: "Which local dishes should I not miss?",
                      },
                      {
                        label: "📅 Day-by-day tweaks",
                        prompt: "Can you tweak Day 2 to be more relaxed?",
                      },
                      {
                        label: "💰 Budget optimizations",
                        prompt:
                          "How can we reduce the overall budget without losing experiences?",
                      },
                      {
                        label: "🕒 Time-saving routes",
                        prompt:
                          "Can you optimize the route to save travel time?",
                      },
                      {
                        label: "📸 Photo spots",
                        prompt: "Where are the best photo spots?",
                      },
                      {
                        label: "🌦️ Weather prep",
                        prompt: "What should I pack given the typical weather?",
                      },
                      {
                        label: "🛍️ Shopping",
                        prompt:
                          "What are the best places to shop for souvenirs?",
                      },
                      {
                        label: "🚶 Walkability",
                        prompt: "Which parts of the itinerary are walkable?",
                      },
                      {
                        label: "🕰️ Opening hours",
                        prompt:
                          "Do any attractions require booking or have tight hours?",
                      },
                      {
                        label: "♿ Accessibility",
                        prompt:
                          "Can you adjust for accessibility considerations?",
                      },
                    ];
                    // Shuffle and pick a subset (6-9)
                    const shuffled = [...pool].sort(() => Math.random() - 0.5);
                    const count = Math.floor(6 + Math.random() * 4);
                    return shuffled.slice(0, count).map((item, idx) => (
                      <button
                        key={idx}
                        className="rounded-full border border-zinc-200 bg-white/80 px-3 py-2 text-xs hover:bg-zinc-50 transition-colors"
                        onClick={() => onSendMessage(item.prompt)}
                      >
                        {item.label}
                      </button>
                    ));
                  })()}
                </div>
              </div>
            </div>
          )}

          {/* Show typing indicator when typing OR when planning is active (even with recent messages) */}
          {(isTyping || planningActive) &&
            (() => {
              // Safety check: Don't show typing if we just received an assistant message
              const lastMessage = visibleMessages[visibleMessages.length - 1];
              const recentAssistantMessage =
                lastMessage &&
                lastMessage.role === "assistant" &&
                lastMessage.createdAt &&
                Date.now() - new Date(lastMessage.createdAt).getTime() < 3000; // reduced to 3 seconds

              // Always show planning animation when planning is active, regardless of recent messages
              if (planningActive || detectedIntent === "PLAN_TRIP") {
                console.log(
                  "[chat-interface] Showing planning animation - planningActive:",
                  planningActive,
                  "intent:",
                  detectedIntent,
                );
                // Show planning animation even if there are recent messages
              } else if (recentAssistantMessage) {
                console.log(
                  "[chat-interface] Suppressing typing indicator due to recent assistant message",
                );
                return null;
              }

              return (
                <div className="mb-6 flex items-start gap-3">
                  <Avatar className="w-8 h-8 flex-shrink-0">
                    <AvatarFallback className="bg-black text-white flex items-center justify-center">
                      <div className="w-2 h-2 bg-white rounded-full"></div>
                    </AvatarFallback>
                  </Avatar>
                  <div className="max-w-[calc(100%-44px)] flex-1">
                    <div className="max-w-none rounded-[20px] border border-[#e5e7eb] bg-white px-5 py-4 shadow-[0_8px_32px_rgba(15,23,42,0.12)]">
                      <div className="leading-relaxed text-sm">
                        {responseType === "planning" || planningActive ? (
                          <InlinePlanningStatus isVisible={true} />
                        ) : (
                          <TypingIndicator isVisible={true} />
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })()}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {showScrollToBottom && (
        <Button
          onClick={() => scrollToBottom()}
          variant="outline"
          className={`absolute bottom-28 z-10 h-11 w-11 rounded-full border border-white/70 bg-white/88 p-0 text-[#1f1b16] shadow-[0_12px_24px_rgba(15,23,42,0.12)] backdrop-blur-xl transition-all hover:-translate-y-0.5 hover:bg-white ${
            isRightPanelVisible ? "right-7" : "right-12"
          }`}
        >
          <ArrowDown className="h-4 w-4" />
        </Button>
      )}

      <div className="absolute bottom-6 left-0 right-0 z-20 flex justify-center pointer-events-none px-6 w-full">


        <form
          onSubmit={handleSubmit}
          className={`pointer-events-auto w-full ${isRightPanelVisible ? "max-w-[85%]" : "max-w-[760px]"}`}
        >
          <div className="h-[64px] rounded-[999px] border border-gray-200/60 bg-white/70 backdrop-blur-xl px-[24px] shadow-[0_8px_32px_rgba(0,0,0,0.08)]">
            <div className="flex h-full flex-row items-center gap-3">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-[18px] w-[18px] flex-shrink-0 rounded-full p-0 text-[#6b7280] hover:bg-transparent"
              >
                <Plus className="h-[18px] w-[18px]" />
              </Button>
              <Input
                value={inputValue}
                onChange={(e) => handleInputChange(e.target.value)}
                placeholder="Ask anything..."
                className="flex-1 border-0 bg-transparent px-0 text-[16px] font-medium text-[#1a1a1a] placeholder:text-[#9ca3af] focus-visible:ring-0 focus-visible:ring-offset-0 shadow-none"
              />
              <div className="flex items-center gap-2 flex-shrink-0">
                <Button
                  type="submit"
                  size="sm"
                  className="h-[34px] w-[34px] rounded-full bg-black text-white hover:bg-black/90"
                >
                  <Send className="h-[15px] w-[15px]" />
                </Button>
              </div>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
