"use client";

import React, { type ReactNode, isValidElement, cloneElement } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Send, Plus, ArrowDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { sendChat } from "@/lib/api";
import { AgenticStatus } from "./agentic-status";
import { StructuredResponseBlocks } from "./structured-response-blocks";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { CachedImage } from "./cached-image";
import { CompactPoiCard } from "./poi-card";
import { PoiTypeIcon } from "./poi-type-icon";
import type { AgentTraceEvent, ChatMessage, POI, SessionPlanningState } from "@/lib/types";

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

function isPlanMutationNote(message: ChatMessage) {
  return (
    message.role === "assistant" &&
    message.meta?.source === "plan-mutation" &&
    !message.meta?.responseBlocks?.length
  );
}

function isStructuredAssistantMessage(message: ChatMessage) {
  return message.role === "assistant" && Boolean(message.meta?.responseBlocks?.length);
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
  onSlotAction?: (action: { field: string; value: string | number }, prompt: string) => void;
  inputValue?: string;
  onInputChange?: (value: string) => void;
  /** Live agent trace events from the current session */
  traces?: AgentTraceEvent[];
  /** The currently active turn ID (streaming) */
  activeTurnId?: string;
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
  onSlotAction,
  inputValue: externalInputValue,
  onInputChange,
  traces,
  activeTurnId,
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
  const [streamingRender, setStreamingRender] = useState<{
    id: string | null;
    content: string;
  }>({
    id: null,
    content: "",
  });
  const isSubmittingRef = useRef(false);
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
  const activeStreamingAssistant =
    lastMessage?.role === "assistant" && !lastMessage.meta?.responseBlocks?.length
      ? lastMessage
      : undefined;
  const hasActiveStreamingResponse =
    Boolean(activeStreamingAssistant?.id) &&
    Boolean(activeStreamingAssistant?.content?.trim().length);

  useEffect(() => {
    if (!activeStreamingAssistant) {
      setStreamingRender((current) =>
        current.id === null && current.content === ""
          ? current
          : { id: null, content: "" }
      );
      return;
    }

    const targetId = activeStreamingAssistant.id;
    const targetContent = String(activeStreamingAssistant.content || "");
    if (!targetId) {
      return;
    }

    setStreamingRender((current) => {
      if (current.id !== targetId) {
        return {
          id: targetId,
          content: isTyping ? "" : targetContent,
        };
      }

      if (!isTyping || current.content.length >= targetContent.length) {
        return current.content === targetContent
          ? current
          : { id: targetId, content: targetContent };
      }

      return current;
    });

    if (!isTyping) {
      return;
    }

    const interval = window.setInterval(() => {
      setStreamingRender((current) => {
        if (current.id !== targetId) {
          return current;
        }

        if (current.content.length >= targetContent.length) {
          return current;
        }

        const remaining = targetContent.length - current.content.length;
        const nextChars = Math.min(remaining, Math.max(1, Math.ceil(remaining / 12)));

        return {
          id: targetId,
          content: targetContent.slice(0, current.content.length + nextChars),
        };
      });
    }, 24);

    return () => window.clearInterval(interval);
  }, [activeStreamingAssistant?.id, activeStreamingAssistant?.content, isTyping]);

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

  // Gate the response type on planningActive OR isTyping — but never on detectedIntent alone,
  // because intent persists across turns and would keep the animation alive forever.
  useEffect(() => {
    if (planningActive) {
      setResponseType("planning");
    } else if (isTyping) {
      if (
        detectedIntent === "PLAN_TRIP" ||
        detectedIntent === "DESTINATION_SEARCH"
      ) {
        setResponseType("planning");
      } else {
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
              className="font-bold text-slate-900 cursor-pointer hover:underline"
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
      <strong className="font-bold text-slate-900">
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
                      <span className="inline-flex items-center gap-1">
                        <PoiTypeIcon poi={matchingPoi} className="h-[1em] w-[1em] shrink-0" />
                        <span>{poiName}</span>
                      </span>
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
                <span className="inline-flex items-center gap-1">
                  <PoiTypeIcon
                    poi={matchingPoi ? matchingPoi : { type: "destination", name: poiName, tags: [] }}
                    className="h-[1em] w-[1em] shrink-0"
                  />
                  <span>{poiName}</span>
                </span>
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

          // Main title (# *Title*)
          if (trimmedLine.match(/^#\s*\*.*\*$/) || trimmedLine.match(/^#\s+(.+)$/)) {
            const title = trimmedLine.replace(/^#\s*\*(.+)\*$/, "$1").replace(/^#\s+(.+)$/, "$1");
            return (
              <h1
                key={index}
                className="text-2xl font-semibold tracking-tight text-slate-900 mb-2.5"
              >
                {renderWithPoiHover(title, true)}
              </h1>
            );
          }

          // Day headers (## *Day 1:*) or regular headers (## Day 1)
          if (trimmedLine.match(/^##\s*\*.*\*$/) || trimmedLine.match(/^##\s+(.+)$/)) {
            const dayTitle = trimmedLine.replace(/^##\s*\*(.+)\*$/, "$1").replace(/^##\s+(.+)$/, "$1");
            return (
              <h2
                key={index}
                className="text-lg font-semibold tracking-tight text-slate-900 mb-2 mt-5"
              >
                {renderWithPoiHover(dayTitle, true)}
              </h2>
            );
          }

          // Time section headers (### ☀️ Morning)
          if (
            trimmedLine.match(
              /^###\s*(☀️|⛵|🌅|🚗|🛍️|🍽️|🏨|✈️)\s*(Morning|Afternoon|Evening|Shopping|Heritage|Departure)/,
            )
          ) {
            const sectionTitle = trimmedLine.replace(/^###\s*/, "");
            return (
              <div key={index} className="mt-4 mb-2">
                <h3 className="text-base font-semibold text-slate-900">
                  {renderWithPoiHover(sectionTitle, true)}
                </h3>
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
              <div key={index} className="mt-4 mb-2">
                <h3 className="text-base font-semibold text-slate-900">
                  {renderWithPoiHover(sectionTitle, true)}
                </h3>
              </div>
            );
          }

          // Meals subheading emitted as a bullet (e.g., "*  Meals:" or "**Meals:**") — render as banner
          {
            const normalized = trimmedLine.replace(/^\*\s*/, "").trim();
            if (/^\**\s*Meals:?\s*\**$/i.test(normalized)) {
              return (
                <div key={index} className="mt-4 mb-1">
                  <h3 className="font-semibold text-slate-900 text-[16px]">
                    Meals
                  </h3>
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
                    <div className="text-sm leading-6 text-neutral-600">
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
                <div className="text-sm leading-6 text-neutral-600">
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
                <div className="text-sm leading-6 text-neutral-600">
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

          // Standalone bold heading: **Day 1: Title** or **Title**
          // The AI often emits these instead of ## headings
          if (/^\*\*[^*]+\*\*$/.test(trimmedLine)) {
            const heading = trimmedLine.replace(/^\*\*(.*)\*\*$/, "$1").trim();
            return (
              <h2
                key={index}
                className="text-lg font-semibold tracking-tight text-slate-900 mb-2 mt-5"
              >
                {renderWithPoiHover(heading, true)}
              </h2>
            );
          }

          // Descriptive paragraphs (no markdown prefix)
          if (
            trimmedLine.match(/^[A-Z].*[.!]$/) &&
            !trimmedLine.includes("#")
          ) {
            return (
              <div
                key={index}
                className="mb-4 italic text-sm leading-6 text-slate-500 pl-4 border-l-2 border-slate-200"
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

          // Regular paragraph text
          return (
            <p key={index} className="text-sm leading-6 text-neutral-600">
              {renderWithPoiHover(trimmedLine)}
            </p>
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
    if (!inputValue.trim() || isSubmittingRef.current) {
      return;
    }
    isSubmittingRef.current = true;
    setUserJustSent(true); // Hide suggestions when user sends a message
    setShowSuggestion(false);

    // Response type will be set by server-detected intent
    onSendMessage(inputValue.trim());
    handleInputChange(""); // Clear both internal and external input
    // Reset after a brief delay to allow the async send to start
    setTimeout(() => {
      isSubmittingRef.current = false;
    }, 800);
  };

  return (
    <div className="relative flex h-full flex-col bg-transparent">

      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className={`flex-1 overflow-y-auto ${isRightPanelVisible ? "px-6 pb-32 pt-20" : "w-full px-10 pb-32 pt-20 sm:px-16 lg:px-24"}`}
      >
        <div className={`w-full ${isRightPanelVisible ? "max-w-full" : "mx-auto max-w-[760px]"}`}>
          {visibleMessages.map((message, i) => {
            if (isPlanMutationNote(message)) {
              return (
                <div
                  key={message.id || message.createdAt || i}
                  className="mb-4 pl-11"
                >
                  <div className="max-w-[680px] text-[13px] italic leading-6 text-slate-500">
                    {normalizeRenderableMessageContent(String(message.content || ""))}
                  </div>
                </div>
              );
            }

            if (isStructuredAssistantMessage(message)) {
              return (
                <div
                  key={message.id || message.createdAt || i}
                  className="mb-8 flex items-start gap-3"
                >
                  <Avatar className="h-8 w-8 flex-shrink-0">
                    <AvatarFallback className="bg-black text-white flex items-center justify-center">
                      <div className="w-2 h-2 rounded-full bg-white" />
                    </AvatarFallback>
                  </Avatar>
                  <div className="max-w-[calc(100%-44px)] flex-1">
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
                      onSlotAction={onSlotAction}
                    />
                  </div>
                </div>
              );
            }

            return (
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
                    className={`max-w-none ${
                      message.role === "user"
                        ? `rounded-[24px] ml-auto w-fit min-w-[72px] bg-white px-5 py-4 text-[#1a1a1a] shadow-[0_28px_48px_rgba(15,23,42,0.12),0_12px_24px_rgba(15,23,42,0.08),0_2px_8px_rgba(15,23,42,0.05),inset_0_1px_0_rgba(255,255,255,0.9)] ${
                            isRightPanelVisible ? "max-w-[400px]" : "max-w-[430px]"
                          }`
                        : `${isRightPanelVisible ? "max-w-[76%]" : "max-w-[72%]"} pt-1`
                    }`}
                  >
                    <div className={message.role === "user" ? "text-[15px] leading-7 text-slate-800" : ""}>
                      {message.role === "user" ? (
                        <ReactMarkdown className="prose prose-slate max-w-none text-[15px] leading-[1.6] text-slate-800 prose-p:my-1 prose-strong:font-semibold prose-strong:text-slate-900">
                          {String(message.content)
                            .replace(/\[object Object\]/g, "")
                            .trim()}
                        </ReactMarkdown>
                      ) : (
                        renderFormattedContent(
                          (
                            message.role === "assistant" &&
                            message.id &&
                            streamingRender.id === message.id
                              ? streamingRender.content
                              : String(message.content)
                          ).replace(/\[object Object\]/g, ""),
                        )
                      )}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}

          {/* Contextual follow-up prompts — only for non-structured replies */}
          {lastAssistant && showSuggestion && (
            <div className="mt-5 flex justify-start pl-12">
              <div className="flex flex-wrap gap-2">
                {[
                  { label: "Tweak the schedule", prompt: "Can you adjust the itinerary to be more relaxed with fewer stops per day?" },
                  { label: "Add local food spots", prompt: "Suggest specific restaurants or street food spots near each day's activities." },
                  { label: "Cut costs", prompt: "How can I reduce the budget without dropping key experiences?" },
                  { label: "Off-the-beaten-path", prompt: "Replace one mainstream stop each day with something locals would recommend." },
                ].map((item) => (
                  <button
                    key={item.label}
                    type="button"
                    className="rounded-full border border-slate-200 bg-white px-4 py-2 text-[13px] font-medium text-slate-700 transition-colors hover:border-slate-300 hover:bg-slate-50"
                    onClick={() => onSendMessage(item.prompt)}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Agentic status / typing indicator
           *
           * Visibility rules (in priority order):
           *  1. planningActive=true  → always show (backend is still working)
           *  2. isTyping=true + no recent assistant message → show generic indicator
           *  3. Everything else → hide
           *
           * The "detectedIntent === PLAN_TRIP" bypass has been removed.
           * planningActive is derived from planningState.status==='running' on
           * the session snapshot and is reset to 'ready' by turn-runner.ts as
           * soon as the turn completes, so this correctly terminates.
           */}
          {(planningActive || isTyping) &&
            (() => {
              if (hasActiveStreamingResponse) {
                return null;
              }

              // If planning is no longer active, check whether a very recent
              // assistant message was just committed — if so, suppress the
              // indicator so there's no "flash" between message arrival and
              // planningActive going false.
              if (!planningActive) {
                const lastMessage = visibleMessages[visibleMessages.length - 1];
                const recentAssistantMessage =
                  lastMessage &&
                  lastMessage.role === "assistant" &&
                  lastMessage.createdAt &&
                  Date.now() - new Date(lastMessage.createdAt).getTime() < 2000;

                if (recentAssistantMessage) {
                  return null;
                }
              }

              return (
                <div className="mb-6 flex items-start gap-3">
                  <Avatar className="w-8 h-8 flex-shrink-0">
                    <AvatarFallback className="bg-black text-white flex items-center justify-center">
                      <div className="w-2 h-2 bg-white rounded-full"></div>
                    </AvatarFallback>
                  </Avatar>
                  <div className="max-w-[calc(100%-44px)] flex-1 pt-1">
                    <AgenticStatus
                      traces={traces}
                      turnId={activeTurnId}
                      mode={planningActive ? "planning" : "general"}
                    />
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
