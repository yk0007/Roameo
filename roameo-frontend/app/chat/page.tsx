"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase/client";
import { TopNavigation } from "@/components/top-navigation";
import { ChatInterface } from "@/components/chat-interface";
import { SearchInterface } from "@/components/search-interface";
import { RightPanel } from "@/components/right-panel";
import { LeftPanelTabs } from "@/components/left-panel-tabs";
import { Button } from "@/components/ui/button";
import {
  MessageCircle,
  Search,
  Bookmark,
  Map as MapIcon,
  Calendar,
} from "lucide-react";
import { connectWs } from "@/lib/ws";
import {
  sendChat,
  tripUpdate,
  createInvite,
  deleteTrip as apiDeleteTrip,
  savePoi as apiSavePoi,
  getSavedPoiIds,
} from "@/lib/api";
import { toast } from "@/hooks/use-toast";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import type {
  ChatMessage,
  Itinerary,
  SearchResults,
  TripContext,
  WsEvent,
  POI,
} from "@/lib/types";

export default function ChatPage() {
  const [activeLeftView, setActiveLeftView] = useState<
    "chat" | "search" | "saved"
  >("chat");
  const [activeRightView, setActiveRightView] = useState<"map" | "itinerary">(
    "map",
  );
  const [isRightPanelVisible, setIsRightPanelVisible] = useState(true);

  const searchParams = useSearchParams();
  const router = useRouter();

  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [inviteId, setInviteId] = useState<string | undefined>(undefined);
  const wsRef = useRef<WebSocket | null>(null);

  const [trip, setTrip] = useState<TripContext>({
    sessionId: "temp",
    title: undefined,
    origin: undefined,
    destination: undefined,
    days: undefined,
    travelers: undefined,
    budget: undefined,
  });

  // Enhanced message state management with better deduplication and immediate updates
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const seenMessageIdsRef = useRef<Set<string>>(new Set());
  const [itinerary, setItinerary] = useState<Itinerary | undefined>(undefined);
  const [searchResults, setSearchResults] = useState<SearchResults | undefined>(
    undefined,
  );
  const [mapData, setMapData] = useState<any>(undefined);

  // Enhanced debugging for itinerary and map data
  useEffect(() => {
    console.log(
      "[client] Itinerary state updated:",
      itinerary ? `${itinerary.daysPlan?.length || 0} days` : "undefined",
    );
    if (itinerary) {
      console.log(
        "[client] Itinerary details:",
        JSON.stringify(itinerary, null, 2),
      );
    }
  }, [itinerary]);

  useEffect(() => {
    console.log(
      "[client] Map data state updated:",
      mapData
        ? `${mapData.pois?.length || 0} POIs, ${mapData.routes?.length || 0} routes`
        : "undefined",
    );
    if (mapData) {
      console.log(
        "[client] Map data details:",
        JSON.stringify(mapData, null, 2),
      );
    }
  }, [mapData]);

  // Hydrate itinerary and map from sessionStorage to avoid losing UI state on reconnects or clarifications
  useEffect(() => {
    if (!sessionId) return;
    try {
      const itinRaw = sessionStorage.getItem(`itin:${sessionId}`);
      if (itinRaw && !itinerary) {
        const parsed = JSON.parse(itinRaw);
        if (parsed?.daysPlan?.length) {
          console.log("[client] Hydrating itinerary from sessionStorage");
          setItinerary(parsed);
        }
      }
      const mapRaw = sessionStorage.getItem(`map:${sessionId}`);
      if (mapRaw && !mapData) {
        const parsed = JSON.parse(mapRaw);
        if (parsed?.pois) {
          console.log("[client] Hydrating mapData from sessionStorage");
          setMapData(parsed);
        }
      }
    } catch {}
  }, [sessionId]);

  useEffect(() => {
    console.log(
      "[client] Search results state updated:",
      searchResults
        ? `${(searchResults.stays?.length || 0) + (searchResults.restaurants?.length || 0) + (searchResults.attractions?.length || 0)} total POIs`
        : "undefined",
    );
  }, [searchResults]);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [detectedIntent, setDetectedIntent] = useState<
    "PLAN_TRIP" | "DESTINATION_SEARCH" | "CHAT" | null
  >(null);
  const planningTimeoutRef = useRef<number | null>(null);
  const planningActiveRef = useRef(false); // Guard to prevent duplicate planning animations
  const planningReplyInjectedRef = useRef(false); // Track whether we've shown a reply for current planning
  const [isPlanning, setIsPlanning] = useState(false); // Mirror ref in state for UI
  const [savedPoiIds, setSavedPoiIds] = useState<Set<string>>(new Set());
  const [inputMessage, setInputMessage] = useState<string>(""); // Add state for controlling input
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<number | null>(null);
  const messagesRef = useRef<ChatMessage[]>([]);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);
  const [authChecked, setAuthChecked] = useState(false);
  const initialMessageHandledRef = useRef(false);
  const lastUserSentRef = useRef<{ content: string; at: number } | null>(null);
  const hadDisconnectRef = useRef(false);
  const manualCloseRef = useRef(false);
  const connectingRef = useRef(false);

  // Enhanced debug logging for messages state changes
  useEffect(() => {
    console.log(
      "[client] Messages state updated, total count:",
      messages.length,
    );
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      console.log("[client] Last message:", {
        id: lastMessage.id,
        role: lastMessage.role,
        contentLength: lastMessage.content?.length,
      });

      // Force typing state clear if we have a recent assistant message
      if (lastMessage.role === "assistant" && isTyping) {
        console.log(
          "[client] Force clearing typing state due to new assistant message",
        );
        setIsTyping(false);
        setDetectedIntent(null);
        planningActiveRef.current = false;
        if (planningTimeoutRef.current) {
          window.clearTimeout(planningTimeoutRef.current);
          planningTimeoutRef.current = null;
        }
      }
    }
  }, [messages, isTyping]);

  // Require login to access chat
  useEffect(() => {
    let mounted = true;
    const check = async () => {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session) {
        router.replace("/auth/login");
        return;
      }
      if (mounted) setAuthChecked(true);
    };
    check();
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_evt, session) => {
      if (!session) router.replace("/auth/login");
    });
    return () => {
      mounted = false;
      subscription.unsubscribe();
    };
  }, [router]);

  // Filter POIs for map display - only show itinerary POIs
  const mapPois = useMemo(() => {
    if (!itinerary?.daysPlan?.length) return [];
    const itineraryPoiIds = new Set();
    itinerary.daysPlan.forEach((day: any) => {
      day.activities?.forEach((activity: any) => {
        if (activity.poiId) {
          itineraryPoiIds.add(activity.poiId);
        }
      });
    });
    // Get all POIs from search results and any existing map data
    const allPois = [
      ...(searchResults?.stays || []),
      ...(searchResults?.restaurants || []),
      ...(searchResults?.attractions || []),
      ...(mapData?.pois || []),
    ];
    return allPois.filter((poi: any) => itineraryPoiIds.has(poi.id)) || [];
  }, [searchResults, itinerary, mapData]);

  // Fallback: Geocode missing itinerary POIs and add them to mapData
  useEffect(() => {
    // Require itinerary and an API key
    if (!itinerary?.daysPlan?.length) return;
    const key = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY;
    if (!key) return;

    // Build set of existing POI ids from known sources
    const existing = new Set<string>();
    const existingPois: POI[] = [
      ...(searchResults?.stays || []),
      ...(searchResults?.restaurants || []),
      ...(searchResults?.attractions || []),
      ...(mapData?.pois || []),
    ];
    existingPois.forEach((p) => {
      if (p?.id) existing.add(p.id);
    });

    // Collect itinerary POIs missing from our sources
    const missing: { id: string; name?: string; address?: string }[] = [];
    itinerary.daysPlan.forEach((d: any) => {
      d.activities?.forEach((a: any) => {
        if (a?.poiId && !existing.has(a.poiId))
          missing.push({ id: a.poiId, name: a.name, address: a.address });
      });
      if (d.accommodation?.poiId && !existing.has(d.accommodation.poiId)) {
        missing.push({
          id: d.accommodation.poiId,
          name: d.accommodation.name,
          address: d.accommodation.address,
        });
      }
    });

    // Nothing to do
    if (missing.length === 0) return;

    // Limit requests per run to avoid quota spikes
    const toLookup = missing.slice(0, 3);
    let cancelled = false;

    (async () => {
      const resolved: POI[] = [];
      for (const m of toLookup) {
        if (cancelled) break;
        const q = encodeURIComponent(
          [m.name, m.address, trip?.destination].filter(Boolean).join(" "),
        );
        try {
          const resp = await fetch(
            `https://maps.googleapis.com/maps/api/geocode/json?address=${q}&key=${key}`,
          );
          const j = await resp.json();
          const first = j?.results?.[0];
          const loc = first?.geometry?.location;
          if (loc) {
            resolved.push({
              id: m.id,
              name: m.name || first.formatted_address || m.id,
              address: m.address || first.formatted_address,
              lat: loc.lat,
              lng: loc.lng,
              type: "attraction",
            } as any);
          }
        } catch (e) {
          console.warn("[client] Geocoding failed for", m, e);
        }
        // Small delay to be nice to the API
        await new Promise((r) => setTimeout(r, 200));
      }
      if (!cancelled && resolved.length > 0) {
        setMapData((prev: any) => ({
          ...(prev || {}),
          pois: [...(prev?.pois || []), ...resolved],
        }));
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [itinerary, searchResults, mapData, trip?.destination]);

  // Set of itinerary POI IDs for 'Added only' map filter
  const itineraryPoiIds = useMemo(() => {
    const ids = new Set<string>();
    if (itinerary?.daysPlan?.length) {
      itinerary.daysPlan.forEach((d) => {
        d.activities.forEach((a) => {
          if (a.poiId) ids.add(a.poiId);
        });
        if (d.accommodation?.poiId) ids.add(d.accommodation.poiId);
      });
    }
    return ids;
  }, [itinerary]);

  // Saved-only results for unified Saved UI using SearchInterface
  const savedResults: SearchResults | undefined = useMemo(() => {
    if (!savedPoiIds || savedPoiIds.size === 0)
      return {
        stays: [] as POI[],
        restaurants: [] as POI[],
        attractions: [] as POI[],
      };
    const all: POI[] = [
      ...(searchResults?.stays || []),
      ...(searchResults?.restaurants || []),
      ...(searchResults?.attractions || []),
      ...(mapData?.pois || []),
    ];
    if (all.length === 0)
      return {
        stays: [] as POI[],
        restaurants: [] as POI[],
        attractions: [] as POI[],
      };
    const uniq = new Map() as Map<string, POI>;
    all.forEach((p) => {
      if (savedPoiIds.has(p.id) && !uniq.has(p.id)) uniq.set(p.id, p);
    });
    const arr = Array.from(uniq.values());
    return {
      stays: arr.filter((p) => p.type === "stay"),
      restaurants: arr.filter((p) => p.type === "restaurant"),
      attractions: arr.filter((p) => p.type === "attraction"),
    };
  }, [savedPoiIds, searchResults, mapData]);

  // Restore sessionId from URL if present
  useEffect(() => {
    const fromQuery = searchParams.get("sessionId") || undefined;
    if (fromQuery && fromQuery !== sessionId) setSessionId(fromQuery);
  }, [searchParams, sessionId]);

  // Handle message parameter from dashboard - auto-send message for existing session
  useEffect(() => {
    const messageFromUrl = searchParams.get("message");
    if (
      messageFromUrl &&
      !initialMessageHandledRef.current &&
      authChecked &&
      sessionId
    ) {
      console.log(
        "[client] Processing message from URL for existing session:",
        messageFromUrl,
      );
      initialMessageHandledRef.current = true;

      // Mark this message as coming from dashboard to prevent duplicates
      const dashboardMessage: ChatMessage = {
        id: `dashboard-${Date.now()}`,
        role: "user",
        content: messageFromUrl.trim(),
        createdAt: new Date().toISOString(),
        fromDashboard: true as any,
      };

      // Add message to UI immediately and show thinking state
      setMessages((prev) => {
        console.log("[client] Adding dashboard message to existing messages");
        return [...prev, dashboardMessage];
      });

      // Set thinking state immediately
      console.log("[client] Setting thinking state for dashboard message");
      setIsTyping(true);
      setDetectedIntent(null);
      planningActiveRef.current = false;
      setIsPlanning(false);
      lastUserSentRef.current = {
        content: messageFromUrl.trim(),
        at: Date.now(),
      };

      // Send to backend
      sendChat({ sessionId, inviteId, message: messageFromUrl.trim() }).catch(
        (e: any) => {
          setIsTyping(false);
          toast({
            title: "Failed to send message",
            description: e?.message || "Please try again.",
            variant: "destructive",
          });
        },
      );

      // Clear URL parameter immediately to prevent re-sending
      setTimeout(() => {
        const newUrl = new URL(window.location.href);
        newUrl.searchParams.delete("message");
        window.history.replaceState({}, "", newUrl.toString());
      }, 100);
    }
  }, [
    searchParams,
    authChecked,
    sessionId,
    inviteId,
    initialMessageHandledRef,
  ]);

  // Handle message parameter from dashboard when no session exists - create new session
  useEffect(() => {
    const messageFromUrl = searchParams.get("message");
    if (
      messageFromUrl &&
      !sessionId &&
      !initialMessageHandledRef.current &&
      authChecked
    ) {
      console.log(
        "[client] Creating new session for dashboard message:",
        messageFromUrl,
      );
      initialMessageHandledRef.current = true;

      // Add user message to UI immediately and show thinking state
      const dashboardMessage: ChatMessage = {
        id: `dashboard-new-${Date.now()}`,
        role: "user",
        content: messageFromUrl.trim(),
        createdAt: new Date().toISOString(),
        fromDashboard: true as any,
      };

      console.log("[client] Setting initial message for new session");
      setMessages([dashboardMessage]);

      // Set thinking state immediately with multiple attempts
      console.log("[client] Setting thinking state for new session");
      setIsTyping(true);
      setDetectedIntent(null);
      planningActiveRef.current = false;
      setIsPlanning(false);
      lastUserSentRef.current = {
        content: messageFromUrl.trim(),
        at: Date.now(),
      };

      // Force UI updates to ensure immediate feedback
      setTimeout(() => {
        console.log("[client] Force updating thinking state");
        setIsTyping(true);
        setIsPlanning(false);
      }, 10);

      setTimeout(() => {
        console.log("[client] Final thinking state update");
        setIsTyping(true);
      }, 100);

      // Send message which will create a new session
      sendChat({ message: messageFromUrl.trim() })
        .then((response) => {
          if (response.sessionId) {
            console.log("[client] New session created:", response.sessionId);
            // Set session info which will trigger WebSocket connection
            setSessionId(response.sessionId);
            setInviteId(response.inviteId);

            // Clear URL parameter and add sessionId after a brief delay
            setTimeout(() => {
              const newUrl = new URL(window.location.href);
              newUrl.searchParams.delete("message");
              newUrl.searchParams.set("sessionId", response.sessionId);
              window.history.replaceState({}, "", newUrl.toString());
              console.log("[client] URL updated with new sessionId");
            }, 500);
          }
        })
        .catch((e: any) => {
          setIsTyping(false);
          console.error(
            "[client] Failed to create session from dashboard message:",
            e,
          );
          toast({
            title: "Failed to send message",
            description: e?.message || "Please try again.",
            variant: "destructive",
          });
        });
    }
  }, [searchParams, sessionId, authChecked, initialMessageHandledRef]);

  // Open WS when we have a sessionId
  useEffect(() => {
    if (!sessionId) return;
    if (wsRef.current) wsRef.current.close();

    const onEvent = (evt: WsEvent) => {
      if (evt.type === "session.ready") {
        setSessionId(evt.data.sessionId);
        setInviteId(evt.data.inviteId);
        // push sessionId to URL if missing
        const qp = new URLSearchParams(window.location.search);
        if (!qp.get("sessionId")) {
          qp.set("sessionId", evt.data.sessionId);
          router.replace(`?${qp.toString()}`);
        }
        // hydrate saved POIs for Saved tab
        (async () => {
          try {
            const res = await getSavedPoiIds(evt.data.sessionId);
            setSavedPoiIds(new Set(res.ids));
          } catch {}
        })();
        // nudge map to render immediately on first connection
        setTimeout(() => {
          try {
            window.dispatchEvent(new Event("resize"));
          } catch {}
        }, 100);
      } else if (evt.type === "chat.history") {
        console.log("[client] Received chat history:", evt.data?.length);
        // Merge history with existing messages while PRESERVING id-less messages.
        // Deduplicate strictly by message id. This prevents temporary assistant messages
        // (often emitted without id for PLAN_TRIP) from being dropped.
        setMessages((prev) => {
          const next = [...prev];
          const existingIds = new Set<string>();
          // Build a set of user messages we injected from the dashboard to avoid re-adding their echoes
          const dashboardContents = new Set(
            prev
              .filter(
                (m) => (m as any).fromDashboard === true && m.role === "user",
              )
              .map((m) => String(m.content).trim()),
          );
          for (const m of prev) {
            if (m?.id) existingIds.add(m.id);
          }

          for (const m of evt.data) {
            // Skip server echoes matching dashboard-sent user messages
            if (m.role === "user" && dashboardContents.has(m.content.trim())) {
              continue;
            }
            if (m?.id) {
              if (!existingIds.has(m.id)) {
                existingIds.add(m.id);
                next.push(m);
              }
            } else {
              // No id: append if not an exact duplicate of recent content-role pair
              const isDup = next
                .slice(-5)
                .some(
                  (pm) =>
                    pm.role === m.role &&
                    String(pm.content).trim() === String(m.content).trim(),
                );
              if (!isDup) next.push(m);
            }
          }

          // Sort by createdAt when available; stable for items without createdAt
          next.sort((a, b) =>
            (a.createdAt || "").localeCompare(b.createdAt || ""),
          );

          // Track seen ids for future de-dup
          next.forEach((m) => {
            if (m.id) seenMessageIdsRef.current.add(m.id);
          });
          return next;
        });
      } else if (evt.type === "chat.append") {
        console.log("[client] Received chat.append event:", evt.data);
        // Deduplicate messages by id to avoid duplicates on reconnect/replay
        setMessages((prevMessages) => {
          const id = evt.data?.id as string | undefined;
          console.log(
            "[client] Processing message with id:",
            id,
            "role:",
            evt.data.role,
            "content length:",
            evt.data?.content?.length,
          );

          if (id) {
            if (seenMessageIdsRef.current.has(id)) {
              console.log("[client] Skipping duplicate message:", id);
              return prevMessages;
            }
            seenMessageIdsRef.current.add(id);
          }

          // Check for any message with fromDashboard flag - these should never be duplicated
          const isDashboardMessage = prevMessages.some(
            (m) =>
              m.fromDashboard === true &&
              m.role === evt.data.role &&
              m.content.trim() === evt.data.content.trim(),
          );

          if (isDashboardMessage) {
            console.log("[client] Skipping duplicate of dashboard message");
            return prevMessages;
          }

          // Enhanced heuristic: suppress immediate server echo of the same user message we just appended locally
          if (evt.data.role === "user" && lastUserSentRef.current) {
            const withinWindow = Date.now() - lastUserSentRef.current.at < 6000; // Extended to 6 seconds
            const contentMatch =
              evt.data.content.trim() ===
              lastUserSentRef.current.content.trim();

            // Also check if we already have this exact message content in recent messages
            const recentUserMessages = prevMessages
              .slice(-3)
              .filter((m) => m.role === "user");
            const alreadyHasContent = recentUserMessages.some(
              (m) => m.content.trim() === evt.data.content.trim(),
            );

            if ((withinWindow && contentMatch) || alreadyHasContent) {
              console.log(
                "[client] Suppressing user message echo - withinWindow:",
                withinWindow,
                "contentMatch:",
                contentMatch,
                "alreadyHasContent:",
                alreadyHasContent,
              );
              return prevMessages;
            }
          }

          const newMessages = [...prevMessages, evt.data];
          console.log(
            "[client] Adding message to state, new total:",
            newMessages.length,
          );

          // For assistant messages, show immediately but handle planning state smartly
          if (evt.data.role === "assistant") {
            console.log(
              "[client] Processing assistant message - showing immediately",
            );

            // Always clear typing for non-planning responses
            if (!planningActiveRef.current) {
              setIsTyping(false);
              setDetectedIntent(null);
              setIsPlanning(false);
            } else {
              // For planning responses, keep animation until itinerary arrives
              console.log(
                "[client] Planning active - keeping animation for planning response",
              );
              // Don't clear planning states yet - wait for itinerary
            }

            planningReplyInjectedRef.current = true;
            // Safety: if another concurrent update overwrote messages, re-assert presence of this assistant message
            const msgId = id;
            const msgCopy = { ...evt.data };
            setTimeout(() => {
              setMessages((prev) => {
                if (msgId && !prev.some((m) => m.id === msgId)) {
                  console.warn(
                    "[client] Assistant message missing after update, re-inserting:",
                    msgId,
                  );
                  seenMessageIdsRef.current.add(msgId);
                  return [...prev, msgCopy];
                }
                return prev;
              });
            }, 5);
          }

          return newMessages;
        });

        // Additional assistant message handling for immediate display
        if (evt.data.role === "assistant") {
          console.log(
            "[client] Assistant message displayed - checking planning state",
          );

          // Only clear states if we're not in an active planning cycle
          if (!planningActiveRef.current) {
            console.log(
              "[client] No active planning - clearing thinking states",
            );
            setIsTyping(false);
            setIsPlanning(false);
            setDetectedIntent(null);

            if (planningTimeoutRef.current) {
              window.clearTimeout(planningTimeoutRef.current);
              planningTimeoutRef.current = null;
            }
          } else {
            console.log(
              "[client] Planning active - message shown but keeping animation",
            );
            // Message is shown, but keep planning animation until itinerary completes
            // SAFETY: If itinerary doesn't arrive within 8s, clear thinking states
            if (planningTimeoutRef.current) {
              window.clearTimeout(planningTimeoutRef.current);
              planningTimeoutRef.current = null;
            }
            planningTimeoutRef.current = window.setTimeout(() => {
              console.warn("[client] Fallback: clearing planning state due to timeout after assistant message");
              planningActiveRef.current = false;
              setIsTyping(false);
              setIsPlanning(false);
              setDetectedIntent(null);
              planningTimeoutRef.current = null;
            }, 8000);
          }
        }
      } else if (evt.type === "navbar.update") {
        setTrip((t) => ({ ...t, ...evt.data }));
      } else if (evt.type === "itinerary.update") {
        // When itinerary is received, it means planning is complete
        console.log(
          "[client] Itinerary received - planning should be complete",
        );
        const data = evt.data;
        if (data && typeof data === "object" && (data as any).daysPlan) {
          // Valid itinerary arrived – set it and NOW clear all planning UI
          const newItin = data as Itinerary;
          console.log(
            `[client] Itinerary complete with ${newItin.daysPlan?.length || 0} days - clearing all planning states`,
          );
          setItinerary(newItin);
          try { sessionStorage.setItem(`itin:${sessionId}`, JSON.stringify(newItin)); } catch {}

          // Check if this was an active planning session before clearing states
          const wasActivePlanning = planningActiveRef.current;

          // NOW we clear all planning states since itinerary is complete
          planningActiveRef.current = false;
          setIsPlanning(false);
          setIsTyping(false);
          setDetectedIntent(null);
          planningReplyInjectedRef.current = false;

          if (planningTimeoutRef.current) {
            window.clearTimeout(planningTimeoutRef.current);
            planningTimeoutRef.current = null;
          }
          // Do not auto-inject any assistant confirmation message here.
          // The backend will send its own appropriate chat response.
        } else {
          // Never clear on null/invalid; preserve existing state
          console.warn("[client] Ignoring empty/invalid itinerary.update to preserve state");
        }
      } else if (evt.type === "search.results") {
        if (evt.data !== null && evt.data !== undefined) {
          setSearchResults(evt.data);
        }
      } else if (evt.type === "map.update") {
        // Never clear map on null/undefined; ignore to preserve existing view
        if (evt.data === null || evt.data === undefined) {
          console.log("[client] Ignoring empty map.update to preserve current map state");
          // no-op
        } else {
          setMapData(evt.data);
          try { sessionStorage.setItem(`map:${sessionId}`, JSON.stringify(evt.data)); } catch {}
        }
      } else if ((evt as any).type === "planning.status") {
        // Backend signals planning lifecycle; keep UI in sync and avoid stuck typing
        const status = (evt as any).data?.status?.toLowerCase?.() || "";
        console.log("[client] planning.status:", status);
        if (status === "started" || status === "running" || status === "thinking" || status === "planning") {
          planningActiveRef.current = true;
          setIsPlanning(true);
          setIsTyping(true);
          if (planningTimeoutRef.current) {
            window.clearTimeout(planningTimeoutRef.current);
            planningTimeoutRef.current = null;
          }
        } else if (status === "idle" || status === "complete" || status === "completed" || status === "error" || status === "stopped") {
          // If no itinerary is on the way, clear safely after a short delay
          if (planningTimeoutRef.current) {
            window.clearTimeout(planningTimeoutRef.current);
            planningTimeoutRef.current = null;
          }
          planningTimeoutRef.current = window.setTimeout(() => {
            planningActiveRef.current = false;
            setIsPlanning(false);
            setIsTyping(false);
            setDetectedIntent(null);
            planningTimeoutRef.current = null;
          }, 400);
        }
      } else if (evt.type === "intent.detected") {
        // Set detected intent when server classifies user message
        console.log(
          "[client] Intent detected:",
          evt.data.intent,
          "for message:",
          evt.data.message,
        );
        setDetectedIntent(evt.data.intent);
        // If planning intent is detected, immediately show planning animation
        if (evt.data.intent === "PLAN_TRIP") {
          console.log(
            "[client] PLAN_TRIP intent detected - starting planning animation",
          );
          planningActiveRef.current = true;
          planningReplyInjectedRef.current = false;
          setIsPlanning(true);
          setIsTyping(true);
          setDetectedIntent("PLAN_TRIP");

          // Clear any existing timeout
          if (planningTimeoutRef.current)
            window.clearTimeout(planningTimeoutRef.current);

          // Set timeout as safety fallback
          planningTimeoutRef.current = window.setTimeout(() => {
            console.log("[client] Planning safety timeout - clearing states");
            planningActiveRef.current = false;
            setIsPlanning(false);
            setIsTyping(false);
            setDetectedIntent(null);
          }, 60000); // Increased to 60 seconds for complex planning
        }
      } else if (evt.type === "planning.status") {
        // Handle planning status to show proper animation
        const status = evt.data?.status || "";
        console.log("[client] Planning status:", status);
        if (/(Creating|Analyzing|Finding|planning|itinerary)/i.test(status)) {
          console.log("[client] Planning status active:", status);
          if (!planningActiveRef.current) {
            planningActiveRef.current = true;
            setIsTyping(true);
            setIsPlanning(true);
            setDetectedIntent("PLAN_TRIP");
          }
          // Refresh timeout on each status update
          if (planningTimeoutRef.current)
            window.clearTimeout(planningTimeoutRef.current);
          planningTimeoutRef.current = window.setTimeout(() => {
            console.log("[client] Planning status timeout - clearing states");
            planningActiveRef.current = false;
            setIsTyping(false);
            setIsPlanning(false);
            setDetectedIntent(null);
          }, 60000);
        } else if (/(completed|finished|done)/i.test(status)) {
          console.log("[client] Planning status completed:", status);
          // Don't clear immediately - wait for itinerary.update event
          // This ensures smooth transition from planning to itinerary display
        }
      }
    };

    // Open connection and setup cleanup
    try {
      const ws = connectWs(sessionId, onEvent, {
        onOpen: () => {
          reconnectAttemptsRef.current = 0;
          connectingRef.current = false;
        },
        onClose: () => {
          hadDisconnectRef.current = true;
        },
        onError: () => {},
        onHealthCheck: (healthy) => {
          if (!healthy) {
            try {
              (wsRef.current as any)?.closeWithCleanup?.();
            } catch {}
            wsRef.current = null;
          }
        },
      });
      wsRef.current = ws;
    } catch (e) {
      console.error("[client] Failed to open WebSocket", e);
    }

    return () => {
      try {
        const w = wsRef.current as any;
        if (w?.closeWithCleanup) w.closeWithCleanup();
        else wsRef.current?.close();
      } catch {}
      wsRef.current = null;
    };
  }, [sessionId]);

  const handleToggleSave = async (poi: POI, nextSaved: boolean) => {
    if (!sessionId) return;
    try {
      await apiSavePoi(sessionId, poi.id, nextSaved);
      setSavedPoiIds((prev) => {
        const next = new Set(prev);
        if (nextSaved) next.add(poi.id);
        else next.delete(poi.id);
        return next;
      });
    } catch (e: any) {
      toast({
        title: "Failed to update saved",
        description: e?.message || "Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleAddPoi = async (poi: POI) => {
    // Just populate the input box, don't auto-send
    const loc = poi.address ? ` at ${poi.address}` : "";
    const coord = poi.lat && poi.lng ? ` (coords: ${poi.lat}, ${poi.lng})` : "";
    const msg = `Please add ${poi.name}${loc}${coord} to my itinerary in an appropriate slot. If needed, adjust nearby activities accordingly.`;
    setInputMessage(msg);
    setActiveLeftView("chat");
  };

  const handlePopulateInput = (text: string) => {
    setInputMessage(text);
    setActiveLeftView("chat");
  };

  const handleReplan = async (poi?: POI) => {
    const base = `Replan the itinerary optimizing for travel time and experience. Keep origin ${trip.origin || ""} and destination ${trip.destination || ""} for ${trip.days || "?"} days.`;
    const withPoi = poi
      ? ` Consider including ${poi.name} (${poi.address || ""}).`
      : "";
    await handleSendMessage(base + withPoi);
    setActiveLeftView("chat");
    setActiveRightView("itinerary");
  };

  // Send a chat message via HTTP (server will emit WS updates)
  const handleSendMessage = async (content?: string) => {
    const text = (content ?? inputMessage).trim();
    if (!text) return;
    if (!sessionId) {
      toast({
        title: "No session",
        description: "Please wait for session to initialize.",
        variant: "destructive",
      });
      return;
    }

    // Optimistically add user message
    const local: ChatMessage = {
      id: `local-${Date.now()}`,
      role: "user",
      content: text,
      createdAt: new Date().toISOString(),
      fromDashboard: false as any,
    };
    setMessages((prev) => [...prev, local]);
    lastUserSentRef.current = { content: text, at: Date.now() };
    setIsTyping(true);
    setDetectedIntent(null);
    planningActiveRef.current = false;
    setIsPlanning(false);
    setInputMessage("");

    try {
      await sendChat({ sessionId, inviteId, message: text });
    } catch (e: any) {
      setIsTyping(false);
      toast({
        title: "Failed to send",
        description: e?.message || "Please try again.",
        variant: "destructive",
      });
    }
  };

  // Save trip handler (frontend acknowledgement; backend generally auto-saves updates)
  const handleSaveTrip = () => {
    toast({
      title: "Trip saved",
      description: "Your changes are saved.",
      variant: "default",
    });
  };

  // Open delete confirmation dialog
  const handleDeleteTrip = () => {
    setShowDeleteDialog(true);
  };

  // Confirm delete and navigate away
  const confirmDeleteTrip = async () => {
    if (!sessionId) return;
    setIsDeleting(true);
    try {
      await apiDeleteTrip(sessionId);
      setShowDeleteDialog(false);
      toast({ title: "Trip deleted" });
      // Navigate to dashboard with refresh parameter
      try {
        await supabase.auth.getSession();
      } catch {}
      window.location.replace("/dashboard?refresh=true");
    } catch (e: any) {
      toast({
        title: "Failed to delete",
        description: e?.message || "Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsDeleting(false);
    }
  };

  if (!authChecked) {
    return (
      <div className="h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 via-white to-indigo-50">
        <div className="text-center">
          {/* Roameo Logo Animation */}
          <div className="mb-6 relative">
            <div className="w-16 h-16 bg-black rounded-full flex items-center justify-center mx-auto mb-3 relative overflow-hidden">
              <div className="w-6 h-6 bg-white rounded-full animate-pulse"></div>
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-30 animate-sweep"></div>
            </div>
            <div className="absolute inset-0 w-16 h-16 bg-black rounded-full mx-auto opacity-20 animate-ping"></div>
          </div>

          {/* Roameo Text */}
          <h2 className="text-2xl font-bold text-gray-900 mb-2">roameo</h2>
          <p className="text-gray-600 mb-4">
            Initializing your travel companion...
          </p>

          {/* Loading dots */}
          <div className="flex justify-center items-center space-x-1">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
            <div
              className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
              style={{ animationDelay: "0.1s" }}
            ></div>
            <div
              className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
              style={{ animationDelay: "0.2s" }}
            ></div>
          </div>
        </div>

        <style jsx>{`
          @keyframes sweep {
            0% {
              transform: translateX(-100%);
            }
            100% {
              transform: translateX(100%);
            }
          }
          .animate-sweep {
            animation: sweep 2s ease-in-out infinite;
          }
        `}</style>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      <TopNavigation
        trip={{
          id: trip.sessionId,
          title: trip.title || "Trip",
          origin: trip.origin || "",
          destination: trip.destination || "",
          duration: `${trip.days || 0} days`,
          travelers: `${trip.travelers || 1} travelers`,
          budget: trip.budget || "Budget",
        }}
        onReplan={() => handleReplan()}
        onPopulateInput={handlePopulateInput}
        inviteLink={
          inviteId
            ? `${typeof window !== "undefined" ? window.location.origin : ""}/chat?inviteId=${inviteId}`
            : undefined
        }
        showBottomBorder={true}
        onTripUpdate={async (t) => {
          const daysMatch = /\d+/.exec(t.duration || "");
          const travelersMatch = /\d+/.exec(t.travelers || "");
          const nextTrip = (prev: TripContext) => ({
            ...prev,
            title: t.title,
            origin: t.origin,
            destination: t.destination,
            days: daysMatch ? parseInt(daysMatch[0], 10) : prev.days,
            travelers: travelersMatch
              ? parseInt(travelersMatch[0], 10)
              : prev.travelers,
            budget: t.budget,
          });
          setTrip(nextTrip);
          if (sessionId) {
            const patch: Partial<TripContext> = {
              title: t.title,
              origin: t.origin,
              destination: t.destination,
              days: daysMatch ? parseInt(daysMatch[0], 10) : undefined,
              travelers: travelersMatch
                ? parseInt(travelersMatch[0], 10)
                : undefined,
              budget: t.budget,
            };
            try {
              await tripUpdate(sessionId, patch);
            } catch {}
          }
        }}
        onInvite={() => {}}
        isRightPanelVisible={isRightPanelVisible}
        onToggleRightPanel={() => setIsRightPanelVisible(!isRightPanelVisible)}
        onSaveTrip={handleSaveTrip}
        onDeleteTrip={handleDeleteTrip}
        isDeleting={isDeleting}
        onSignOut={async () => {
          try {
            await supabase.auth.signOut();
          } catch (e) {
            // ignore
          } finally {
            window.location.replace("/auth/login");
          }
        }}
      />

      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Chat/Search */}
        <div
          className={`${isRightPanelVisible ? "w-1/2" : "w-full"} relative h-full border-r border-gray-200 transition-all duration-500 ease-in-out`}
        >
          <div className="h-full flex flex-col">
            {isRightPanelVisible ? (
              <LeftPanelTabs
                activeView={activeLeftView}
                onViewChange={setActiveLeftView}
              />
            ) : (
              <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 bg-white/95 backdrop-blur-md rounded-full p-1 border border-white/30 shadow-xl transition-all duration-300">
                <Button
                  variant={activeLeftView === "chat" ? "default" : "ghost"}
                  onClick={() => setActiveLeftView("chat")}
                  className={`rounded-full px-4 ${
                    activeLeftView === "chat"
                      ? "bg-black/80 text-white hover:bg-black/90 backdrop-blur-sm"
                      : "hover:bg-white/80 backdrop-blur-sm"
                  }`}
                >
                  <MessageCircle className="w-4 h-4 mr-1" />
                  Chat
                </Button>
                <Button
                  variant={activeLeftView === "search" ? "default" : "ghost"}
                  size="sm"
                  onClick={() => setActiveLeftView("search")}
                  className={`rounded-full px-4 ${
                    activeLeftView === "search"
                      ? "bg-black/80 text-white hover:bg-black/90 backdrop-blur-sm"
                      : "hover:bg-white/80 backdrop-blur-sm"
                  }`}
                >
                  <Search className="w-4 h-4 mr-1" />
                  Search
                </Button>
                <Button
                  variant={activeLeftView === "saved" ? "default" : "ghost"}
                  size="sm"
                  onClick={() => setActiveLeftView("saved")}
                  className={`rounded-full px-4 ${
                    activeLeftView === "saved"
                      ? "bg-black/80 text-white hover:bg-black/90 backdrop-blur-sm"
                      : "hover:bg-white/80 backdrop-blur-sm"
                  }`}
                >
                  <Bookmark className="w-4 h-4 mr-1" />
                  Saved
                </Button>
                <div className="h-4 w-px bg-gray-200 mx-1"></div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="rounded-full px-4 hover:bg-white/80 backdrop-blur-sm"
                  onClick={() => {
                    setActiveRightView("map");
                    setIsRightPanelVisible(true);
                  }}
                >
                  <MapIcon className="w-4 h-4 mr-1" />
                  Map
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="rounded-full px-4 hover:bg-white/80 backdrop-blur-sm"
                  onClick={() => {
                    setActiveRightView("itinerary");
                    setIsRightPanelVisible(true);
                  }}
                >
                  <Calendar className="w-4 h-4 mr-1" />
                  Itinerary
                </Button>
              </div>
            )}
            <div className="h-full flex flex-col overflow-y-auto">
              {activeLeftView === "chat" && (
                <ChatInterface
                  messages={messages}
                  onSendMessage={handleSendMessage}
                  activeView="chat"
                  onViewChange={setActiveLeftView}
                  isRightPanelVisible={isRightPanelVisible}
                  activeRightView={activeRightView}
                  setActiveRightView={setActiveRightView}
                  setIsRightPanelVisible={setIsRightPanelVisible}
                  pois={mapPois}
                  isTyping={isTyping}
                  detectedIntent={detectedIntent}
                  planningActive={isPlanning}
                  savedIds={savedPoiIds}
                  onToggleSave={handleToggleSave}
                  onAddPoi={handleAddPoi}
                  onReplan={handleReplan}
                  onPopulateInput={handlePopulateInput}
                  inputValue={inputMessage}
                  onInputChange={setInputMessage}
                />
              )}
              {activeLeftView === "search" && (
                <SearchInterface
                  activeView="search"
                  onViewChange={setActiveLeftView}
                  results={searchResults}
                  savedIds={savedPoiIds}
                  itineraryPoiIds={itineraryPoiIds}
                  onAddPoi={handleAddPoi}
                  onToggleSave={handleToggleSave}
                  onReplan={handleReplan}
                />
              )}
              {activeLeftView === "saved" && (
                <SearchInterface
                  activeView="saved"
                  onViewChange={setActiveLeftView}
                  results={savedResults}
                  savedIds={savedPoiIds}
                  itineraryPoiIds={itineraryPoiIds}
                  onAddPoi={handleAddPoi}
                  onToggleSave={handleToggleSave}
                  onReplan={handleReplan}
                />
              )}
            </div>
          </div>
        </div>

        {/* Right Panel - Map/Itinerary */}
        <div
          className={`${isRightPanelVisible ? "w-1/2" : "w-0"} h-full transition-all duration-500 ease-in-out overflow-hidden`}
        >
          <RightPanel
            activeView={activeRightView}
            onViewChange={setActiveRightView}
            onClose={() => setIsRightPanelVisible(false)}
            trip={{
              id: trip.sessionId,
              title: trip.title || "Trip",
              origin: trip.origin || "",
              destination: trip.destination || "",
              duration: `${trip.days || 0} days`,
              travelers: `${trip.travelers || 1} travelers`,
              budget: trip.budget || "Budget",
            }}
            itinerary={itinerary}
            mapData={mapData}
            savedIds={savedPoiIds}
            itineraryPoiIds={itineraryPoiIds}
            onToggleSave={handleToggleSave}
            onAddPoi={handleAddPoi}
            onReplan={handleReplan}
          />
        </div>
      </div>

      {/* Delete Trip Confirmation Dialog */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent className="sm:max-w-[400px] p-0 bg-white rounded-2xl border-0 shadow-2xl">
          {/* Close Button */}
          <button
            onClick={() => setShowDeleteDialog(false)}
            className="absolute top-4 right-4 w-8 h-8 flex items-center justify-center text-gray-400 hover:text-gray-600 transition-colors z-10"
            disabled={isDeleting}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>

          <div className="p-8 text-center">
            {/* Warning Icon */}
            <div className="mx-auto mb-6 w-20 h-20 rounded-full bg-gradient-to-br from-yellow-400 to-orange-500 flex items-center justify-center relative overflow-hidden">
              {/* Diagonal stripes pattern */}
              <div className="absolute inset-0 bg-black opacity-20">
                <div
                  className="absolute inset-0"
                  style={{
                    backgroundImage:
                      "repeating-linear-gradient(45deg, transparent, transparent 3px, black 3px, black 6px)",
                    opacity: 0.8,
                  }}
                ></div>
              </div>
            </div>

            {/* Title */}
            <h2 className="text-xl font-bold text-gray-900 mb-3">
              Are you sure you want to delete?
            </h2>

            {/* Description */}
            <p className="text-gray-600 text-sm mb-6 leading-relaxed">
              Click on Agree if you like to delete this trip permanently. If not
              click on cancel!
            </p>

            {/* Warning Banner */}
            <div className="bg-orange-50 border border-orange-200 rounded-lg p-3 mb-6 flex items-center gap-2">
              <div className="w-5 h-5 rounded-full bg-orange-400 flex items-center justify-center flex-shrink-0">
                <span className="text-white text-xs font-bold">!</span>
              </div>
              <span className="text-orange-800 text-sm font-medium">
                You can't revert back if once deleted!
              </span>
            </div>

            {/* Action Buttons */}
            <div className="space-y-3">
              <button
                onClick={confirmDeleteTrip}
                disabled={isDeleting}
                className="w-full bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white font-semibold py-3 px-6 rounded-full transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isDeleting ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2 inline-block" />
                    Deleting...
                  </>
                ) : (
                  "Agree"
                )}
              </button>

              <button
                onClick={() => setShowDeleteDialog(false)}
                disabled={isDeleting}
                className="w-full bg-transparent border border-orange-300 text-orange-600 hover:bg-orange-50 font-semibold py-3 px-6 rounded-full transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Cancel
              </button>
            </div>
          </div>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
