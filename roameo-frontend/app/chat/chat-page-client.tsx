"use client";

import { startTransition, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { Calendar, Map as MapIcon, MessageCircle, Search } from "lucide-react";
import type { AuthChangeEvent, Session } from "@supabase/supabase-js";
import { ChatInterface } from "@/components/chat-interface";
import { LeftPanelTabs } from "@/components/left-panel-tabs";
import { RightPanel } from "@/components/right-panel";
import { SearchInterface } from "@/components/search-interface";
import { TopNavigation } from "@/components/top-navigation";
import { Button } from "@/components/ui/button";
import { toast } from "@/hooks/use-toast";
import {
  createSession,
  deleteTrip,
  discoverPlaces,
  getSession,
  mutatePlan,
  savePoi,
  sendMessage,
  updateSession
} from "@/lib/api";
import {
  buildOverviewMutation,
  buildItinerary,
  buildItineraryPoiIds,
  buildMapData,
  buildSearchResults,
  buildTripContext
} from "@/lib/session-view";
import { getVisibleMessages, useSessionStore } from "@/lib/session-store";
import { redirectToLogin } from "@/lib/auth-redirect";
import { supabase } from "@/lib/supabase/client";
import type {
  POI,
  SessionPlanMutation,
  SessionPlanningState,
  TripContext
} from "@/lib/types";
import { connectWs } from "@/lib/ws";

export default function ChatPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const initialSessionId = searchParams.get("sessionId") || undefined;
  const queuedMessage = searchParams.get("message")?.trim() || "";

  const [activeLeftView, setActiveLeftView] = useState<"chat" | "search" | "saved">("chat");
  const [activeRightView, setActiveRightView] = useState<"map" | "itinerary">("map");
  const [isRightPanelVisible, setIsRightPanelVisible] = useState(true);
  const [authReady, setAuthReady] = useState(false);
  const [sessionId, setSessionId] = useState(initialSessionId);
  const [isDeleting, setIsDeleting] = useState(false);
  const [draftInput, setDraftInput] = useState("");
  const [optimisticTrip, setOptimisticTrip] = useState<TripContext | null>(null);
  const [searchStatus, setSearchStatus] = useState<string | null>(null);
  const handledQueryRef = useRef<string | null>(null);
  const searchParamsRef = useRef(searchParams);
  searchParamsRef.current = searchParams;
  const isSendingRef = useRef(false);

  const {
    snapshot,
    streamingMessage,
    isStreaming,
    error,
    reset,
    hydrate,
    applyEvent,
    setSavedPoiIds,
    activeTurnId
  } = useSessionStore();

  const sessionQuery = useQuery({
    queryKey: ["session", sessionId],
    queryFn: () => getSession(sessionId!),
    enabled: authReady && Boolean(sessionId),
    staleTime: 0
  });

  useEffect(() => {
    setSessionId(initialSessionId);
  }, [initialSessionId]);

  useEffect(() => {
    let mounted = true;
    const bootstrap = async () => {
      const {
        data: { session }
      } = await supabase.auth.getSession();

      if (!session) {
        redirectToLogin();
        return;
      }

      if (mounted) {
        setAuthReady(true);
      }
    };

    void bootstrap();
    const {
      data: { subscription }
    } = supabase.auth.onAuthStateChange(
      (_event: AuthChangeEvent, session: Session | null) => {
      if (!session) {
        redirectToLogin();
      }
      }
    );

    return () => {
      mounted = false;
      subscription.unsubscribe();
    };
  }, [router]);

  useEffect(() => {
    if (!sessionId) {
      reset();
      return;
    }

    if (snapshot?.id && snapshot.id !== sessionId) {
      reset();
    }
  }, [reset, sessionId, snapshot?.id]);

  useEffect(() => {
    if (sessionQuery.data) {
      hydrate(sessionQuery.data);
    }
  }, [hydrate, sessionQuery.data]);

  useEffect(() => {
    if (!authReady || !sessionId) {
      return;
    }

    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
    let connection: { close: () => void } | undefined;

    const openStream = () => {
      connection = connectWs(sessionId, applyEvent, {
        getAccessToken: async () => {
          const {
            data: { session }
          } = await supabase.auth.getSession();
          return session?.access_token || null;
        },
        onError: (streamError: Error) => {
          if (!cancelled) {
            toast({
              title: "Live sync paused",
              description: streamError.message,
              variant: "destructive"
            });
          }
        },
        onClose: () => {
          if (!cancelled) {
            reconnectTimer = setTimeout(openStream, 2000);
          }
        }
      });
    };

    openStream();

    return () => {
      cancelled = true;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
      connection?.close();
    };
  }, [applyEvent, authReady, sessionId]);

  useEffect(() => {
    if (!error) {
      return;
    }

    toast({
      title: "Trip update failed",
      description: error,
      variant: "destructive"
    });
  }, [error]);

  useEffect(() => {
    if (!authReady) {
      return;
    }

    if (!queuedMessage) {
      handledQueryRef.current = null;
      return;
    }

    if (handledQueryRef.current === queuedMessage) {
      return;
    }

    handledQueryRef.current = queuedMessage;

    const clearQueuedMessage = (nextSessionId?: string) => {
      const params = new URLSearchParams(searchParamsRef.current.toString());
      params.delete("message");
      if (nextSessionId) {
        params.set("sessionId", nextSessionId);
      }
      const href = params.toString() ? `/chat?${params.toString()}` : "/chat";
      startTransition(() => {
        router.replace(href);
      });
    };

    if (sessionId) {
      void sendMessage(sessionId, { content: queuedMessage }).then(() => {
        clearQueuedMessage(sessionId);
      });
      return;
    }

    void createSession({ initialMessage: queuedMessage }).then((session) => {
      setSessionId(session.id);
      hydrate(session);
      clearQueuedMessage(session.id);
    });
  }, [authReady, hydrate, queuedMessage, router, sessionId]);

  const visibleMessages = useMemo(
    () => getVisibleMessages(snapshot, streamingMessage),
    [snapshot, streamingMessage]
  );
  const trip = useMemo(() => buildTripContext(snapshot), [snapshot]);
  const displayedTrip = optimisticTrip || trip;
  const itinerary = useMemo(() => buildItinerary(snapshot), [snapshot]);
  const searchResults = useMemo(() => buildSearchResults(snapshot), [snapshot]);
  const mapData = useMemo(() => buildMapData(snapshot), [snapshot]);
  const itineraryPoiIds = useMemo(() => buildItineraryPoiIds(snapshot), [snapshot]);
  const savedIds = useMemo(
    () => new Set<string>(snapshot?.savedPoiIds ?? []),
    [snapshot?.savedPoiIds]
  );
  const planningState: SessionPlanningState | undefined = snapshot?.memory.planningState;
  const planningUnavailable = planningState?.status === "unavailable";
  const savedResults = useMemo(() => {
    const all = [
      ...searchResults.stays,
      ...searchResults.restaurants,
      ...searchResults.attractions
    ];
    const filtered = all.filter((poi) => savedIds.has(poi.id));
    return {
      stays: filtered.filter((poi) => poi.type === "stay"),
      restaurants: filtered.filter((poi) => poi.type === "restaurant"),
      attractions: filtered.filter((poi) => poi.type === "attraction")
    };
  }, [savedIds, searchResults]);

  const runSessionAction = async (content: string) => {
    if (!content.trim()) {
      return;
    }

    // Prevent double-fire from React StrictMode or rapid submissions
    if (isSendingRef.current) {
      return;
    }
    isSendingRef.current = true;

    try {
      if (snapshot?.id) {
        await sendMessage(snapshot.id, { content });
        return;
      }

      const session = await createSession({ initialMessage: content });
      setSessionId(session.id);
      hydrate(session);
      startTransition(() => {
        router.replace(`/chat?sessionId=${encodeURIComponent(session.id)}`);
      });
    } finally {
      isSendingRef.current = false;
    }
  };

  const handleSendMessage = async (content: string) => {
    try {
      await runSessionAction(content);
      setDraftInput("");
    } catch (sendError) {
      toast({
        title: "Message failed",
        description:
          sendError instanceof Error
            ? sendError.message
            : "Please try again.",
        variant: "destructive"
      });
    }
  };


  const applyStructuredMutation = async (mutation: SessionPlanMutation) => {
    if (!snapshot?.id) {
      return;
    }

    const updated = await mutatePlan(snapshot.id, mutation);
    hydrate(updated);
  };

  const handleToggleSave = async (poi: POI, nextSaved: boolean) => {
    if (!snapshot?.id) {
      return;
    }

    const nextIds = new Set(savedIds);
    if (nextSaved) {
      nextIds.add(poi.id);
    } else {
      nextIds.delete(poi.id);
    }
    setSavedPoiIds(Array.from(nextIds));

    try {
      const response = await savePoi(snapshot.id, poi.id, nextSaved);
      setSavedPoiIds(response.ids);
    } catch (saveError) {
      setSavedPoiIds(Array.from(savedIds));
      toast({
        title: "Could not update saved places",
        description:
          saveError instanceof Error ? saveError.message : "Please try again.",
        variant: "destructive"
      });
    }
  };

  const handleTripUpdate = async (nextTrip: TripContext) => {
    if (!snapshot?.id) {
      return;
    }

    setOptimisticTrip(nextTrip);
    try {
      const mutation = buildOverviewMutation(trip, nextTrip);
      if (mutation) {
        await applyStructuredMutation(mutation);
      } else if (nextTrip.title && nextTrip.title !== trip.title) {
        const updated = await updateSession(snapshot.id, { title: nextTrip.title });
        hydrate(updated);
      }
      setOptimisticTrip(null);
    } catch (updateError) {
      setOptimisticTrip(null);
      toast({
        title: "Could not update trip",
        description:
          updateError instanceof Error
            ? updateError.message
            : "Please try again.",
        variant: "destructive"
      });
    }
  };

  const handleSlotAction = async (action: { field: string; value: string | number }, prompt: string) => {
    const nextTrip = { ...displayedTrip, [action.field]: action.value };
    await handleTripUpdate(nextTrip);
    setDraftInput("");
    await runSessionAction(prompt);
  };

  const handleDelete = async () => {
    if (!snapshot?.id) {
      return;
    }

    setIsDeleting(true);
    try {
      await deleteTrip(snapshot.id);
      reset();
      router.replace("/dashboard");
    } catch (deleteError) {
      toast({
        title: "Could not delete trip",
        description:
          deleteError instanceof Error
            ? deleteError.message
            : "Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsDeleting(false);
    }
  };

  const handleInvite = async () => {
    if (!snapshot?.id || typeof window === "undefined") {
      return;
    }

    const shareUrl = `${window.location.origin}/chat?sessionId=${encodeURIComponent(snapshot.id)}`;
    await navigator.clipboard.writeText(shareUrl);
    toast({
      title: "Link copied",
      description: "Share this session URL to continue planning together."
    });
  };

  const handlePlanMutation = async (
    poi: POI | undefined,
    action: "add" | "replan"
  ) => {
    if (!snapshot?.id) {
      return;
    }
    if (planningUnavailable) {
      toast({
        title: "Planning is temporarily unavailable",
        description: "Roameo kept your last accepted plan visible. Retry once the AI provider recovers.",
        variant: "destructive"
      });
      return;
    }

    const mutation: SessionPlanMutation =
      action === "add" && poi
        ? {
            type: "add_poi",
            poiId: poi.id
          }
        : {
            type: "rebalance_trip",
            focusPoiId: poi?.id
          };

    try {
      await applyStructuredMutation(mutation);
    } catch (mutationError) {
      toast({
        title: "Could not update itinerary",
        description:
          mutationError instanceof Error
            ? mutationError.message
            : "Please try again.",
        variant: "destructive"
      });
    }
  };

  const handleDiscoverCategory = async (
    category: "stay" | "restaurant" | "attraction" | "all"
  ) => {
    if (!snapshot?.id) {
      return;
    }

    const destination = displayedTrip.destination?.trim();
    if (!destination) {
      return;
    }

    const label =
      category === "stay"
        ? "stays"
        : category === "restaurant"
          ? "restaurants"
          : category === "attraction"
            ? "attractions"
            : "places";

    setSearchStatus(`Finding ${label} in ${destination}...`);

    try {
      const updated = await discoverPlaces(snapshot.id, {
        destination,
        category
      });
      hydrate(updated);
    } catch (discoverError) {
      toast({
        title: `Could not load ${label}`,
        description:
          discoverError instanceof Error
            ? discoverError.message
            : "Please try again.",
        variant: "destructive"
      });
    } finally {
      setSearchStatus(null);
    }
  };

  if (!authReady) {
    return (
      <div className="flex h-[100dvh] items-center justify-center overflow-hidden bg-white relative">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-slate-50/50 rounded-full blur-3xl -z-10"></div>
        <div className="rounded-[32px] border border-slate-100/60 bg-white/60 backdrop-blur-2xl px-10 py-8 shadow-[0_20px_80px_rgba(15,23,42,0.06),_0_6px_20px_rgba(15,23,42,0.04)] text-center transition-all flex flex-col items-center">
          <p className="font-semibold text-[11px] uppercase tracking-[0.2em] text-slate-400 mb-3">
            Loading workspace
          </p>
          <h1 className="text-xl font-bold tracking-tight text-slate-900">
            Restoring your trip context
          </h1>
          <div className="flex justify-center items-center space-x-2 mt-6">
            <div className="w-2 h-2 bg-slate-300 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
            <div className="w-2 h-2 bg-slate-800 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative flex h-[100dvh] flex-col overflow-hidden bg-[white] text-[#1f1b16]">
      <TopNavigation
        trip={{
          id: displayedTrip.id,
          title: displayedTrip.title,
          origin: displayedTrip.origin,
          destination: displayedTrip.destination,
          destinations: displayedTrip.destinations,
          startDate: displayedTrip.startDate,
          endDate: displayedTrip.endDate,
          dateFlexibility: displayedTrip.dateFlexibility,
          duration: displayedTrip.days ? `${displayedTrip.days} days` : "",
          travelers: displayedTrip.travelers,
          budget: displayedTrip.budget
        }}
        onTripUpdate={(nextTrip) => {
          void handleTripUpdate({
            ...displayedTrip,
            ...nextTrip,
            days:
              Number.parseInt(nextTrip.duration || String(displayedTrip.days), 10) ||
              displayedTrip.days
          });
        }}
        isRightPanelVisible={isRightPanelVisible}
        onToggleRightPanel={() => setIsRightPanelVisible((value) => !value)}
        onSaveTrip={() =>
          toast({
            title: "Saved automatically",
            description: "Roameo keeps session state, itinerary, and map data in sync."
          })
        }
        onInvite={() => {
          void handleInvite();
        }}
        inviteLink={
          snapshot?.id && typeof window !== "undefined"
            ? `${window.location.origin}/chat?sessionId=${encodeURIComponent(snapshot.id)}`
            : undefined
        }
        onDeleteTrip={() => {
          void handleDelete();
        }}
        isDeleting={isDeleting}
        onReplan={() => {
          void handlePlanMutation(undefined, "replan");
        }}
        onPopulateInput={setDraftInput}
      />

      {planningUnavailable && (
        <div className="z-40 w-full border-b border-amber-200 bg-amber-50 px-4 py-2 text-center text-sm font-medium text-amber-900 shadow-sm">
          AI planning is temporarily unavailable. Roameo kept your last accepted trip visible until the provider recovers.
        </div>
      )}

      <div className="flex min-h-0 flex-1 overflow-hidden bg-transparent lg:flex-row">
        <section className="relative flex min-h-0 flex-1 flex-col overflow-hidden bg-transparent">
          <LeftPanelTabs
            activeView={activeLeftView}
            onViewChange={setActiveLeftView}
            activeRightView={activeRightView}
            onRightViewChange={(view) => {
              setActiveRightView(view);
              setIsRightPanelVisible(true);
            }}
            showRightTabs={!isRightPanelVisible}
            suppressRightActive={!isRightPanelVisible}
          />

          <div className="min-h-0 flex-1 overflow-hidden">
            {activeLeftView === "chat" && (
              <ChatInterface
                messages={visibleMessages}
                onSendMessage={(content) => {
                  void handleSendMessage(content);
                }}
                activeView={activeLeftView}
                onViewChange={setActiveLeftView}
                isRightPanelVisible={isRightPanelVisible}
                activeRightView={activeRightView}
                setActiveRightView={setActiveRightView}
                setIsRightPanelVisible={setIsRightPanelVisible}
                pois={mapData.pois}
                isTyping={isStreaming}
                detectedIntent={null}
                planningActive={planningState?.status === "running"}
                planningState={planningState}
                savedIds={savedIds}
                itineraryPoiIds={itineraryPoiIds}
                onToggleSave={(poi, nextSaved) => {
                  void handleToggleSave(poi, nextSaved);
                }}
                onAddPoi={(poi) => {
                  void handlePlanMutation(poi, "add");
                }}
                onReplan={(poi) => handlePlanMutation(poi, "replan")}
                onPopulateInput={(text) => setDraftInput(text)}
                onSlotAction={handleSlotAction}
                inputValue={draftInput}
                onInputChange={setDraftInput}
                traces={snapshot?.traces}
                activeTurnId={activeTurnId}
              />
            )}

            {activeLeftView !== "chat" && (
              <SearchInterface
                activeView={activeLeftView}
                onViewChange={setActiveLeftView}
                sessionId={snapshot?.id}
                destination={displayedTrip.destination}
                results={activeLeftView === "saved" ? savedResults : searchResults}
                planningState={planningState}
                savedIds={savedIds}
                itineraryPoiIds={itineraryPoiIds}
                onAddPoi={(poi) => {
                  void handlePlanMutation(poi, "add");
                }}
                onToggleSave={(poi, nextSaved) => {
                  void handleToggleSave(poi, nextSaved);
                }}
                onReplan={(poi) => {
                  void handlePlanMutation(poi, "replan");
                }}
                onDiscoverCategory={(category) => {
                  void handleDiscoverCategory(category);
                }}
                isLoading={sessionQuery.isLoading && Boolean(sessionId)}
                searchStatus={searchStatus ?? undefined}
                isSplitView={isRightPanelVisible}
              />
            )}
          </div>
        </section>

        <aside
          className={`min-h-0 w-full flex-col overflow-hidden bg-transparent transition-[width,opacity,transform] duration-300 ease-out ${
            isRightPanelVisible
              ? "flex lg:flex lg:w-1/2 lg:translate-x-0 lg:opacity-100 lg:rounded-l-[24px]"
              : "hidden lg:flex lg:w-0 lg:translate-x-4 lg:opacity-0 lg:pointer-events-none"
          }`}
        >
          <RightPanel
            activeView={activeRightView}
            onViewChange={setActiveRightView}
            trip={trip}
            itinerary={itinerary}
            mapData={mapData}
            onClose={() => setIsRightPanelVisible(false)}
            planningState={planningState}
            savedIds={savedIds}
            itineraryPoiIds={itineraryPoiIds}
            onToggleSave={(poi, nextSaved) => {
              void handleToggleSave(poi, nextSaved);
            }}
            onAddPoi={(poi) => {
              void handlePlanMutation(poi, "add");
            }}
            onReplan={(poi) => {
              void handlePlanMutation(poi, "replan");
            }}
          />
        </aside>
      </div>

    </div>
  );
}
