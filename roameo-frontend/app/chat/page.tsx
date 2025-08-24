"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { useSearchParams, useRouter } from "next/navigation"
import { supabase } from "@/lib/supabase/client"
import { TopNavigation } from "@/components/top-navigation"
import { ChatInterface } from "@/components/chat-interface"
import { SearchInterface } from "@/components/search-interface"
import { RightPanel } from "@/components/right-panel"
import { LeftPanelTabs } from "@/components/left-panel-tabs"
import { Button } from "@/components/ui/button"
import { MessageCircle, Search, Bookmark, Map as MapIcon, Calendar } from "lucide-react"
import { connectWs } from "@/lib/ws"
import { sendChat, tripUpdate, createInvite, deleteTrip as apiDeleteTrip, savePoi as apiSavePoi, getSavedPoiIds } from "@/lib/api"
import { toast } from "@/hooks/use-toast"
import type { ChatMessage, Itinerary, SearchResults, TripContext, WsEvent, POI } from "@/lib/types"

export default function ChatPage() {
  const [activeLeftView, setActiveLeftView] = useState<"chat" | "search" | "saved">("chat")
  const [activeRightView, setActiveRightView] = useState<"map" | "itinerary">("map")
  const [isRightPanelVisible, setIsRightPanelVisible] = useState(true)

  const searchParams = useSearchParams()
  const router = useRouter()

  const [sessionId, setSessionId] = useState<string | undefined>(undefined)
  const [inviteId, setInviteId] = useState<string | undefined>(undefined)
  const wsRef = useRef<WebSocket | null>(null)

  const [trip, setTrip] = useState<TripContext>({
    sessionId: "temp",
    title: undefined,
    origin: undefined,
    destination: undefined,
    days: undefined,
    travelers: undefined,
    budget: undefined,
  })

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const seenMessageIdsRef = useRef<Set<string>>(new Set())
  const [itinerary, setItinerary] = useState<Itinerary | undefined>(undefined)
  const [searchResults, setSearchResults] = useState<SearchResults | undefined>(undefined)
  const [mapData, setMapData] = useState<any>(undefined)
  const [isDeleting, setIsDeleting] = useState(false)
  const [isTyping, setIsTyping] = useState(false)
  const [savedPoiIds, setSavedPoiIds] = useState<Set<string>>(new Set())
  const reconnectAttemptsRef = useRef(0)
  const reconnectTimerRef = useRef<number | null>(null)
  const [authChecked, setAuthChecked] = useState(false)
  const initialMessageHandledRef = useRef(false)
  const lastUserSentRef = useRef<{ content: string; at: number } | null>(null)
  const hadDisconnectRef = useRef(false)
  const manualCloseRef = useRef(false)
  const connectingRef = useRef(false)

  // Require login to access chat
  useEffect(() => {
    let mounted = true
    const check = async () => {
      const { data: { session } } = await supabase.auth.getSession()
      if (!session) {
        router.replace("/auth/login")
        return
      }
      if (mounted) setAuthChecked(true)
    }
    check()
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_evt, session) => {
      if (!session) router.replace("/auth/login")
    })
    return () => {
      mounted = false
      subscription.unsubscribe()
    }
  }, [router])

  // Filter POIs for map display - only show itinerary POIs
  const mapPois = useMemo(() => {
    if (!itinerary?.daysPlan?.length) return []
    const itineraryPoiIds = new Set()
    itinerary.daysPlan.forEach((day: any) => {
      day.activities?.forEach((activity: any) => {
        if (activity.poiId) {
          itineraryPoiIds.add(activity.poiId)
        }
      })
    })
    // Get all POIs from search results
    const allPois = [
      ...(searchResults?.stays || []),
      ...(searchResults?.restaurants || []),
      ...(searchResults?.attractions || [])
    ]
    return allPois.filter((poi: any) => itineraryPoiIds.has(poi.id)) || []
  }, [searchResults, itinerary])

  // Set of itinerary POI IDs for 'Added only' map filter
  const itineraryPoiIds = useMemo(() => {
    const ids = new Set<string>()
    if (itinerary?.daysPlan?.length) {
      itinerary.daysPlan.forEach((d) => {
        d.activities.forEach((a) => { if (a.poiId) ids.add(a.poiId) })
        if (d.accommodation?.poiId) ids.add(d.accommodation.poiId)
      })
    }
    return ids
  }, [itinerary])

  // Saved-only results for unified Saved UI using SearchInterface
  const savedResults: SearchResults | undefined = useMemo(() => {
    if (!savedPoiIds || savedPoiIds.size === 0) return { stays: [] as POI[], restaurants: [] as POI[], attractions: [] as POI[] }
    const all: POI[] = [
      ...(searchResults?.stays || []),
      ...(searchResults?.restaurants || []),
      ...(searchResults?.attractions || []),
      ...(mapData?.pois || []),
    ]
    if (all.length === 0) return { stays: [] as POI[], restaurants: [] as POI[], attractions: [] as POI[] }
    const uniq = new Map() as Map<string, POI>
    all.forEach((p) => {
      if (savedPoiIds.has(p.id) && !uniq.has(p.id)) uniq.set(p.id, p)
    })
    const arr = Array.from(uniq.values())
    return {
      stays: arr.filter((p) => p.type === "stay"),
      restaurants: arr.filter((p) => p.type === "restaurant"),
      attractions: arr.filter((p) => p.type === "attraction"),
    }
  }, [savedPoiIds, searchResults, mapData])

  // Restore sessionId from URL if present
  useEffect(() => {
    const fromQuery = searchParams.get("sessionId") || undefined
    if (fromQuery && fromQuery !== sessionId) setSessionId(fromQuery)
  }, [searchParams, sessionId])

  // Open WS when we have a sessionId
  useEffect(() => {
    if (!sessionId) return
    if (wsRef.current) wsRef.current.close()

    const onEvent = (evt: WsEvent) => {
      if (evt.type === "session.ready") {
        setSessionId(evt.data.sessionId)
        setInviteId(evt.data.inviteId)
        // push sessionId to URL if missing
        const qp = new URLSearchParams(window.location.search)
        if (!qp.get("sessionId")) {
          qp.set("sessionId", evt.data.sessionId)
          router.replace(`?${qp.toString()}`)
        }
        // hydrate saved POIs for Saved tab
        ;(async () => {
          try {
            const res = await getSavedPoiIds(evt.data.sessionId)
            setSavedPoiIds(new Set(res.ids))
          } catch {}
        })()
        // nudge map to render immediately on first connection
        setTimeout(() => {
          try { window.dispatchEvent(new Event("resize")) } catch {}
        }, 100)

      } else if (evt.type === "chat.history") {
        setMessages(evt.data)
        evt.data.forEach((m: ChatMessage) => {
          if (m.id) seenMessageIdsRef.current.add(m.id)
        })
      } else if (evt.type === "chat.append") {
        // Deduplicate messages by id to avoid duplicates on reconnect/replay
        setMessages((m) => {
          const id = evt.data?.id as string | undefined
          if (id) {
            if (seenMessageIdsRef.current.has(id)) return m
            seenMessageIdsRef.current.add(id)
          }
          // Heuristic: suppress immediate server echo of the same user message we just appended locally
          if (evt.data.role === "user" && lastUserSentRef.current) {
            const withinWindow = Date.now() - lastUserSentRef.current.at < 4000
            if (withinWindow && evt.data.content.trim() === lastUserSentRef.current.content.trim()) {
              return m
            }
          }
          return [...m, evt.data]
        })
        if (evt.data.role === "assistant") setIsTyping(false)
      } else if (evt.type === "navbar.update") {
        setTrip((t) => ({ ...t, ...evt.data }))
      } else if (evt.type === "itinerary.update") {
        // Only clear itinerary if explicitly empty, not on null/undefined
        if (evt.data !== null && evt.data !== undefined) {
          setItinerary(evt.data)
        }
      } else if (evt.type === "search.results") {
        if (evt.data !== null && evt.data !== undefined) {
          setSearchResults(evt.data)
        }
      } else if (evt.type === "map.update") {
        if (evt.data !== null && evt.data !== undefined) {
          setMapData(evt.data)
        }
      }
    }

    const connectWithCallbacks = () => {
      if (connectingRef.current) return wsRef.current as WebSocket
      connectingRef.current = true
      const ws = connectWs(sessionId, onEvent, {
        onOpen: () => {
          if (hadDisconnectRef.current) toast({ title: "Reconnected" })
          hadDisconnectRef.current = false
          reconnectAttemptsRef.current = 0
          connectingRef.current = false
        },
        onClose: (ev) => {
          // schedule reconnect
          if (manualCloseRef.current) {
            // intentional close (e.g., dep change/unmount)
            manualCloseRef.current = false
            connectingRef.current = false
            return
          }
          // If the backend rejected the session (e.g., restarted and lost memory), clear session
          if (ev?.code === 1008) {
            toast({ title: "Session expired", description: "Starting a new trip." })
            setMessages([])
            setItinerary(undefined)
            setSearchResults(undefined)
            setMapData(undefined)
            setInviteId(undefined)
            setSessionId(undefined)
            const qp = new URLSearchParams(window.location.search)
            qp.delete("sessionId")
            router.replace(qp.toString() ? `?${qp.toString()}` : "?")
            connectingRef.current = false
            return
          }
          hadDisconnectRef.current = true
          const attempt = reconnectAttemptsRef.current + 1
          reconnectAttemptsRef.current = attempt
          const delay = Math.min(1000 * Math.pow(2, attempt), 10000)
          // Avoid spamming toasts: only show on first disconnect in a cycle
          if (attempt === 1) {
            toast({ title: "Connection lost", description: `Reconnecting in ${Math.round(delay / 1000)}s...` })
          }
          if (reconnectTimerRef.current) window.clearTimeout(reconnectTimerRef.current)
          reconnectTimerRef.current = window.setTimeout(() => {
            if (sessionId) {
              wsRef.current = connectWithCallbacks()
            }
          }, delay)
        },
        onError: () => {
          // surface lightweight error toast once per disconnect cycle
        },
      })
      wsRef.current = ws
      return ws
    }

    const w = connectWithCallbacks()
    return () => {
      if (reconnectTimerRef.current) window.clearTimeout(reconnectTimerRef.current)
      if (w && (w.readyState === WebSocket.CONNECTING || w.readyState === WebSocket.OPEN)) {
        manualCloseRef.current = true
        w.close()
      }
    }
  }, [sessionId, router])

  // Nudge Google Map to render when panel becomes visible or when switching to Map view
  useEffect(() => {
    if (typeof window === "undefined") return
    if (!isRightPanelVisible) return
    if (activeRightView !== "map") return
    const timer = window.setTimeout(() => {
      window.dispatchEvent(new Event("resize"))
    }, 150)
    return () => window.clearTimeout(timer)
  }, [isRightPanelVisible, activeRightView])

  // Set tab title to chat session/trip title
  useEffect(() => {
    if (typeof document === "undefined") return
    const base = "Roameo"
    const sessionSuffix = sessionId ? ` – ${sessionId.slice(-6)}` : ""
    const title = trip.title ? `Chat – ${trip.title}` : `Chat${sessionSuffix}`
    document.title = `${base} | ${title}`
  }, [trip.title, sessionId])

  // On first load: capture inviteId and initial message from query
  useEffect(() => {
    const initialInviteId = searchParams.get("inviteId") || undefined
    if (initialInviteId && initialInviteId !== inviteId) setInviteId(initialInviteId)

    const initialMessage = searchParams.get("message")
    if (authChecked && initialMessage && initialMessage.trim() && !initialMessageHandledRef.current) {
      initialMessageHandledRef.current = true
      // send once and then remove from URL to avoid duplicates
      handleSendMessage(initialMessage.trim())
      const qp = new URLSearchParams(window.location.search)
      qp.delete("message")
      const next = qp.toString()
      router.replace(next ? `?${next}` : "?")
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authChecked])

  const handleSaveTrip = () => {
    console.log("[backend] Saving trip:", trip)
  }

  const handleDeleteTrip = async () => {
    if (!sessionId) return
    if (isDeleting) return
    
    if (typeof window !== "undefined") {
      const ok = window.confirm("Delete this trip and all its data? This cannot be undone.")
      if (!ok) return
    }
    
    setIsDeleting(true)
    
    try {
      // Close WebSocket connection before deletion to prevent reconnection issues
      if (wsRef.current) {
        manualCloseRef.current = true
        wsRef.current.close()
        wsRef.current = null
      }
      
      // Delete trip via API
      await apiDeleteTrip(sessionId)
      
      // Reset local state immediately after successful deletion
      setMessages([])
      setItinerary(undefined)
      setSearchResults(undefined)
      setMapData(undefined)
      setInviteId(undefined)
      setSessionId(undefined)
      
      // Show success message
      toast({ title: "Trip deleted successfully" })
      
      // Navigate to dashboard immediately
      router.push("/dashboard")
      
    } catch (e: any) {
      console.error("Failed to delete trip:", e)
      toast({ 
        title: "Failed to delete trip", 
        description: e?.message || "Please try again.", 
        variant: "destructive" 
      })
    } finally {
      setIsDeleting(false)
    }
  }

  const handleSendMessage = async (message: string) => {
    // append locally for immediate UX
    const localId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    setMessages((m) => {
      seenMessageIdsRef.current.add(localId)
      return [
        ...m,
        { id: localId, role: "user", content: message, createdAt: new Date().toISOString() },
      ]
    })
    lastUserSentRef.current = { content: message, at: Date.now() }
    setIsTyping(true)
    const res = await sendChat({ sessionId, inviteId, message })
    if (!sessionId && res.sessionId) {
      setSessionId(res.sessionId)
      const qp = new URLSearchParams(window.location.search)
      qp.set("sessionId", res.sessionId)
      router.replace(`?${qp.toString()}`)
    }
    // Apply any events returned by HTTP immediately (mirror WS handler)
    if (res.events && Array.isArray(res.events)) {
      res.events.forEach((evt) => {
        if (evt.type === "chat.append") {
          setMessages((m) => {
            const id = (evt as any).data?.id as string | undefined
            if (id) {
              if (seenMessageIdsRef.current.has(id)) return m
              seenMessageIdsRef.current.add(id)
            }
            // Avoid echo of our just-sent user message
            if ((evt as any).data?.role === "user" && lastUserSentRef.current) {
              const withinWindow = Date.now() - lastUserSentRef.current.at < 4000
              if (withinWindow && (evt as any).data?.content?.trim() === lastUserSentRef.current.content.trim()) {
                return m
              }
            }
            return [...m, (evt as any).data]
          })
          if ((evt as any).data?.role === "assistant") setIsTyping(false)
        } else if (evt.type === "navbar.update") {
          setTrip((t) => ({ ...t, ...(evt as any).data }))
        } else if (evt.type === "itinerary.update") {
          const data = (evt as any).data
          if (data !== null && data !== undefined) {
            setItinerary(data)
          }
        } else if (evt.type === "search.results") {
          const data = (evt as any).data
          if (data !== null && data !== undefined) {
            setSearchResults(data)
          }
        } else if (evt.type === "map.update") {
          const data = (evt as any).data
          if (data !== null && data !== undefined) {
            setMapData(data)
          }

        }
      })
    }
  }

  const handleToggleSave = async (poi: POI, nextSaved: boolean) => {
    if (!sessionId) return
    try {
      await apiSavePoi(sessionId, poi.id, nextSaved)
      setSavedPoiIds((prev) => {
        const next = new Set(prev)
        if (nextSaved) next.add(poi.id)
        else next.delete(poi.id)
        return next
      })
    } catch (e: any) {
      toast({ title: "Failed to update saved", description: e?.message || "Please try again.", variant: "destructive" })
    }
  }

  const handleAddPoi = async (poi: POI) => {
    // MVP: ask AI to add POI to itinerary
    const loc = poi.address ? ` at ${poi.address}` : ""
    const coord = poi.lat && poi.lng ? ` (coords: ${poi.lat}, ${poi.lng})` : ""
    const msg = `Please add ${poi.name}${loc}${coord} to my itinerary in an appropriate slot. If needed, adjust nearby activities accordingly.`
    await handleSendMessage(msg)
    setActiveLeftView("chat")
  }

  const handleReplan = async (poi?: POI) => {
    const base = `Replan the itinerary optimizing for travel time and experience. Keep origin ${trip.origin || ""} and destination ${trip.destination || ""} for ${trip.days || "?"} days.`
    const withPoi = poi ? ` Consider including ${poi.name} (${poi.address || ""}).` : ""
    await handleSendMessage(base + withPoi)
    setActiveLeftView("chat")
    setActiveRightView("itinerary")
  }

  if (!authChecked) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Preparing chat…</p>
        </div>
      </div>
    )
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
        inviteLink={inviteId ? `${typeof window !== "undefined" ? window.location.origin : ""}/chat?inviteId=${inviteId}` : undefined}
        onTripUpdate={async (t) => {
          const daysMatch = /\d+/.exec(t.duration || "")
          const travelersMatch = /\d+/.exec(t.travelers || "")
          const nextTrip = (prev: TripContext) => ({
            ...prev,
            title: t.title,
            origin: t.origin,
            destination: t.destination,
            days: daysMatch ? parseInt(daysMatch[0], 10) : prev.days,
            travelers: travelersMatch ? parseInt(travelersMatch[0], 10) : prev.travelers,
            budget: t.budget,
          })
          setTrip(nextTrip)
          if (sessionId) {
            const patch: Partial<TripContext> = {
              title: t.title,
              origin: t.origin,
              destination: t.destination,
              days: daysMatch ? parseInt(daysMatch[0], 10) : undefined,
              travelers: travelersMatch ? parseInt(travelersMatch[0], 10) : undefined,
              budget: t.budget,
            }
            try { await tripUpdate(sessionId, patch) } catch {}
          }
        }}
        onInvite={async () => {
          if (!sessionId) return
          try {
            const res = await createInvite(sessionId)
            setInviteId(res.inviteId)
          } catch {}
        }}
        isRightPanelVisible={isRightPanelVisible}
        onToggleRightPanel={() => setIsRightPanelVisible(!isRightPanelVisible)}
        onSaveTrip={handleSaveTrip}
        onDeleteTrip={handleDeleteTrip}
        isDeleting={isDeleting}
        onSignOut={async () => {
          await supabase.auth.signOut()
          router.replace("/auth/login")
        }}
      />

      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Chat/Search */}
        <div className={`${isRightPanelVisible ? "w-1/2" : "w-full"} relative h-full border-r border-gray-200 transition-all duration-500 ease-in-out`}>
          <div className="h-full flex flex-col">
            {isRightPanelVisible ? (
              <LeftPanelTabs activeView={activeLeftView} onViewChange={setActiveLeftView} />
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
                    setActiveRightView("map")
                    setIsRightPanelVisible(true)
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
                    setActiveRightView("itinerary")
                    setIsRightPanelVisible(true)
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
                  savedIds={savedPoiIds}
                  onToggleSave={handleToggleSave}
                  onAddPoi={handleAddPoi}
                  onReplan={handleReplan}
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
        <div className={`${isRightPanelVisible ? "w-1/2" : "w-0"} h-full transition-all duration-500 ease-in-out overflow-hidden`}>
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
    </div>
  )
}
