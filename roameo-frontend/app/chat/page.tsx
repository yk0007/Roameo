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
} from "@/components/ui/alert-dialog"
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

  // Enhanced message state management with better deduplication and immediate updates
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const seenMessageIdsRef = useRef<Set<string>>(new Set())
  const [itinerary, setItinerary] = useState<Itinerary | undefined>(undefined)
  const [searchResults, setSearchResults] = useState<SearchResults | undefined>(undefined)
  const [mapData, setMapData] = useState<any>(undefined)
  
  // Enhanced debugging for itinerary and map data
  useEffect(() => {
    console.log('[client] Itinerary state updated:', itinerary ? `${itinerary.daysPlan?.length || 0} days` : 'undefined')
    if (itinerary) {
      console.log('[client] Itinerary details:', JSON.stringify(itinerary, null, 2))
    }
  }, [itinerary])
  
  useEffect(() => {
    console.log('[client] Map data state updated:', mapData ? `${mapData.pois?.length || 0} POIs, ${mapData.routes?.length || 0} routes` : 'undefined')
    if (mapData) {
      console.log('[client] Map data details:', JSON.stringify(mapData, null, 2))
    }
  }, [mapData])
  
  useEffect(() => {
    console.log('[client] Search results state updated:', searchResults ? `${(searchResults.stays?.length || 0) + (searchResults.restaurants?.length || 0) + (searchResults.attractions?.length || 0)} total POIs` : 'undefined')
  }, [searchResults])
  const [isDeleting, setIsDeleting] = useState(false)
  const [isTyping, setIsTyping] = useState(false)
  const [detectedIntent, setDetectedIntent] = useState<"PLAN_TRIP" | "DESTINATION_SEARCH" | "CHAT" | null>(null)
  const planningTimeoutRef = useRef<number | null>(null)
  const planningActiveRef = useRef(false) // Guard to prevent duplicate planning animations
  const [savedPoiIds, setSavedPoiIds] = useState<Set<string>>(new Set())
  const [inputMessage, setInputMessage] = useState<string>("") // Add state for controlling input
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const reconnectAttemptsRef = useRef(0)
  const reconnectTimerRef = useRef<number | null>(null)
  const [authChecked, setAuthChecked] = useState(false)
  const initialMessageHandledRef = useRef(false)
  const lastUserSentRef = useRef<{ content: string; at: number } | null>(null)
  const hadDisconnectRef = useRef(false)
  const manualCloseRef = useRef(false)
  const connectingRef = useRef(false)
  
  // Enhanced debug logging for messages state changes
  useEffect(() => {
    console.log('[client] Messages state updated, total count:', messages.length)
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1]
      console.log('[client] Last message:', { id: lastMessage.id, role: lastMessage.role, contentLength: lastMessage.content?.length })
      
      // Force typing state clear if we have a recent assistant message
      if (lastMessage.role === "assistant" && isTyping) {
        console.log('[client] Force clearing typing state due to new assistant message')
        setIsTyping(false)
        setDetectedIntent(null)
        planningActiveRef.current = false
        if (planningTimeoutRef.current) {
          window.clearTimeout(planningTimeoutRef.current)
          planningTimeoutRef.current = null
        }
      }
    }
  }, [messages, isTyping])

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
        // Merge history with any messages already in state to avoid overwriting recent appends
        setMessages((prev) => {
          const byId = new Map<string, ChatMessage>()
          
          // Find any dashboard messages in the existing messages
          const dashboardMessages = prev.filter(m => m.fromDashboard === true);
          const dashboardContents = new Set(dashboardMessages.map(m => m.content.trim()));
          
          // Seed with existing messages first
          for (const m of prev) {
            if (m?.id) byId.set(m.id, m)
          }
          
          // Add any new history messages by id, but skip those matching dashboard messages
          for (const m of evt.data) {
            // Skip server messages that match our dashboard messages
            if (m.role === "user" && dashboardContents.has(m.content.trim())) {
              
              continue;
            }
            
            if (m?.id && !byId.has(m.id)) byId.set(m.id, m)
          }
          
          const merged = Array.from(byId.values())
          // Sort by createdAt to keep chronological order if available
          merged.sort((a, b) => (a.createdAt || '').localeCompare(b.createdAt || ''))
          // Track seen ids for future de-dup
          merged.forEach((m) => { if (m.id) seenMessageIdsRef.current.add(m.id) })
          return merged
        })
      } else if (evt.type === "chat.append") {
        console.log('[client] Received chat.append event:', evt.data)
        // Deduplicate messages by id to avoid duplicates on reconnect/replay
        setMessages((prevMessages) => {
          const id = evt.data?.id as string | undefined
          console.log('[client] Processing message with id:', id, 'role:', evt.data.role, 'content length:', evt.data?.content?.length)
          
          if (id) {
            if (seenMessageIdsRef.current.has(id)) {
              console.log('[client] Skipping duplicate message:', id)
              return prevMessages
            }
            seenMessageIdsRef.current.add(id)
          }
          
          // Check for any message with fromDashboard flag - these should never be duplicated
          const isDashboardMessage = prevMessages.some(m => 
            m.fromDashboard === true && 
            m.role === evt.data.role && 
            m.content.trim() === evt.data.content.trim()
          );
          
          if (isDashboardMessage) {
            console.log('[client] Skipping duplicate of dashboard message')
            return prevMessages
          }
          
          // Enhanced heuristic: suppress immediate server echo of the same user message we just appended locally
          if (evt.data.role === "user" && lastUserSentRef.current) {
            const withinWindow = Date.now() - lastUserSentRef.current.at < 6000 // Extended to 6 seconds
            const contentMatch = evt.data.content.trim() === lastUserSentRef.current.content.trim()
            
            // Also check if we already have this exact message content in recent messages
            const recentUserMessages = prevMessages.slice(-3).filter(m => m.role === "user")
            const alreadyHasContent = recentUserMessages.some(m => m.content.trim() === evt.data.content.trim())
            
            if ((withinWindow && contentMatch) || alreadyHasContent) {
              console.log('[client] Suppressing user message echo - withinWindow:', withinWindow, 'contentMatch:', contentMatch, 'alreadyHasContent:', alreadyHasContent)
              return prevMessages
            }
          }
          
          const newMessages = [...prevMessages, evt.data]
          console.log('[client] Adding message to state, new total:', newMessages.length)
          
          // For assistant messages, ensure immediate visibility by clearing any blocking states
          if (evt.data.role === "assistant") {
            console.log('[client] Assistant message added - scheduling immediate state clear')
            // Use setTimeout to ensure this happens after the state update
            setTimeout(() => {
              setIsTyping(false)
              setDetectedIntent(null)
              planningActiveRef.current = false
              if (planningTimeoutRef.current) {
                window.clearTimeout(planningTimeoutRef.current)
                planningTimeoutRef.current = null
              }
            }, 0)
          }
          
          return newMessages
        })
        
        if (evt.data.role === "assistant") {
          console.log('[client] Processing assistant message')
          
          // CRITICAL: Always clear all typing and planning states IMMEDIATELY
          // Use immediate state updates to ensure message visibility
          setIsTyping(false)
          setDetectedIntent(null)
          planningActiveRef.current = false
          
          // Clear timeout
          if (planningTimeoutRef.current) {
            window.clearTimeout(planningTimeoutRef.current)
            planningTimeoutRef.current = null
          }
          
          // Force multiple state clearing attempts to ensure UI updates
          setTimeout(() => {
            console.log('[client] First UI update after assistant message')
            setIsTyping(false)
            setDetectedIntent(null)
            planningActiveRef.current = false
          }, 0)
          
          setTimeout(() => {
            console.log('[client] Second UI update after assistant message')
            setIsTyping(false)
            setDetectedIntent(null)
            planningActiveRef.current = false
          }, 10)
          
          setTimeout(() => {
            console.log('[client] Final UI update after assistant message')
            setIsTyping(false)
            setDetectedIntent(null)
            planningActiveRef.current = false
          }, 50)
        }
      } else if (evt.type === "navbar.update") {
        setTrip((t) => ({ ...t, ...evt.data }))
      } else if (evt.type === "itinerary.update") {
        // When itinerary is received, it means planning is complete
        console.log('[client] Itinerary received - planning should be complete')
        const data = evt.data
        if (data !== null && data !== undefined) {
          if (data === null) {
            console.log('[client] Clearing itinerary as requested')
            setItinerary(undefined)
          } else if (data && typeof data === 'object' && data.daysPlan) {
            console.log(`[client] Setting itinerary with ${data.daysPlan.length} days`)
            // First set the itinerary immediately without delay
            setItinerary(data)
            // Clear planning animation when itinerary is successfully loaded
            // Clear planning animation immediately when itinerary is successfully loaded
              planningActiveRef.current = false
              setIsTyping(false)
              setDetectedIntent(null)
              if (planningTimeoutRef.current) {
                window.clearTimeout(planningTimeoutRef.current)
                planningTimeoutRef.current = null
              }
          } else {
            console.warn('[client] Received invalid itinerary data, ignoring:', data)
          }
        }
      } else if (evt.type === "search.results") {
        if (evt.data !== null && evt.data !== undefined) {
          setSearchResults(evt.data)
        }
      } else if (evt.type === "map.update") {
        if (evt.data !== null && evt.data !== undefined) {
          setMapData(evt.data)
        }
      } else if (evt.type === "intent.detected") {
        // Set detected intent when server classifies user message
        console.log('[client] Intent detected:', evt.data.intent, 'for message:', evt.data.message)
        setDetectedIntent(evt.data.intent)
        // If planning intent is detected, immediately show planning animation
        if (evt.data.intent === "PLAN_TRIP" && !planningActiveRef.current) {
          console.log('[client] Immediately showing planning animation for PLAN_TRIP intent')
          planningActiveRef.current = true
          setIsTyping(true)
          // Explicitly set the detected intent to ensure it's properly passed to the chat interface
          setDetectedIntent("PLAN_TRIP")
          // Clear any existing timeout
          if (planningTimeoutRef.current) {
            window.clearTimeout(planningTimeoutRef.current)
         }
          // Set a fallback timeout to clear planning status after 30 seconds
          planningTimeoutRef.current = window.setTimeout(() => {
            console.log('[client] Planning timeout - clearing status')
            planningActiveRef.current = false
            setIsTyping(false)
            setDetectedIntent(null)
          }, 30000)
        } else if (evt.data.intent === "PLAN_TRIP" && planningActiveRef.current) {
          console.log('[client] Planning already active, skipping duplicate animation')
        }
      } else if (evt.type === "planning.status") {
        // Handle planning status to show proper animation
        console.log('[client] Planning status:', evt.data.status)
        
        // Check if planning is starting or completing based on status message
        const status = evt.data.status || ''
        
        if (status.includes('Creating') || status.includes('Analyzing') || status.includes('Finding') || status.includes('planning') || status.includes('itinerary')) {
          // Only start planning animation if not already active
          if (!planningActiveRef.current) {
            console.log('[client] Starting planning animation from status event')
            planningActiveRef.current = true
            setIsTyping(true)
            setDetectedIntent("PLAN_TRIP")
            // Clear any existing timeout
            if (planningTimeoutRef.current) {
              window.clearTimeout(planningTimeoutRef.current)
            }
            // Set a fallback timeout to clear planning status after 30 seconds
            planningTimeoutRef.current = window.setTimeout(() => {
              console.log('[client] Planning timeout - clearing status')
              planningActiveRef.current = false
              setIsTyping(false)
              setDetectedIntent(null)
            }, 30000)
          } else {
            console.log('[client] Planning already active, ignoring duplicate status event:', status)
          }
        } else if (status.includes('completed') || status.includes('finished') || status.includes('done')) {
          // Planning is complete
          console.log('[client] Planning completed via status')
          planningActiveRef.current = false
          // Clear timeout
          if (planningTimeoutRef.current) {
            window.clearTimeout(planningTimeoutRef.current)
            planningTimeoutRef.current = null
          }
          setIsTyping(false)
          setDetectedIntent(null)
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
            toast({ 
              title: "Session expired", 
              description: "Starting a new trip.",
              variant: "warning" as any
            })
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
            toast({ 
              title: "Connection lost", 
              description: `Reconnecting in ${Math.round(delay / 1000)}s...`,
              variant: "warning" as any
            })
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
      if (planningTimeoutRef.current) window.clearTimeout(planningTimeoutRef.current)
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

  // Set tab title to chat session/trip title - only update when planning occurs
  useEffect(() => {
    if (typeof document === "undefined") return
    const base = "Roameo"
    const sessionSuffix = sessionId ? ` – ${sessionId.slice(-6)}` : ""
    
    // Only update title when we have meaningful trip data (destination or itinerary)
    const hasPlanning = trip.destination || itinerary
    const title = hasPlanning && trip.title ? `Chat – ${trip.title}` : `Chat${sessionSuffix}`
    document.title = `${base} | ${title}`
  }, [trip.destination, trip.title, itinerary, sessionId])

  // On first load: capture inviteId and check for initial message in sessionStorage
  useEffect(() => {
    const initialInviteId = searchParams.get("inviteId") || undefined
    if (initialInviteId && initialInviteId !== inviteId) setInviteId(initialInviteId)

    // Check for URL parameter for backward compatibility
    const initialMessageFromUrl = searchParams.get("message")
    if (initialMessageFromUrl) {
      // Clear URL parameter immediately
      const qp = new URLSearchParams(window.location.search)
      qp.delete("message")
      const next = qp.toString()
      router.replace(next ? `?${next}` : "?")
    }
    
    // Check for message in sessionStorage (new approach)
    const initialMessageFromStorage = typeof window !== 'undefined' ? sessionStorage.getItem('initialChatMessage') : null
    
    // Process message from either source, prioritizing sessionStorage
    const initialMessage = initialMessageFromStorage || initialMessageFromUrl
    
    // Only process if we have a message and haven't handled it yet
    if (authChecked && initialMessage && !initialMessageHandledRef.current) {
      // Mark as handled immediately to prevent duplicate processing
      initialMessageHandledRef.current = true
      
      // Clear from sessionStorage to prevent duplicate handling on refresh
      if (typeof window !== 'undefined') {
        sessionStorage.removeItem('initialChatMessage')
      }
      
      // Add a flag to track the first message specifically
      const isFirstMessageFromDashboard = true;
      
      // Use setTimeout to ensure this runs after any initial state setup
      setTimeout(() => {
        // Double-check for duplicates right before sending
        const isDuplicate = messages.some(m => 
          m.role === "user" && m.content.trim() === initialMessage.trim()
        )
        
        if (!isDuplicate) {
          console.log('[client] Sending initial message from dashboard:', initialMessage.trim())
          
          // Create a unique ID for this message to help with deduplication
          const dashboardMessageId = `dashboard-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
          
          // Add to seen message IDs before sending to prevent duplicates from server echo
          seenMessageIdsRef.current.add(dashboardMessageId);
          
          // Add message directly to state with the special ID
          setMessages((prev) => [
            ...prev,
            { 
              id: dashboardMessageId, 
              role: "user", 
              content: initialMessage.trim(), 
              createdAt: new Date().toISOString(),
              fromDashboard: true // Mark this message as coming from dashboard
            }
          ]);
          
          // Send to server but don't add to messages again
          sendChat({ sessionId, inviteId, message: initialMessage.trim() })
            .then(res => {
              if (!sessionId && res.sessionId) {
                setSessionId(res.sessionId)
                const qp = new URLSearchParams(window.location.search)
                qp.set("sessionId", res.sessionId)
                router.replace(`?${qp.toString()}`)
              }
            })
            .catch(err => 
            
          // Set typing indicator
          setIsTyping(true);
        } else {
          console.log('[client] Skipping duplicate initial message from dashboard')
        }
      }, 100);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authChecked])

  const handleSaveTrip = () => {
    console.log("[backend] Saving trip:", trip)
  }

  const handleDeleteTrip = () => {
    setShowDeleteDialog(true)
  }
  
  const confirmDeleteTrip = async () => {
    if (!sessionId) return
    if (isDeleting) return
    
    setIsDeleting(true)
    setShowDeleteDialog(false)
    
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
      toast({ 
        title: "Trip deleted successfully",
        variant: "success" as any
      })
      
      // Navigate to dashboard immediately - set flag to disable animations
      window.history.replaceState({ ...window.history.state, fromChat: true }, '')
      sessionStorage.setItem('fromChat', 'true')
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
    console.log('[client] Sending message:', message)
    console.log('[client] Current WebSocket state:', wsRef.current?.readyState)
    
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
    console.log('[client] Setting isTyping=true for user message')
    setIsTyping(true)
    
    try {
      const res = await sendChat({ sessionId, inviteId, message })
      if (!sessionId && res.sessionId) {
        setSessionId(res.sessionId)
        const qp = new URLSearchParams(window.location.search)
        qp.set("sessionId", res.sessionId)
        router.replace(`?${qp.toString()}`)
      }
    // Apply any events returned by HTTP immediately (mirror WS handler)
    // Only process HTTP events if WebSocket is not connected or ready to avoid race conditions
    if (res.events && Array.isArray(res.events)) {
      const wsConnected = wsRef.current?.readyState === WebSocket.OPEN
      console.log('[client] Processing HTTP events:', res.events.length, 'events, WebSocket connected:', wsConnected)
      
      res.events.forEach((evt, index) => {
        console.log(`[client] Processing HTTP event ${index + 1}:`, evt.type)
        // Process all events regardless of WebSocket state to ensure reliability
        // WebSocket deduplication will handle any duplicates by message ID
        
        if (evt.type === "chat.append") {
          console.log('[client] Received HTTP chat.append event:', (evt as any).data)
          setMessages((prevMessages) => {
            const id = (evt as any).data?.id as string | undefined
            console.log('[client] Processing HTTP message with id:', id, 'role:', (evt as any).data?.role, 'content length:', (evt as any).data?.content?.length)
            
            if (id) {
              if (seenMessageIdsRef.current.has(id)) {
                console.log('[client] Skipping duplicate HTTP message:', id)
                return prevMessages
              }
              seenMessageIdsRef.current.add(id)
            }
            
            // Enhanced duplicate suppression for HTTP user messages
            if ((evt as any).data?.role === "user" && lastUserSentRef.current) {
              const withinWindow = Date.now() - lastUserSentRef.current.at < 6000 // Extended to 6 seconds
              const contentMatch = (evt as any).data?.content?.trim() === lastUserSentRef.current.content.trim()
              
              // Also check if we already have this exact message content in recent messages
              const recentUserMessages = prevMessages.slice(-3).filter(m => m.role === "user")
              const alreadyHasContent = recentUserMessages.some(m => m.content.trim() === (evt as any).data?.content?.trim())
              
              if ((withinWindow && contentMatch) || alreadyHasContent) {
                console.log('[client] Suppressing HTTP user message echo - withinWindow:', withinWindow, 'contentMatch:', contentMatch, 'alreadyHasContent:', alreadyHasContent)
                return prevMessages
              }
            }
            
            const newMessages = [...prevMessages, (evt as any).data]
            console.log('[client] Adding HTTP message to state, new total:', newMessages.length)
            console.log('[client] HTTP message content preview:', (evt as any).data?.content?.substring(0, 100) + '...')
            
            // Force immediate display for assistant messages
            if ((evt as any).data?.role === "assistant") {
              console.log('[client] HTTP Assistant message received - forcing immediate display')
              console.log('[client] Full assistant message content:', (evt as any).data?.content)
              
              // Schedule immediate state clear for HTTP assistant messages too
              setTimeout(() => {
                setIsTyping(false)
                setDetectedIntent(null)
                planningActiveRef.current = false
                if (planningTimeoutRef.current) {
                  window.clearTimeout(planningTimeoutRef.current)
                  planningTimeoutRef.current = null
                }
              }, 0)
            }
            
            return newMessages
          })
          
          if ((evt as any).data?.role === "assistant") {
            console.log('[client] Processing HTTP assistant message')
            
            // CRITICAL: Always clear all typing and planning states IMMEDIATELY
            // Use immediate state updates to ensure message visibility
            setIsTyping(false)
            setDetectedIntent(null)
            planningActiveRef.current = false
            
            // Clear timeout
            if (planningTimeoutRef.current) {
              window.clearTimeout(planningTimeoutRef.current)
              planningTimeoutRef.current = null
            }
            
            // Force multiple state clearing attempts for HTTP responses
            setTimeout(() => {
              console.log('[client] First HTTP UI update after assistant message')
              setIsTyping(false)
              setDetectedIntent(null)
              planningActiveRef.current = false
            }, 0)
            
            setTimeout(() => {
              console.log('[client] Second HTTP UI update after assistant message')
              setIsTyping(false)
              setDetectedIntent(null)
              planningActiveRef.current = false
            }, 10)
            
            setTimeout(() => {
              console.log('[client] Final HTTP UI update after assistant message')
              setIsTyping(false)
              setDetectedIntent(null)
              planningActiveRef.current = false
            }, 50)
          }
        } else if (evt.type === "navbar.update") {
          setTrip((t) => ({ ...t, ...(evt as any).data }))
        } else if (evt.type === "itinerary.update") {
          const data = (evt as any).data
          console.log('[client] Received HTTP itinerary update:', data)
          // Only update if we have valid itinerary data or explicit null to clear
          if (data !== undefined) {
            if (data === null) {
              console.log('[client] Clearing HTTP itinerary as requested')
              setItinerary(undefined)
            } else if (data && typeof data === 'object' && data.daysPlan) {
              console.log(`[client] Setting HTTP itinerary with ${data.daysPlan.length} days`)
              setItinerary(data)
              // Clear planning animation immediately when itinerary is received
            planningActiveRef.current = false
            setIsTyping(false)
            setDetectedIntent(null)
            if (planningTimeoutRef.current) {
              window.clearTimeout(planningTimeoutRef.current)
              planningTimeoutRef.current = null
            }
            } else {
              console.warn('[client] Received invalid HTTP itinerary data, ignoring:', data)
            }
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
        } else if (evt.type === "intent.detected") {
          // Set detected intent when server classifies user message via HTTP
          console.log('[client] Intent detected via HTTP:', (evt as any).data.intent, 'for message:', (evt as any).data.message)
          setDetectedIntent((evt as any).data.intent)
          // If planning intent is detected, immediately show planning animation
          if ((evt as any).data.intent === "PLAN_TRIP" && !planningActiveRef.current) {
            console.log('[client] Immediately showing planning animation for PLAN_TRIP intent via HTTP')
            planningActiveRef.current = true
            setIsTyping(true)
            // Clear any existing timeout
            if (planningTimeoutRef.current) {
              window.clearTimeout(planningTimeoutRef.current)
            }
            // Set a fallback timeout to clear planning status after 30 seconds
            planningTimeoutRef.current = window.setTimeout(() => {
              console.log('[client] Planning timeout - clearing status')
              planningActiveRef.current = false
              setIsTyping(false)
              setDetectedIntent(null)
            }, 30000)
          } else if ((evt as any).data.intent === "PLAN_TRIP" && planningActiveRef.current) {
            console.log('[client] Planning already active via HTTP, skipping duplicate animation')
          }
        }
      })
    }
    } catch (error) {
      console.error('[client] Error sending message:', error)
      // If HTTP fails, ensure we still clear the typing indicator
      setIsTyping(false)
      planningActiveRef.current = false
      setDetectedIntent(null)
      if (planningTimeoutRef.current) {
        window.clearTimeout(planningTimeoutRef.current)
        planningTimeoutRef.current = null
      }
      
      toast({
        title: "Failed to send message",
        description: "Please check your connection and try again.",
        variant: "destructive"
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
      toast({ 
        title: "Failed to update saved", 
        description: e?.message || "Please try again.", 
        variant: "destructive"
      })
    }
  }


  const handleAddPoi = async (poi: POI) => {
    // Just populate the input box, don't auto-send
    const loc = poi.address ? ` at ${poi.address}` : ""
    const coord = poi.lat && poi.lng ? ` (coords: ${poi.lat}, ${poi.lng})` : ""
    const msg = `Please add ${poi.name}${loc}${coord} to my itinerary in an appropriate slot. If needed, adjust nearby activities accordingly.`
    setInputMessage(msg)
    setActiveLeftView("chat")
  }

  const handlePopulateInput = (text: string) => {
    setInputMessage(text)
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
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            roameo
          </h2>
          <p className="text-gray-600 mb-4">Initializing your travel companion...</p>
          
          {/* Loading dots */}
          <div className="flex justify-center items-center space-x-1">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
          </div>
        </div>
        
        <style jsx>{`
          @keyframes sweep {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
          }
          .animate-sweep {
            animation: sweep 2s ease-in-out infinite;
          }
        `}</style>
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
        onPopulateInput={handlePopulateInput}
        inviteLink={inviteId ? `${typeof window !== "undefined" ? window.location.origin : ""}/chat?inviteId=${inviteId}` : undefined}
        showBottomBorder={true}
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
        onInvite={undefined}
        isRightPanelVisible={isRightPanelVisible}
        onToggleRightPanel={() => setIsRightPanelVisible(!isRightPanelVisible)}
        onSaveTrip={handleSaveTrip}
        onDeleteTrip={handleDeleteTrip}
        isDeleting={isDeleting}
        onSignOut={async () => {
          router.push("/auth/login")
          await supabase.auth.signOut()
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
                  detectedIntent={detectedIntent}
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
      
      {/* Delete Trip Confirmation Dialog */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent className="sm:max-w-[400px] p-0 bg-white rounded-2xl border-0 shadow-2xl">
          {/* Close Button */}
          <button
            onClick={() => setShowDeleteDialog(false)}
            className="absolute top-4 right-4 w-8 h-8 flex items-center justify-center text-gray-400 hover:text-gray-600 transition-colors z-10"
            disabled={isDeleting}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
          
          <div className="p-8 text-center">
            {/* Warning Icon */}
            <div className="mx-auto mb-6 w-20 h-20 rounded-full bg-gradient-to-br from-yellow-400 to-orange-500 flex items-center justify-center relative overflow-hidden">
              {/* Diagonal stripes pattern */}
              <div className="absolute inset-0 bg-black opacity-20">
                <div className="absolute inset-0" style={{
                  backgroundImage: 'repeating-linear-gradient(45deg, transparent, transparent 3px, black 3px, black 6px)',
                  opacity: 0.8
                }}></div>
              </div>
            </div>
            
            {/* Title */}
            <h2 className="text-xl font-bold text-gray-900 mb-3">
              Are you sure you want to delete?
            </h2>
            
            {/* Description */}
            <p className="text-gray-600 text-sm mb-6 leading-relaxed">
              Click on Agree if you like to delete this trip permanently. If not click on cancel!
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
                  'Agree'
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
  )
}
