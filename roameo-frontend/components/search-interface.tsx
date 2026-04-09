"use client"

import { useEffect, useMemo, useRef, useState, useCallback, memo } from "react"
import { Search, Calendar, Users, SlidersHorizontal, Heart, Plus, Hotel, MapPin, Star, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { CachedImage } from "@/components/cached-image"
import { SearchCard } from "@/components/search-card"
import { useSearchDebounce } from "@/lib/utils/performance"
import type { SearchResults, POI, SessionPlanningState } from "@/lib/types"

interface SearchInterfaceProps {
  activeView?: "chat" | "search" | "saved"
  onViewChange?: (view: "chat" | "search" | "saved") => void
  sessionId?: string
  destination?: string
  results?: SearchResults
  savedIds?: Set<string>
  itineraryPoiIds?: Set<string>
  onAddPoi?: (poi: POI) => void
  pendingAddPoiIds?: Set<string>
  onToggleSave?: (poi: POI, nextSaved: boolean) => void
  onReplan?: (poi?: POI) => void
  onDiscoverCategory?: (category: "stay" | "restaurant" | "attraction") => void
  isLoading?: boolean
  planningState?: SessionPlanningState
  searchStatus?: string
  hasBillingError?: boolean
  isSplitView?: boolean
}

export const SearchInterface = memo(function SearchInterface({ activeView, onViewChange, sessionId, destination, results, savedIds, itineraryPoiIds, onAddPoi, pendingAddPoiIds, onToggleSave, onReplan, onDiscoverCategory, isLoading = false, planningState, searchStatus, hasBillingError = false, isSplitView = false }: SearchInterfaceProps) {
  const [activeTab, setActiveTab] = useState("Stays")
  const [searchQuery, setSearchQuery] = useState("")
  const [debouncedQuery, setDebouncedQuery] = useState("")
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const scrollPosRef = useRef<Record<string, number>>({ Stays: 0, Restaurants: 0, Attractions: 0 })
  const requestedCategoriesRef = useRef<Set<string>>(new Set())

  const tabs = ["Stays", "Restaurants", "Attractions"]
  const contentShellClassName = "w-full px-6 lg:px-8 xl:px-10 2xl:px-12"

  // Debounced search to avoid excessive filtering
  const debouncedSetQuery = useSearchDebounce((query: string) => {
    setDebouncedQuery(query)
  }, 300)

  // Trigger debounced search when query changes
  useEffect(() => {
    debouncedSetQuery(searchQuery)
  }, [searchQuery, debouncedSetQuery])

  const currentResults = useMemo(() => {
    if (!results) return [] as POI[]
    if (activeTab === "Stays") return results.stays || []
    if (activeTab === "Restaurants") return results.restaurants || []
    return results.attractions || []
  }, [activeTab, results])

  const allResults = useMemo(() => {
    const q = debouncedQuery.trim().toLowerCase()
    if (!q) return currentResults

    return currentResults.filter((p) =>
      p.name.toLowerCase().includes(q) || (p.address || "").toLowerCase().includes(q)
    )
  }, [currentResults, debouncedQuery])

  useEffect(() => {
    requestedCategoriesRef.current.clear()
  }, [destination, sessionId])

  useEffect(() => {
    if (
      activeView !== "search" ||
      !sessionId ||
      !destination ||
      !onDiscoverCategory ||
      isLoading ||
      searchStatus
    ) {
      return
    }

    const category =
      activeTab === "Stays"
        ? "stay"
        : activeTab === "Restaurants"
          ? "restaurant"
          : "attraction"

    const requestKey = `${destination}::${category}`
    if (currentResults.length > 0 || requestedCategoriesRef.current.has(requestKey)) {
      return
    }

    requestedCategoriesRef.current.add(requestKey)
    onDiscoverCategory(category)
  }, [
    activeTab,
    activeView,
    currentResults.length,
    destination,
    isLoading,
    onDiscoverCategory,
    searchStatus,
    sessionId
  ])

  // Restore scroll position on tab switch
  useEffect(() => {
    const el = scrollRef.current
    if (el) {
      const y = scrollPosRef.current[activeTab] ?? 0
      el.scrollTop = y
    }
  }, [activeTab])

  return (
    <div className="relative flex h-full flex-col bg-white">
      {/* Loading Overlay */}
      {(isLoading || searchStatus) && (
        <div className="absolute inset-0 bg-white/80 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="flex flex-col items-center space-y-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <div className="text-sm font-medium text-gray-700">
              {searchStatus || "Searching for places..."}
            </div>
          </div>
        </div>
      )}
      
      <div className={`pb-5 pt-24 ${contentShellClassName}`}>
        <div className="flex items-center gap-4">
          <div className="flex shrink-0 items-center gap-2">
            {tabs.map((tab) => (
              <Button
                key={tab}
                variant={activeTab === tab ? "default" : "ghost"}
                size="sm"
                onClick={() => setActiveTab(tab)}
                className={`rounded-full border px-4 transition-all ${
                  activeTab === tab
                    ? "border-black bg-black text-white hover:bg-black/90"
                    : "border-[#e5e7eb] bg-white text-[#4b5563] hover:bg-[#f8fafc]"
                }`}
              >
                {tab}
              </Button>
            ))}
          </div>
          <div className="h-8 w-px shrink-0 bg-slate-200" />
          <div className="flex min-w-0 flex-1 items-center gap-3">
            <div className="relative min-w-0 flex-1">
              <Search className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
              <Input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search places by name or address"
                className="h-14 rounded-full border border-[rgba(255,255,255,0.7)] bg-[rgba(255,255,255,0.8)] pl-11 pr-4 text-[14px] shadow-[0_8px_32px_rgba(0,0,0,0.12)] backdrop-blur-xl"
              />
            </div>
            <Button
              className="h-14 shrink-0 rounded-full bg-[#4a4a4a] px-8 text-white hover:bg-[#3f3f3f]"
              onClick={useCallback(() => {
                if (activeView !== "search" || !sessionId || !destination || !onDiscoverCategory) {
                  return
                }

                const category =
                  activeTab === "Stays"
                    ? "stay"
                    : activeTab === "Restaurants"
                      ? "restaurant"
                      : "attraction"
                requestedCategoriesRef.current.delete(`${destination}::${category}`)
                onDiscoverCategory(category)
              }, [activeTab, activeView, destination, onDiscoverCategory, sessionId])}
            >
              Search
            </Button>
          </div>
        </div>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto py-4"
        onScroll={(e) => {
          const el = e.currentTarget
          scrollPosRef.current[activeTab] = el.scrollTop
        }}
      >
        <div className={contentShellClassName}>
          {hasBillingError ? (
            <div className="flex h-full items-center justify-center">
              <div className="max-w-md p-8 text-center">
                <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-red-100">
                  <svg className="h-8 w-8 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.732 15.5c-.77.833.192 2.5 1.732 2.5z" />
                  </svg>
                </div>
                <h3 className="mb-3 text-lg font-medium text-gray-900">Search Features Limited</h3>
                <p className="mb-4 text-sm text-gray-600">
                  Google Maps API has reached its billing limit. Place search and recommendations are currently unavailable.
                </p>
                <div className="rounded-lg bg-blue-50 p-3 text-xs text-gray-500">
                  <strong>Note:</strong> Your trip planning and chat features continue to work normally. 
                  Only place search is affected.
                </div>
              </div>
            </div>
          ) : !results ? (
            <div className="text-sm text-gray-500">No results yet. Ask Roameo to search for places.</div>
          ) : allResults.length === 0 ? (
            <div className="text-sm text-gray-500">No matches for your filters.</div>
          ) : (
            <div className={`grid gap-x-6 gap-y-10 ${isSplitView ? 'grid-cols-2' : 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4'}`}>
              {allResults.map((poi) => (
                <SearchCard
                  key={poi.id}
                  poi={poi}
                  isSaved={savedIds?.has(poi.id) || false}
                  isItineraryItem={!!itineraryPoiIds?.has(poi.id)}
                  isAddPending={!!pendingAddPoiIds?.has(poi.id)}
                  onToggleSave={onToggleSave ? (p: POI, n: boolean) => onToggleSave(p, n) : () => {}}
                  onAddPoi={onAddPoi ? (p: POI) => onAddPoi(p) : () => {}}
                  onReplan={onReplan ? (p: POI) => onReplan(p) : () => {}}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
})
