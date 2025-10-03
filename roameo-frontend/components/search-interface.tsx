"use client"

import { useEffect, useMemo, useRef, useState, useCallback, memo } from "react"
import { Search, Calendar, Users, SlidersHorizontal, Heart, Plus, Hotel, MapPin, Star, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { CachedImage } from "@/components/cached-image"
import { SearchCard } from "@/components/search-card"
import { useSearchDebounce } from "@/lib/utils/performance"
import type { SearchResults, POI } from "@/lib/types"

interface SearchInterfaceProps {
  activeView?: "chat" | "search" | "saved"
  onViewChange?: (view: "chat" | "search" | "saved") => void
  results?: SearchResults
  savedIds?: Set<string>
  itineraryPoiIds?: Set<string>
  onAddPoi?: (poi: POI) => void
  onToggleSave?: (poi: POI, nextSaved: boolean) => void
  onReplan?: (poi?: POI) => void
  isLoading?: boolean
  searchStatus?: string
  hasBillingError?: boolean
}

export const SearchInterface = memo(function SearchInterface({ activeView, onViewChange, results, savedIds, itineraryPoiIds, onAddPoi, onToggleSave, onReplan, isLoading = false, searchStatus, hasBillingError = false }: SearchInterfaceProps) {
  const [activeTab, setActiveTab] = useState("Stays")
  const [searchQuery, setSearchQuery] = useState("")
  const [debouncedQuery, setDebouncedQuery] = useState("")
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const scrollPosRef = useRef<Record<string, number>>({ Stays: 0, Restaurants: 0, Attractions: 0 })

  const tabs = ["Stays", "Restaurants", "Attractions"]

  // Debounced search to avoid excessive filtering
  const debouncedSetQuery = useSearchDebounce((query: string) => {
    setDebouncedQuery(query)
  }, 300)

  // Trigger debounced search when query changes
  useEffect(() => {
    debouncedSetQuery(searchQuery)
  }, [searchQuery, debouncedSetQuery])

  const allResults = useMemo(() => {
    const q = debouncedQuery.trim().toLowerCase()
    let arr: POI[] = []
    
    if (!results) return arr
    if (activeTab === "Stays") arr = results.stays || []
    if (activeTab === "Restaurants") arr = results.restaurants || []
    if (activeTab === "Attractions") arr = results.attractions || []
    
    if (!q) return arr
    
    // Filter results based on search query
    return arr.filter((p) => 
      p.name.toLowerCase().includes(q) || (p.address || "").toLowerCase().includes(q)
    )
  }, [results, activeTab, debouncedQuery])

  // Restore scroll position on tab switch
  useEffect(() => {
    const el = scrollRef.current
    if (el) {
      const y = scrollPosRef.current[activeTab] ?? 0
      el.scrollTop = y
    }
  }, [activeTab])

  return (
    <div className="flex flex-col h-full bg-white pt-20 relative">
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
      
      <div className="flex items-center justify-center gap-2 p-2 border-b border-white/20">
        {tabs.map((tab) => (
          <Button
            key={tab}
            variant={activeTab === tab ? "default" : "ghost"}
            size="sm"
            onClick={() => setActiveTab(tab)}
            className={`backdrop-blur-md border rounded-full px-4 transition-all ${
              activeTab === tab ? "bg-black text-white hover:bg-gray-800" : "hover:bg-gray-100"
            }`}
          >
            {tab}
          </Button>
        ))}
      </div>

      {/* Search Bar */}
      <div className="p-4 border-b border-white/20 space-y-4">
        <div className="flex items-center gap-2">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search places by name or address"
              className="pl-10 rounded-2xl bg-white/80 backdrop-blur-sm border-white/30"
            />
          </div>
          <Button className="bg-black/80 text-white hover:bg-black/90 rounded-2xl px-6 backdrop-blur-sm" onClick={useCallback(() => { /* hook to future manual search */ }, [])}>
            Search
          </Button>
        </div>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-4"
        onScroll={(e) => {
          const el = e.currentTarget
          scrollPosRef.current[activeTab] = el.scrollTop
        }}
      >
        {hasBillingError ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center p-8 max-w-md">
              <div className="w-16 h-16 mx-auto mb-4 bg-red-100 rounded-full flex items-center justify-center">
                <svg className="w-8 h-8 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.732 15.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-3">Search Features Limited</h3>
              <p className="text-sm text-gray-600 mb-4">
                Google Maps API has reached its billing limit. Place search and recommendations are currently unavailable.
              </p>
              <div className="text-xs text-gray-500 bg-blue-50 p-3 rounded-lg">
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
          <>
            <div className="grid grid-cols-[repeat(auto-fill,minmax(320px,1fr))] gap-6">
              {allResults.map((poi) => (
                <SearchCard
                  key={poi.id}
                  poi={poi}
                  isSaved={savedIds?.has(poi.id) || false}
                  isItineraryItem={!!itineraryPoiIds?.has(poi.id)}
                  onToggleSave={onToggleSave ? (p: POI, n: boolean) => onToggleSave(p, n) : () => {}}
                  onAddPoi={onAddPoi ? (p: POI) => onAddPoi(p) : () => {}}
                  onReplan={onReplan ? (p: POI) => onReplan(p) : () => {}}
                />
              ))}
            </div>


          </>
        )}
      </div>
    </div>
  )
})
