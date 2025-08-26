"use client"

import { useEffect, useMemo, useRef, useState, useCallback } from "react"
import { Search, Calendar, Users, SlidersHorizontal, Heart, Plus, Hotel, MapPin, Star, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { CachedImage } from "@/components/cached-image"
import { SearchCard } from "@/components/search-card"
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
}

export function SearchInterface({ activeView, onViewChange, results, savedIds, itineraryPoiIds, onAddPoi, onToggleSave, onReplan }: SearchInterfaceProps) {
  const [activeTab, setActiveTab] = useState("Stays")
  const [searchQuery, setSearchQuery] = useState("")
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const scrollPosRef = useRef<Record<string, number>>({ Stays: 0, Restaurants: 0, Attractions: 0 })

  const tabs = ["Stays", "Restaurants", "Attractions"]

  const allResults = useMemo(() => {
    const q = searchQuery.trim().toLowerCase()
    let arr: POI[] = []
    
    if (!results) return arr
    if (activeTab === "Stays") arr = results.stays || []
    if (activeTab === "Restaurants") arr = results.restaurants || []
    if (activeTab === "Attractions") arr = results.attractions || []
    
    if (!q) return arr
    return arr.filter((p) => p.name.toLowerCase().includes(q) || (p.address || "").toLowerCase().includes(q))
  }, [results, activeTab, searchQuery])

  const list = useMemo(() => {
    return allResults
  }, [allResults])



  // Restore scroll position on tab switch
  useEffect(() => {
    const el = scrollRef.current
    if (el) {
      const y = scrollPosRef.current[activeTab] ?? 0
      el.scrollTop = y
    }
  }, [activeTab])

  return (
    <div className="flex flex-col h-full bg-white pt-20">
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
          <Button className="bg-black/80 text-white hover:bg-black/90 rounded-2xl px-6 backdrop-blur-sm" onClick={() => { /* hook to future manual search */ }}>
            Search
          </Button>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <Button variant="outline" size="sm" className="rounded-2xl bg-white/50 backdrop-blur-sm border-white/30">
            <Calendar className="w-4 h-4 mr-1" />
            Dates
          </Button>
          <Button variant="outline" size="sm" className="rounded-2xl bg-white/50 backdrop-blur-sm border-white/30">
            <Users className="w-4 h-4 mr-1" />Guests
          </Button>
          <Button variant="outline" size="sm" className="rounded-2xl bg-white/50 backdrop-blur-sm border-white/30">
            Location
          </Button>
          <Button variant="outline" size="sm" className="rounded-2xl bg-white/50 backdrop-blur-sm border-white/30">
            Any budget
          </Button>
          <Button variant="outline" size="sm" className="rounded-2xl bg-white/50 backdrop-blur-sm border-white/30">
            <SlidersHorizontal className="w-4 h-4 mr-1" />
            Filters
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
        {!results ? (
          <div className="text-sm text-gray-500">No results yet. Ask Roameo to search for places.</div>
        ) : list.length === 0 ? (
          <div className="text-sm text-gray-500">No matches for your filters.</div>
        ) : (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(320px,1fr))] gap-6">
            {list.map((poi) => (
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
        )}
      </div>
    </div>
  )
}
