"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { MapPin, Star, Clock, Search } from "lucide-react"
import { useState } from "react"

interface POIPanelProps {
  pois: any[]
  onPOIUpdate: (pois: any[]) => void
}

export function POIPanel({ pois, onPOIUpdate }: POIPanelProps) {
  const [searchTerm, setSearchTerm] = useState("")
  const [filterType, setFilterType] = useState("all")

  const filteredPOIs = pois.filter((poi) => {
    const matchesSearch = poi.name.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesFilter = filterType === "all" || poi.type === filterType
    return matchesSearch && matchesFilter
  })

  const poiTypes = [...new Set(pois.map((poi) => poi.type))]

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-xl font-bold mb-2">Points of Interest</h2>
        <p className="text-sm text-muted-foreground">{pois.length} locations found</p>
      </div>

      <div className="space-y-3">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search locations..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>

        <div className="flex gap-2 flex-wrap">
          <Button size="sm" variant={filterType === "all" ? "default" : "outline"} onClick={() => setFilterType("all")}>
            All
          </Button>
          {poiTypes.map((type) => (
            <Button
              key={type}
              size="sm"
              variant={filterType === type ? "default" : "outline"}
              onClick={() => setFilterType(type)}
            >
              {type}
            </Button>
          ))}
        </div>
      </div>

      <div className="space-y-3">
        {filteredPOIs.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <MapPin className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No locations match your search</p>
          </div>
        ) : (
          filteredPOIs.map((poi) => (
            <Card key={poi.id} className="hover:shadow-md transition-shadow cursor-pointer">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <CardTitle className="text-base">{poi.name}</CardTitle>
                    <CardDescription className="flex items-center gap-1 mt-1">
                      <MapPin className="w-3 h-3" />
                      {poi.location}
                    </CardDescription>
                  </div>
                  <Badge variant="secondary" className="ml-2">
                    {poi.type}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <p className="text-sm text-muted-foreground mb-3">{poi.description}</p>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3 text-xs text-muted-foreground">
                    <div className="flex items-center gap-1">
                      <Star className="w-3 h-3 fill-yellow-400 text-yellow-400" />
                      4.5
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      2-3 hours
                    </div>
                  </div>
                  <Button size="sm" variant="ghost">
                    View Details
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  )
}
