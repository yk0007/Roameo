"use client"

import React from "react"
import { CompactPoiCard } from "./poi-card"
import type { POI } from "@/lib/types"

interface PoiMentionDetectorProps {
  content: string
  pois?: POI[]
  savedIds?: Set<string>
  onToggleSave?: (poi: POI, nextSaved: boolean) => void
  onAddPoi?: (poi: POI) => void
  onReplan?: (poi: POI) => void
}

// Common POI keywords and patterns
const POI_PATTERNS = [
  // Restaurants & Food
  /\b(restaurant|cafe|coffee|dining|eatery|bistro|bar|pub|food court)\b/gi,
  // Hotels & Accommodation  
  /\b(hotel|resort|lodge|hostel|guesthouse|accommodation|stay)\b/gi,
  // Attractions & Places
  /\b(temple|church|mosque|museum|gallery|park|garden|beach|fort|palace|monument|market|mall|shopping)\b/gi,
  // Activities
  /\b(tour|trek|hike|cruise|safari|adventure|experience|activity)\b/gi,
]

// Extract potential POI names from text
function extractPOINames(text: string): string[] {
  const names: string[] = []
  
  // Look for quoted places: "Place Name"
  const quotedMatches = text.match(/"([^"]+)"/g)
  if (quotedMatches) {
    names.push(...quotedMatches.map(m => m.slice(1, -1)))
  }
  
  // Look for capitalized place names after common words
  const placePatterns = [
    /(?:visit|go to|check out|explore|see|at|near)\s+([A-Z][a-zA-Z\s]+?)(?:\s|,|\.|\!|\?|$)/g,
    /([A-Z][a-zA-Z\s]+?)\s+(?:temple|church|mosque|museum|gallery|park|beach|fort|palace|market|mall)/gi,
  ]
  
  placePatterns.forEach(pattern => {
    let match
    while ((match = pattern.exec(text)) !== null) {
      const placeName = match[1].trim()
      if (placeName.length > 2 && placeName.length < 50) {
        names.push(placeName)
      }
    }
  })
  
  return [...new Set(names)]
}

// Find matching POIs for mentioned names
function findMatchingPOIs(mentionedNames: string[], pois: POI[] = []): POI[] {
  const matches: POI[] = []
  
  mentionedNames.forEach(name => {
    const matchedPoi = pois.find(poi => 
      poi.name.toLowerCase().includes(name.toLowerCase()) ||
      name.toLowerCase().includes(poi.name.toLowerCase())
    )
    if (matchedPoi && !matches.find(m => m.id === matchedPoi.id)) {
      matches.push(matchedPoi)
    }
  })
  
  return matches
}

export function PoiMentionDetector({
  content,
  pois = [],
  savedIds = new Set(),
  onToggleSave = () => {},
  onAddPoi = () => {},
  onReplan = () => {}
}: PoiMentionDetectorProps) {
  // Extract mentioned POI names from content
  const mentionedNames = extractPOINames(content)
  
  // Find matching POIs from the available list
  const matchedPOIs = findMatchingPOIs(mentionedNames, pois)
  
  // Only show if we have matches
  if (matchedPOIs.length === 0) {
    return null
  }
  
  return (
    <div className="mt-4 space-y-3">
      <h4 className="text-sm font-medium text-gray-700 mb-2">
        🎯 Places mentioned:
      </h4>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {matchedPOIs.map(poi => (
          <CompactPoiCard
            key={poi.id}
            poi={poi}
            isSaved={savedIds.has(poi.id)}
            onToggleSave={onToggleSave}
            onAddPoi={onAddPoi}
            onReplan={onReplan}
          />
        ))}
      </div>
    </div>
  )
}
