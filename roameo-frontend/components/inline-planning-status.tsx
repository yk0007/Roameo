"use client"

import { useEffect, useState } from "react"

const PLANNING_PHRASES = [
  "planning route…",
  "scanning nearby POIs…", 
  "ranking best places…",
  "picking hotels…",
  "optimizing itinerary…",
  "finalizing day plan…"
]

interface InlinePlanningStatusProps {
  isVisible: boolean
  onComplete?: () => void
}

export function InlinePlanningStatus({ isVisible, onComplete }: InlinePlanningStatusProps) {
  const [currentPhraseIndex, setCurrentPhraseIndex] = useState(0)

  useEffect(() => {
    if (!isVisible) {
      setCurrentPhraseIndex(0)
      return
    }

    // Cycle through phrases without typing animation
    const interval = setInterval(() => {
      setCurrentPhraseIndex((prev) => {
        if (prev < PLANNING_PHRASES.length - 1) {
          return prev + 1
        } else {
          return 0 // Loop back to start
        }
      })
    }, 2000) // Change phrase every 2 seconds

    return () => clearInterval(interval)
  }, [isVisible])

  if (!isVisible) return null

  return (
    <div className="flex items-center text-gray-600 text-sm font-medium py-2">
      <span>{PLANNING_PHRASES[currentPhraseIndex]}</span>
    </div>
  )
}
