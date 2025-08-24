"use client"

import { Button } from "@/components/ui/button"
import { MapPin, Calendar, Route } from "lucide-react"

interface HeaderProps {
  activeView: "plan" | "map" | "itinerary"
  onViewChange: (view: "plan" | "map" | "itinerary") => void
}

export function Header({ activeView, onViewChange }: HeaderProps) {
  return (
    <header className="h-16 border-b border-white/30 bg-white/80 backdrop-blur-md px-6 flex items-center justify-between shadow-lg">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
          <MapPin className="w-5 h-5 text-primary-foreground" />
        </div>
        <h1 className="text-xl font-bold text-foreground">Roameo</h1>
      </div>

      <nav className="flex items-center gap-2">
        <Button
          variant={activeView === "plan" ? "default" : "ghost"}
          size="sm"
          onClick={() => onViewChange("plan")}
          className="gap-2"
        >
          <Route className="w-4 h-4" />
          Plan Trip
        </Button>
        <Button
          variant={activeView === "map" ? "default" : "ghost"}
          size="sm"
          onClick={() => onViewChange("map")}
          className="gap-2"
        >
          <MapPin className="w-4 h-4" />
          Map
        </Button>
        <Button
          variant={activeView === "itinerary" ? "default" : "ghost"}
          size="sm"
          onClick={() => onViewChange("itinerary")}
          className="gap-2"
        >
          <Calendar className="w-4 h-4" />
          Itinerary
        </Button>
      </nav>
    </header>
  )
}
