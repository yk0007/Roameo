"use client"

import { Button } from "@/components/ui/button"
import { MessageCircle, Search, Bookmark, MapPin, Calendar } from "lucide-react"

interface LeftPanelTabsProps {
  activeView: "chat" | "search" | "saved"
  onViewChange: (view: "chat" | "search" | "saved") => void
  activeRightView?: "map" | "itinerary"
  onRightViewChange?: (view: "map" | "itinerary") => void
  showRightTabs?: boolean
  suppressRightActive?: boolean
}

export function LeftPanelTabs({
  activeView,
  onViewChange,
  activeRightView,
  onRightViewChange,
  showRightTabs = false,
  suppressRightActive = false,
}: LeftPanelTabsProps) {
  const buttonClassName = (isActive: boolean) =>
    `relative flex h-[31px] items-center gap-2 rounded-full px-4 text-[13px] transition-colors ${
      isActive
        ? "bg-black text-white hover:bg-gray-800 hover:text-white"
        : "text-gray-500 hover:bg-transparent hover:text-black"
    }`

  const renderButton = (view: "chat" | "search" | "saved", label: string, icon: React.ReactNode) => {
    const isActive = activeView === view
    return (
      <Button
        variant={isActive ? "default" : "ghost"}
        onClick={() => onViewChange(view)}
        className={buttonClassName(isActive)}
      >
        {icon}
        {label}
      </Button>
    )
  }

  const renderRightButton = (
    view: "map" | "itinerary",
    label: string,
    icon: React.ReactNode
  ) => {
    const isActive = !suppressRightActive && activeRightView === view
    return (
      <Button
        variant={isActive ? "default" : "ghost"}
        onClick={() => onRightViewChange?.(view)}
        className={buttonClassName(isActive)}
      >
        {icon}
        {label}
      </Button>
    )
  }

  return (
    <div className="pointer-events-none absolute inset-x-0 top-4 z-20 flex justify-center px-5">
      <div className="pointer-events-auto flex items-center gap-0.5 rounded-full border border-white/40 bg-white/50 p-[5px] shadow-[0_8px_32px_rgba(15,23,42,0.12),0_2px_8px_rgba(15,23,42,0.06)] backdrop-blur-2xl">
        {renderButton("chat", "Chat", <MessageCircle className="w-4 h-4" />)}
        {renderButton("search", "Search", <Search className="w-4 h-4" />)}
        {renderButton("saved", "Saved", <Bookmark className="w-4 h-4" />)}
        {showRightTabs && (
          <>
            {renderRightButton("map", "Map", <MapPin className="w-4 h-4" />)}
            {renderRightButton("itinerary", "Itinerary", <Calendar className="w-4 h-4" />)}
          </>
        )}
      </div>
    </div>
  )
}
