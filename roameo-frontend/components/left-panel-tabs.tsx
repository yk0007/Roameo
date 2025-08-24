"use client"

import { Button } from "@/components/ui/button"
import { MessageCircle, Search, Bookmark } from "lucide-react"

interface LeftPanelTabsProps {
  activeView: "chat" | "search" | "saved"
  onViewChange: (view: "chat" | "search" | "saved") => void
}

export function LeftPanelTabs({ activeView, onViewChange }: LeftPanelTabsProps) {
  const renderButton = (view: "chat" | "search" | "saved", label: string, icon: React.ReactNode) => {
    const isActive = activeView === view
    return (
      <Button
        variant={isActive ? "default" : "ghost"}
        onClick={() => onViewChange(view)}
        className={`relative flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-full transition-colors ${
          isActive
            ? "bg-black text-white hover:bg-gray-800 hover:text-white"
            : "text-gray-600 hover:bg-gray-100 hover:text-black"
        }`}
      >
        {icon}
        {label}
      </Button>
    )
  }

  return (
    <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 bg-white/80 backdrop-blur-lg rounded-full p-1 border border-white/30 shadow-xl transition-all duration-300">
      {renderButton("chat", "Chat", <MessageCircle className="w-4 h-4" />)}
      {renderButton("search", "Search", <Search className="w-4 h-4" />)}
      {renderButton("saved", "Saved", <Bookmark className="w-4 h-4" />)}
    </div>
  )
}
