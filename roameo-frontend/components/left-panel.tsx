"use client"
import { MessageCircle, Search, Bookmark } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ChatInterface } from "@/components/chat-interface"
import { SearchInterface } from "@/components/search-interface"

interface LeftPanelProps {
  activeView: "chat" | "search" | "saved"
  onViewChange: (view: "chat" | "search" | "saved") => void
  messages: any[]
  onSendMessage: (content: string) => void
}

export function LeftPanel({ activeView, onViewChange, messages, onSendMessage }: LeftPanelProps) {
  return (
    <div className="bg-white flex flex-col relative h-full">
      <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 bg-white/90 backdrop-blur-sm rounded-full p-1 border border-gray-200 shadow-lg">
        <Button
          variant={activeView === "chat" ? "default" : "ghost"}
          size="sm"
          onClick={() => onViewChange("chat")}
          className={`rounded-full px-4 ${activeView === "chat" ? "bg-black text-white hover:bg-gray-800" : "hover:bg-gray-100"}`}
        >
          <MessageCircle className="w-4 h-4 mr-1" />
          Chat
        </Button>
        <Button
          variant={activeView === "search" ? "default" : "ghost"}
          size="sm"
          onClick={() => onViewChange("search")}
          className={`rounded-full px-4 ${activeView === "search" ? "bg-black text-white hover:bg-gray-800" : "hover:bg-gray-100"}`}
        >
          <Search className="w-4 h-4 mr-1" />
          Search
        </Button>
        <Button
          variant={activeView === "saved" ? "default" : "ghost"}
          size="sm"
          onClick={() => onViewChange("saved")}
          className={`rounded-full px-4 ${activeView === "saved" ? "bg-black text-white hover:bg-gray-800" : "hover:bg-gray-100"}`}
        >
          <Bookmark className="w-4 h-4 mr-1" />
          Saved
        </Button>
      </div>

      <div className="flex-1 overflow-hidden h-full">
        {activeView === "chat" && (
          <div className="h-full">
            <ChatInterface
              messages={messages}
              onSendMessage={onSendMessage}
              activeView={activeView}
              onViewChange={onViewChange}
              isRightPanelVisible={false}
            />
          </div>
        )}

        {activeView === "search" && (
          <div className="h-full overflow-y-auto p-4 pt-20">
            <SearchInterface/>
          </div>
        )}

        {activeView === "saved" && (
          <div className="h-full flex items-center justify-center pt-20">
            <div className="text-center text-gray-500">
              <Bookmark className="w-12 h-12 mx-auto mb-4 text-gray-300" />
              <p>Your saved places will appear here</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
