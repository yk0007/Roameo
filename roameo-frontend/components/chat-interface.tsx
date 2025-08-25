"use client"

import React, { type ReactNode, isValidElement, cloneElement } from "react"
import { useEffect, useMemo, useRef, useState } from "react"
import ReactMarkdown from "react-markdown"
import { Send, Plus, Mic, ArrowDown } from "lucide-react"
import { Button } from "@/components/ui/button"
import { sendChat } from "@/lib/api"
import { InlinePlanningStatus } from "./inline-planning-status"
import { TypingIndicator } from "./typing-indicator"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card"
import { CachedImage } from "./cached-image"
import { CompactPoiCard } from "./poi-card"
import type { ChatMessage, POI } from "@/lib/types"

interface ChatInterfaceProps {
  messages: ChatMessage[]
  onSendMessage: (content: string) => void
  activeView: "chat" | "search" | "saved"
  onViewChange: (view: "chat" | "search" | "saved") => void
  isRightPanelVisible?: boolean
  activeRightView?: "map" | "itinerary"
  setActiveRightView?: (view: "map" | "itinerary") => void
  setIsRightPanelVisible?: (visible: boolean) => void
  pois?: POI[]
  isTyping?: boolean
  savedIds?: Set<string>
  onToggleSave?: (poi: POI, nextSaved: boolean) => void
  onAddPoi?: (poi: POI) => void
  onReplan?: (poi: POI) => void
}


export function ChatInterface({
  messages,
  onSendMessage,
  activeView,
  onViewChange,
  isRightPanelVisible,
  activeRightView,
  setActiveRightView,
  setIsRightPanelVisible,
  pois,
  isTyping,
  savedIds,
  onToggleSave,
  onAddPoi,
  onReplan,
}: ChatInterfaceProps) {
  const [inputValue, setInputValue] = useState("")
  const [showSuggestion, setShowSuggestion] = useState(false)
  const [showScrollToBottom, setShowScrollToBottom] = useState(false)
  const [userJustSent, setUserJustSent] = useState(false)
  const [responseType, setResponseType] = useState<'general' | 'planning' | null>(null)
  const lastAssistantIdRef = useRef<string | null>(null)
  const typedMessageIdsRef = useRef<Set<string>>(new Set())
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const safeMessages = messages || []
  const lastMessage = useMemo(() => {
    return safeMessages.length > 0 ? safeMessages[safeMessages.length - 1] : undefined
  }, [safeMessages])

  const lastAssistant = useMemo(() => {
    for (let i = safeMessages.length - 1; i >= 0; i--) {
      if (safeMessages[i].role === "assistant") return safeMessages[i]
    }
    return undefined
  }, [safeMessages])

  // Decide when to show the suggestion chip: only after a substantive assistant reply,
  // only once per assistant turn, and only on the Chat view, and not when user just sent a message
  useEffect(() => {
    if (!lastAssistant) return
    if (activeView !== "chat") return
    if (userJustSent) return // Hide suggestions when user just sent a message
    const isNewAssistantTurn = lastAssistantIdRef.current !== lastAssistant.id
    if (isNewAssistantTurn) {
      lastAssistantIdRef.current = lastAssistant.id
      const shouldShow = (lastAssistant.content?.length || 0) > 80 && safeMessages.length >= 3
      setShowSuggestion(shouldShow)
      setUserJustSent(false) // Reset after assistant responds
    }
  }, [lastAssistant, safeMessages.length, activeView, userJustSent])

  const scrollToBottom = (behavior: "smooth" | "auto" = "smooth") => {
    messagesEndRef.current?.scrollIntoView({ behavior })
  }

  useEffect(() => {
    // Auto-scroll on new messages
    scrollToBottom("auto")
  }, [safeMessages, isTyping])

  // Reset response type when typing stops
  useEffect(() => {
    if (!isTyping) {
      setResponseType(null)
    }
  }, [isTyping])

  const handleScroll = () => {
    const container = scrollContainerRef.current
    if (container) {
      const isScrolledUp = container.scrollHeight - container.scrollTop > container.clientHeight + 150
      setShowScrollToBottom(isScrolledUp)
    }
  }

  // Shared strong renderer that turns bold names into POI hover cards when matched
  const StrongWithPoiHover: React.FC<any> = ({ children, ..._props }) => {
    const flat = React.Children.toArray(children)
      .map((c: any) => (typeof c === 'string' ? c : ''))
      .join('')
      .trim()
    const matchingPoi = flat
      ? pois?.find((p: any) =>
          p.name.toLowerCase().includes(flat.toLowerCase()) ||
          flat.toLowerCase().includes(p.name.toLowerCase())
        )
      : undefined
    if (matchingPoi && onAddPoi) {
      const isSaved = savedIds?.has?.(matchingPoi.id || '') ?? false
      return (
        <HoverCard>
          <HoverCardTrigger asChild>
            <strong className="font-semibold text-gray-900 cursor-pointer hover:underline" onClick={() => onAddPoi(matchingPoi)}>
              {children}
            </strong>
          </HoverCardTrigger>
          <HoverCardContent className="w-80 p-0" side="top" align="start">
            <CompactPoiCard
              poi={matchingPoi}
              isSaved={isSaved}
              isItineraryItem={false}
              onToggleSave={(poi, next) => onToggleSave?.(poi, next)}
              onAddPoi={(poi) => onAddPoi?.(poi)}
              onReplan={(poi) => onReplan?.(poi)}
            />
          </HoverCardContent>
        </HoverCard>
      )
    }
    return <strong className="font-semibold text-gray-900">{renderWithPoiHover(children, true)}</strong>
  }

  const renderWithPoiHover = (children: React.ReactNode, isAssistant: boolean = false): React.ReactNode => {
    const processNode = (node: React.ReactNode): React.ReactNode => {
      if (typeof node === 'string') {
        // Split by POI markers (📍, 📌, pushpin U+1F4CD) with optional VS16 and process each part
        const parts = node.split(/((?:📍|📌|\u{1F4CD})(?:\uFE0F)?\s*[\w\s.'-]+\b)/gu)      
          
        return parts.map((part: string, index: number) => {
          if (/^(?:📍|📌|\u{1F4CD})(?:\uFE0F)?/u.test(part)) {
            const pinMatch = part.match(/^(?:📍|📌|\u{1F4CD})(?:\uFE0F)?/u)
            const pin = pinMatch ? pinMatch[0] : '📍'
            const poiName = part.replace(/^(?:📍|📌|\u{1F4CD})(?:\uFE0F)?/u, '').trim()
            const matchingPoi = pois?.find((poi: any) => 
              poi.name.toLowerCase().includes(poiName.toLowerCase()) ||
              poiName.toLowerCase().includes(poi.name.toLowerCase())
            )
            
            if (matchingPoi && isAssistant && onAddPoi) {
              const isSaved = savedIds?.has?.(matchingPoi.id || '') ?? false
              return (
                <HoverCard key={`${poiName}-${index}`}>
                  <HoverCardTrigger asChild>
                    <span
                      className="cursor-pointer text-blue-600 hover:text-blue-800 hover:underline font-semibold"
                      onClick={() => onAddPoi(matchingPoi)}
                      title={`Click to add ${matchingPoi.name} to map`}
                    >
                      {pin} {poiName}
                    </span>
                  </HoverCardTrigger>
                  <HoverCardContent className="w-80 p-0" side="top" align="start">
                    <CompactPoiCard
                      poi={matchingPoi}
                      isSaved={isSaved}
                      isItineraryItem={false}
                      onToggleSave={(poi, next) => onToggleSave?.(poi, next)}
                      onAddPoi={(poi) => onAddPoi?.(poi)}
                      onReplan={(poi) => onReplan?.(poi)}
                    />
                  </HoverCardContent>
                </HoverCard>
              )
            }
            // Fallback: bold the pin text even if we don't have a matching POI
            return (
              <strong key={`${poiName}-${index}`} className="font-semibold text-gray-900">{pin} {poiName}</strong>
            )
          }
          return part
        })
      }

      if (React.isValidElement(node)) {
        return React.cloneElement(node as React.ReactElement<any>, {
          children: React.Children.map((node as any).props.children, (child) => processNode(child))
        })
      }

      return node
    }

    return processNode(children)
  }

  const renderFormattedContent = (content: string) => {
    // Clean the content thoroughly
    let cleanedContent = content
      .replace(/,?\[object Object\],?/g, "")
      .replace(/\[object Object\]/g, "")
      .replace(/object Object/g, "")
      .replace(/,\s*,/g, ",")
      .replace(/^\s*,\s*/, "")
      .replace(/\s*,\s*$/, "")
      .trim();

    // Helper: remove bold around POI names so asterisks don't show and POIs aren't bolded
    const stripPoiBold = (text: string) => text // deprecated, keep for backward compat but no-op

    // Helper: remove bold around currency ranges so **₹500 - ₹1000** -> ₹500 - ₹1000
    const stripPriceBold = (text: string) =>
      text
        // Specifically handle lines that include Price range: **₹...**
        .replace(/(Price range:\s*)\*\*([^*]+)\*\*/gi, '$1$2')
        // Fallback: any bold that starts with a currency or digit, unbold it
        .replace(/\*\*([₹$€£]\s?[^*]+)\*\*/g, '$1');

    // Debug: print the raw and cleaned markdown so we can compare with the UI rendering
    try {
      // Only log non-trivial assistant payloads
      if (cleanedContent && cleanedContent.length > 0) {
      }
    } catch {}

    if (!cleanedContent) {
      return <div className="text-gray-500 italic">Empty content</div>;
    }

    // Process questions, but do not return early; we'll append them after itinerary rendering
    const questionHeader = "🤔 Got More Questions?"
    let extractedQuestions: string[] = []
    if (cleanedContent.includes(questionHeader)) {
      const parts = cleanedContent.split(questionHeader)
      cleanedContent = parts[0]
      const questionsText = parts[1] || ""
      extractedQuestions = questionsText.match(/"(.*?)"/g)?.map(q => q.slice(1, -1)) || []
    }

    // Process content line by line for better control (preserve blank lines for spacing)
    const lines = cleanedContent.split('\n')
    
    return (
      <div className="space-y-3.5">
        {lines.map((line, index) => {
          const trimmedLine = line.trim()
          // Per-line classification logs
          if (trimmedLine.length) {
          }
          
          // Main title (# *Title*) — italic, slightly lighter weight
          if (trimmedLine.match(/^#\s*\*.*\*$/)) {
            const title = trimmedLine.replace(/^#\s*\*(.+)\*$/, '$1')
            return <h1 key={index} className="text-[22px] md:text-2xl font-semibold italic mb-2.5 text-gray-900">{renderWithPoiHover(title, true)}</h1>
          }
          
          // Day headers (## *Day 1:*) — italic, lighter weight with subtle divider
          if (trimmedLine.match(/^##\s*\*.*\*$/)) {
            const dayTitle = trimmedLine.replace(/^##\s*\*(.+)\*$/, '$1')
            return <h2 key={index} className="text-xl font-medium italic mb-2.5 mt-5 text-gray-900 border-b border-gray-200 pb-2">{renderWithPoiHover(dayTitle, true)}</h2>
          }
          
          // Time section headers (### ☀️ Morning) — orange banner with strong left bar
          if (trimmedLine.match(/^###\s*(☀️|⛵|🌅|🚗|🛍️|🍽️|🏨|✈️)\s*(Morning|Afternoon|Evening|Shopping|Heritage|Departure)/)) {
            const sectionTitle = trimmedLine.replace(/^###\s*/, '')
            return (
              <div key={index} className="mt-5 mb-2">
                <div className="bg-orange-50 border-l-8 border-orange-400 rounded-md px-3 py-1.5">
                  <h3 className="font-semibold text-gray-900 text-base">{renderWithPoiHover(sectionTitle, true)}</h3>
                </div>
              </div>
            )
          }

          // Skip stray '###' lines (sometimes produced by model)
          if (/^###\s*$/.test(trimmedLine)) {
            return <div key={index} className="h-1.5" />
          }

          // Generic ### headings (e.g., Accommodation & Meals, Estimated Budget)
          if (trimmedLine.startsWith('### ')) {
            const sectionTitle = trimmedLine.replace(/^###\s*/, '')
            return (
              <div key={index} className="mt-5 mb-2">
                <div className="bg-orange-50 border-l-8 border-orange-400 rounded-md px-3 py-2">
                  <h3 className="font-semibold text-gray-900 text-base">{renderWithPoiHover(sectionTitle, true)}</h3>
                </div>
              </div>
            )
          }

          // Meals subheading emitted as a bullet (e.g., "*  Meals:" or "**Meals:**") — render as banner
          {
            const normalized = trimmedLine.replace(/^\*\s*/, '').trim()
            if (/^\**\s*Meals:?\s*\**$/i.test(normalized)) {
              return (
                <div key={index} className="mt-4 mb-1">
                  <div className="bg-orange-50 border-l-8 border-orange-400 rounded-md px-3 py-1.5">
                    <h3 className="font-semibold text-gray-900 text-base">Meals</h3>
                  </div>
                </div>
              )
            }
          }

          // Time entries (* **9:00 AM:** Description)
          if (trimmedLine.match(/^\*\s*\*\*\d{1,2}:\d{2}\s*(AM|PM):\*\*/)) {
            const timeMatch = trimmedLine.match(/^\*\s*\*\*(\d{1,2}:\d{2}\s*(AM|PM)):\*\*(.*)/)
            if (timeMatch) {
              const time = timeMatch[1]
              const description = (timeMatch[3] || '').trim()
              
              return (
                <div key={index} className="flex items-start gap-3 py-1.5 ml-4">
                  <span className="w-2 h-2 bg-gray-300 rounded-full mt-2 flex-shrink-0"></span>
                  <div className="flex-1">
                    <div className="font-bold text-gray-900 mb-1">{time}:</div>
                    <div className="text-gray-600 leading-relaxed">
                      <ReactMarkdown components={{
                        p: ({ children }) => <span>{renderWithPoiHover(children, true)}</span>,
                        strong: StrongWithPoiHover
                      }}>
                        {stripPriceBold(description)}
                      </ReactMarkdown>
                    </div>
                  </div>
                </div>
              )
            }
          }

          // Regular bullet points (* Item)
          if (trimmedLine.match(/^\*\s*\*\*.*\*\*/)) {
            const content = stripPriceBold(trimmedLine.replace(/^\*\s+/, ''))
            return (
              <div key={index} className="flex items-start gap-3 py-0.5 ml-4">
                <span className="w-2 h-2 bg-gray-300 rounded-full mt-2 flex-shrink-0"></span>
                <div className="text-gray-600 leading-relaxed">
                  <ReactMarkdown components={{
                    p: ({ children }) => <span>{renderWithPoiHover(children, true)}</span>,
                    strong: StrongWithPoiHover
                  }}>
                    {content}
                  </ReactMarkdown>
                </div>
              </div>
            )
          }

          // Generic bullets (e.g., "*   📍 Place To Bee: ...")
          if (/^\*\s+/.test(trimmedLine)) {
            const content = stripPriceBold(stripPoiBold(trimmedLine.replace(/^\*\s+/, '')))
            return (
              <div key={index} className="flex items-start gap-3 py-0.5 ml-4">
                <span className="w-2 h-2 bg-gray-300 rounded-full mt-2 flex-shrink-0"></span>
                <div className="text-gray-600 leading-relaxed">
                  <ReactMarkdown components={{
                    p: ({ children }) => <span>{renderWithPoiHover(children, true)}</span>,
                    strong: StrongWithPoiHover
                  }}>
                    {content}
                  </ReactMarkdown>
                </div>
              </div>
            )
          }
          
          // Preserve explicit blank lines for spacing
          if (!trimmedLine.length) {
            return <div key={index} className="h-2" />
          }

          // Descriptive paragraphs (no markdown prefix)
          if (trimmedLine.match(/^[A-Z].*[.!]$/) && !trimmedLine.includes('#')) {
            return (
              <div key={index} className="mb-4 leading-relaxed italic text-gray-500 text-sm pl-4 border-l-2 border-gray-200">
                <ReactMarkdown components={{
                  p: ({ children }) => <span>{renderWithPoiHover(children, true)}</span>,
                  strong: StrongWithPoiHover
                }}>
                  {stripPriceBold(trimmedLine)}
                </ReactMarkdown>
              </div>
            )
          }
          
          // Regular paragraphs
          return (
            <div key={index} className="mb-2.5 leading-relaxed text-gray-700">
              <ReactMarkdown components={{
                p: ({ children }) => <span>{renderWithPoiHover(children, true)}</span>,
                strong: StrongWithPoiHover
              }}>
                {stripPriceBold(trimmedLine)}
              </ReactMarkdown>
            </div>
          )
        })}

        {extractedQuestions.length > 0 && (
          <div className="mt-6">
            <h3 className="text-md font-bold mb-2">{questionHeader}</h3>
            <div className="flex flex-col space-y-2 mt-2">
              {extractedQuestions.map((q, i) => (
                <button
                  key={i}
                  className="bg-white/80 border border-zinc-200 rounded-lg px-3 py-2 text-sm text-left hover:bg-zinc-50 transition-colors"
                  onClick={() => onSendMessage(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }


const handleSubmit = (e: React.FormEvent) => {
  e.preventDefault()
  if (inputValue.trim()) {
    setUserJustSent(true) // Hide suggestions when user sends a message
    setShowSuggestion(false)
    
    // Determine response type based on message content
    const message = inputValue.trim().toLowerCase()
    const isPlanningMessage = message.includes('plan') || message.includes('itinerary') || 
                             message.includes('trip') || message.includes('travel') ||
                             message.includes('visit') || message.includes('go to') ||
                             message.includes('destination') || message.includes('days')
    
    setResponseType(isPlanningMessage ? 'planning' : 'general')
    onSendMessage(inputValue.trim())
    setInputValue("")
  }
}

return (
    <div className="flex flex-col h-full bg-white relative">
      <div ref={scrollContainerRef} onScroll={handleScroll} className={`flex-1 overflow-y-auto ${isRightPanelVisible ? "p-6" : "px-6 py-4 w-full"} pt-20`}>
        <div className="w-full">
          {safeMessages.map((message) => (
            <div key={message.id} className={`flex gap-3 mb-4 ${message.role === "user" ? "justify-end" : "justify-start"}`}>
              <Avatar className={`w-8 h-8 flex-shrink-0 ${message.role === "user" ? "order-2" : "order-1"}`}>
                <AvatarFallback className={message.role === "user" ? "bg-orange-500 text-white" : "bg-black text-white flex items-center justify-center"}>
                  {message.role === "user" ? "N" : <div className="w-3 h-3 bg-white rounded-full"></div>}
                </AvatarFallback>
              </Avatar>
              <div className={`max-w-[78%] ${message.role === "user" ? "order-1" : "order-2"}`}>
                <div className={`prose prose-sm max-w-none px-4 py-3 rounded-2xl shadow-sm ${message.role === "user" ? "bg-orange-50 border border-orange-100" : "bg-white/85 border border-zinc-200"}`}>
                  <div className="leading-relaxed">
                    {message.role === "user" ? (
                      <ReactMarkdown>{String(message.content).replace(/\\[object Object\\]/g, "").trim()}</ReactMarkdown>
                    ) : (
                      renderFormattedContent(String(message.content).replace(/\\[object Object\\]/g, ""))
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
          {isTyping && (
            <div className="flex gap-3 mb-4">
              <Avatar className="w-8 h-8 flex-shrink-0">
                <AvatarFallback className="bg-black text-white flex items-center justify-center">
                  <div className="w-3 h-3 bg-white rounded-full"></div>
                </AvatarFallback>
              </Avatar>
              <div className="max-w-[78%]">
                <div className="prose prose-sm max-w-none px-4 py-3 rounded-2xl shadow-sm bg-white/85 border border-zinc-200">
                  {responseType === 'planning' ? (
                    <InlinePlanningStatus isVisible={true} />
                  ) : (
                    <TypingIndicator isVisible={true} />
                  )}
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {showScrollToBottom && (
        <Button onClick={() => scrollToBottom()} variant="outline" size="icon" className="absolute bottom-32 right-10 z-10 rounded-full shadow-lg bg-white/80 backdrop-blur-sm animate-in fade-in">
          <ArrowDown className="h-5 w-5" />
        </Button>
      )}

      <div className={`w-full ${isRightPanelVisible ? "p-6" : "px-8 py-6"}`}>
        <form onSubmit={handleSubmit} className="relative">
          <div className="bg-white/80 backdrop-blur-md border rounded-3xl shadow-lg p-4 border-zinc-300">
            <div className="flex items-center gap-3 flex-row">
              <Button type="button" variant="ghost" size="sm" className="flex-shrink-0 w-10 h-10 rounded-full hover:bg-white/80">
                <Plus className="w-5 h-5" />
              </Button>
              <Input
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Ask anything..."
                className="flex-1 border-0 bg-transparent text-lg placeholder:text-gray-500 focus-visible:ring-0 focus-visible:ring-offset-0 px-0 shadow-none"
              />
              <div className="flex items-center gap-2 flex-shrink-0">
                <Button type="button" variant="ghost" size="sm" className="w-10 h-10 rounded-full hover:bg-white/80">
                  <Mic className="w-5 h-5" />
                </Button>
                <Button type="submit" size="sm" className="w-10 h-10 rounded-full bg-black text-white">
                  <Send className="w-5 h-5" />
                </Button>
              </div>
            </div>
          </div>
        </form>
      </div>
    </div>
  )
}
