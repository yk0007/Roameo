"use client"

import { useState, useCallback, useMemo, memo } from "react"
import { useRouter } from "next/navigation"
import { supabase } from "@/lib/supabase/client"
import { ChevronDown, User, MapPin, Calendar, Users, LogOut, Settings, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import Link from "next/link"

interface Trip {
  id: string
  title: string
  origin: string
  destination: string
  destinations?: string[]
  duration: string
  travelers: string
  budget: string
}

interface TopNavigationProps {
  trip: Trip
  onTripUpdate: (trip: Trip) => void
  isRightPanelVisible: boolean
  onToggleRightPanel: () => void
  onSaveTrip: () => void
  onInvite: () => void
  inviteLink?: string
  onDeleteTrip?: () => void
  isDeleting?: boolean
  onReplan?: () => void
  onPopulateInput?: (text: string) => void
  onSignOut?: () => void
  showBottomBorder?: boolean // New prop to control gray border
}

export const TopNavigation = memo(function TopNavigation({
  trip,
  onTripUpdate,
  isRightPanelVisible,
  onToggleRightPanel,
  onSaveTrip,
  onInvite,
  inviteLink,
  onDeleteTrip,
  isDeleting,
  onReplan,
  onPopulateInput,
  onSignOut,
  showBottomBorder = false, // Default to false
}: TopNavigationProps) {
  const [editingField, setEditingField] = useState<string | null>(null)
  const [tempValues, setTempValues] = useState({
    title: trip.title,
    origin: trip.origin,
    destination: trip.destination,
    duration: trip.duration,
    travelers: trip.travelers,
    budget: trip.budget,
  })
  const [showInvitePopover, setShowInvitePopover] = useState(false)
  const router = useRouter()

  const handleSignOut = useCallback(async () => {
    // Navigate immediately for instant UX
    window.location.href = "/auth/login"
    // Sign out in background
    await supabase.auth.signOut()
  }, [])

  const handleLogoClick = useCallback(async () => {
    try {
      const { data: { session } } = await supabase.auth.getSession()
      if (session) {
        // Set flags to indicate navigation from chat
        window.history.replaceState({ ...window.history.state, fromChat: true }, '')
        sessionStorage.setItem('fromChat', 'true')
        router.push("/dashboard")
      } else {
        router.push("/")
      }
    } catch {
      router.push("/")
    }
  }, [router])

  const handleEdit = useCallback((field: string) => {
    setEditingField(field)
    setTempValues({
      title: trip.title,
      origin: trip.origin,
      destination: trip.destination,
      duration: trip.duration,
      travelers: trip.travelers,
      budget: trip.budget,
    })
  }, [trip])

  const handleSave = useCallback((field: string) => {
    onTripUpdate({
      ...trip,
      [field]: tempValues[field as keyof typeof tempValues],
    })
    setEditingField(null)
  }, [onTripUpdate, trip, tempValues])

  const handleCancel = useCallback(() => {
    setEditingField(null)
    setTempValues({
      title: trip.title,
      origin: trip.origin,
      destination: trip.destination,
      duration: trip.duration,
      travelers: trip.travelers,
      budget: trip.budget,
    })
  }, [trip])

  return (
    <div className={`flex items-center justify-between px-8 py-3 bg-white shadow-sm ${
      showBottomBorder ? 'border-b border-gray-200' : ''
    }`}>
      {/* Left Section - Logo & Trip Title */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-3">
          <button 
            onClick={handleLogoClick} 
            className="flex items-center gap-3"
          >
            <div className="w-8 h-8 bg-black rounded-full flex items-center justify-center">
              <div className="w-2 h-2 bg-white rounded-full"></div>
            </div>
            <span className="text-xl font-bold text-black">roameo</span>
          </button>
        </div>
        
        {/* Trip Title - Compact format like "Vizag → Coonoor, 2 days" */}
        <div className="flex items-center gap-2">
          {editingField === "title" ? (
            <div className="flex items-center gap-2">
              <Input
                value={tempValues.title}
                onChange={(e) => setTempValues({ ...tempValues, title: e.target.value })}
                className="w-64 h-8 text-sm border-gray-300 rounded-md"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSave("title")
                  if (e.key === "Escape") handleCancel()
                }}
                autoFocus
              />
              <Button size="sm" onClick={() => handleSave("title")} className="h-8 px-3 text-sm rounded-md">✓</Button>
            </div>
          ) : (
            <Button
              variant="ghost"
              className="text-lg font-medium px-2 py-1 h-auto hover:bg-gray-100 flex items-center gap-2"
              onClick={() => handleEdit("title")}
            >
              <span className="text-gray-900">{trip.title || "My Trip"}</span>
              <ChevronDown className="w-4 h-4 text-gray-500" />
            </Button>
          )}
        </div>
      </div>

      {/* Center Section - Trip Details in horizontal badges */}
      <div className="flex items-center gap-4">
        {/* Origin */}
        <div className="flex items-center gap-1">
          <MapPin className="w-4 h-4 text-gray-500" />
          {editingField === "origin" ? (
            <div className="flex items-center gap-2">
              <Input
                value={tempValues.origin}
                onChange={(e) => setTempValues({ ...tempValues, origin: e.target.value })}
                className="w-24 h-7 text-sm border-gray-300 rounded-md"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSave("origin")
                  if (e.key === "Escape") handleCancel()
                }}
                autoFocus
              />
              <Button size="sm" onClick={() => handleSave("origin")} className="h-7 px-2 text-sm rounded-md">✓</Button>
            </div>
          ) : (
            <span 
              onClick={() => handleEdit("origin")}
              className="text-xs text-gray-600 cursor-pointer hover:text-gray-900 font-medium"
            >
              {trip.origin || "Origin"}
            </span>
          )}
        </div>

        {/* Destination */}
        <div className="flex items-center gap-1">
          <MapPin className="w-4 h-4 text-gray-500" />
          {editingField === "destination" ? (
            <div className="flex items-center gap-2">
              <Input
                value={tempValues.destination}
                onChange={(e) => setTempValues({ ...tempValues, destination: e.target.value })}
                className="w-28 h-7 text-sm border-gray-300 rounded-md"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSave("destination")
                  if (e.key === "Escape") handleCancel()
                }}
                autoFocus
              />
              <Button size="sm" onClick={() => handleSave("destination")} className="h-7 px-2 text-sm rounded-md">✓</Button>
            </div>
          ) : (
            <span 
              onClick={() => handleEdit("destination")}
              className="text-xs text-gray-600 cursor-pointer hover:text-gray-900 font-medium"
            >
              {trip.destinations && trip.destinations.length > 1 
                ? `${trip.destinations.length} destinations` 
                : trip.destination || trip.destinations?.[0] || "Destination"}
            </span>
          )}
        </div>

        {/* Duration */}
        <div className="flex items-center gap-1">
          <Calendar className="w-4 h-4 text-gray-500" />
          {editingField === "duration" ? (
            <div className="flex items-center gap-2">
              <Input
                type="number"
                placeholder="Days"
                value={parseInt(/\d+/.exec(tempValues.duration || "")?.[0] || "", 10) as any}
                onChange={(e) => {
                  const v = e.target.value
                  setTempValues({ ...tempValues, duration: v ? `${v} days` : "" })
                }}
                className="w-20 h-7 text-sm border-gray-300 rounded-md"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSave("duration")
                  if (e.key === "Escape") handleCancel()
                }}
                autoFocus
              />
              <Button size="sm" onClick={() => handleSave("duration")} className="h-7 px-2 text-sm rounded-md">✓</Button>
            </div>
          ) : (
            <span 
              onClick={() => handleEdit("duration")}
              className="text-xs text-gray-600 cursor-pointer hover:text-gray-900 font-medium"
            >
              {trip.duration || "0 days"}
            </span>
          )}
        </div>

        {/* Travelers */}
        <div className="flex items-center gap-1">
          <Users className="w-4 h-4 text-gray-500" />
          {editingField === "travelers" ? (
            <div className="flex items-center gap-2">
              <Input
                type="number"
                placeholder="Travelers"
                value={parseInt(/\d+/.exec(tempValues.travelers || "")?.[0] || "", 10) as any}
                onChange={(e) => {
                  const v = e.target.value
                  setTempValues({ ...tempValues, travelers: v ? `${v} travelers` : "" })
                }}
                className="w-20 h-7 text-sm border-gray-300 rounded-md"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSave("travelers")
                  if (e.key === "Escape") handleCancel()
                }}
                autoFocus
              />
              <Button size="sm" onClick={() => handleSave("travelers")} className="h-7 px-2 text-sm rounded-md">✓</Button>
            </div>
          ) : (
            <span 
              onClick={() => handleEdit("travelers")}
              className="text-xs text-gray-600 cursor-pointer hover:text-gray-900 font-medium"
            >
              {trip.travelers || "1 travelers"}
            </span>
          )}
        </div>

        {/* Budget */}
        <div className="flex items-center gap-1">
          <span className="text-gray-500 font-medium">₹</span>
          {editingField === "budget" ? (
            <div className="flex items-center gap-2">
              <Input
                type="number"
                placeholder="Budget"
                value={parseInt((tempValues.budget || "").replace(/[^0-9]/g, "") || "", 10) as any}
                onChange={(e) => {
                  const v = e.target.value
                  setTempValues({ ...tempValues, budget: v })
                }}
                className="w-24 h-7 text-sm border-gray-300 rounded-md"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSave("budget")
                  if (e.key === "Escape") handleCancel()
                }}
                autoFocus
              />
              <Button size="sm" onClick={() => handleSave("budget")} className="h-7 px-2 text-sm rounded-md">✓</Button>
            </div>
          ) : (
            <span 
              onClick={() => handleEdit("budget")}
              className="text-xs text-gray-600 cursor-pointer hover:text-gray-900 font-medium"
            >
              Budget
            </span>
          )}
        </div>

        {/* Update Plan Button - positioned beside budget as requested */}
        {onPopulateInput && (
          <Button
            variant="outline"
            size="sm"
            className="border-gray-300 text-gray-700 hover:bg-gray-50 px-4 py-2 h-8 text-sm rounded-full"
            onClick={() => {
              const base = `Replan the itinerary optimizing for travel time and experience. Keep origin ${trip.origin || ""} and destination ${trip.destination || ""} for ${trip.duration || "?"}. ${trip.travelers ? `For ${trip.travelers}.` : ""} ${trip.budget && trip.budget !== "Budget" ? `Budget: ${trip.budget}.` : ""}`
              onPopulateInput(base)
            }}
            title="Update plan with current trip details"
          >
            Update plan
          </Button>
        )}
      </div>

      {/* Right Section - Invite and User Menu */}
      <div className="flex items-center gap-3">
        
        {/* Invite Button */}
        <div className="relative">
          <Button
            variant="outline"
            size="sm"
            className="border-gray-300 text-gray-700 hover:bg-gray-50 px-4 py-2 h-8 text-sm rounded-full"
            onClick={async () => {
              if (!inviteLink) {
                onInvite()
              }
              setShowInvitePopover((v) => !v)
            }}
          >
            Invite
          </Button>
          {showInvitePopover && (
            <div className="absolute right-0 mt-3 w-80 bg-white border border-gray-200 rounded-lg shadow-xl p-4 z-[9999]">
              {inviteLink ? (
                <>
                  <div className="text-sm text-gray-600 mb-3 font-medium">Share this link to invite collaborators:</div>
                  <div className="flex items-center gap-2">
                    <Input 
                      readOnly 
                      value={inviteLink} 
                      className="h-8 text-sm bg-gray-50 border-gray-200" 
                      onFocus={(e) => e.currentTarget.select()} 
                    />
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-8 px-3 rounded-md"
                      onClick={async () => {
                        try { 
                          await navigator.clipboard.writeText(inviteLink)
                          // Could add a toast notification here
                        } catch {}
                      }}
                    >
                      Copy
                    </Button>
                  </div>
                </>
              ) : (
                <div className="text-sm text-gray-600 flex items-center gap-2">
                  <div className="animate-spin w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
                  Generating invite link...
                </div>
              )}
            </div>
          )}
        </div>
        
        {/* User Menu */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="outline"
              size="sm"
              className="border-gray-300 bg-white hover:bg-gray-50 w-8 h-8 p-0 rounded-full"
            >
              <User className="w-4 h-4 text-gray-700" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-52 bg-white border border-gray-200 shadow-xl rounded-lg p-2 z-[10001]">
            {onDeleteTrip && (
              <DropdownMenuItem
                className="flex items-center gap-3 cursor-pointer text-red-600 hover:bg-red-50 rounded-lg p-3 transition-all"
                disabled={!!isDeleting}
                onClick={isDeleting ? undefined : onDeleteTrip}
              >
                {isDeleting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Deleting...</span>
                  </>
                ) : (
                  <>
                    <span className="text-red-600 text-sm">🗑️</span>
                    <span className="font-medium">Delete trip</span>
                  </>
                )}
              </DropdownMenuItem>
            )}
            <DropdownMenuSeparator className="my-2 bg-gray-100" />
            <DropdownMenuItem asChild>
              <Link href="/profile" className="flex items-center gap-3 cursor-pointer hover:bg-gray-50 rounded-lg p-3 transition-all">
                <User className="w-4 h-4 text-blue-600" />
                <span className="font-medium">Profile</span>
              </Link>
            </DropdownMenuItem>
            <DropdownMenuSeparator className="my-2 bg-gray-100" />
            <DropdownMenuItem 
              className="flex items-center gap-3 cursor-pointer text-red-600 hover:bg-red-50 rounded-lg p-3 transition-all"
              onClick={onSignOut || handleSignOut}
            >
              <LogOut className="w-4 h-4 text-red-600" />
              <span className="font-medium">Logout</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  )
})
