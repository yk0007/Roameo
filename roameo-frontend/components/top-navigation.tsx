"use client"

import { useState } from "react"
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
  onSignOut?: () => void
}

export function TopNavigation({
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
  onSignOut,
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

  const handleSignOut = async () => {
    // Navigate immediately for instant UX
    window.location.href = "/auth/login"
    // Sign out in background
    await supabase.auth.signOut()
  }

  const handleLogoClick = async () => {
    try {
      const { data: { session } } = await supabase.auth.getSession()
      if (session) router.push("/dashboard")
      else router.push("/")
    } catch {
      router.push("/")
    }
  }

  const handleEdit = (field: string) => {
    setEditingField(field)
    setTempValues({
      title: trip.title,
      origin: trip.origin,
      destination: trip.destination,
      duration: trip.duration,
      travelers: trip.travelers,
      budget: trip.budget,
    })
  }

  const handleSave = (field: string) => {
    onTripUpdate({
      ...trip,
      [field]: tempValues[field as keyof typeof tempValues],
    })
    setEditingField(null)
  }

  const handleCancel = () => {
    setEditingField(null)
    setTempValues({
      title: trip.title,
      origin: trip.origin,
      destination: trip.destination,
      duration: trip.duration,
      travelers: trip.travelers,
      budget: trip.budget,
    })
  }

  return (
    <div className="flex items-center justify-between px-6 py-3 relative">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <button onClick={handleLogoClick} className="hover:opacity-80 transition-opacity flex items-center gap-3">
            <div className="w-8 h-8 bg-black rounded-full flex items-center justify-center">
              <div className="w-2 h-2 bg-white rounded-full"></div>
            </div>
            <span className="text-xl font-bold text-gray-900">roameo</span>
          </button>
        </div>
        {editingField === "title" ? (
          <div className="flex items-center gap-2">
            <Input
              value={tempValues.title}
              onChange={(e) => setTempValues({ ...tempValues, title: e.target.value })}
              className="w-48 h-8 text-sm rounded-full bg-white/80 backdrop-blur-sm border-white/30"
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSave("title")
                if (e.key === "Escape") handleCancel()
              }}
              autoFocus
            />
            <Button size="sm" onClick={() => handleSave("title")} className="h-8 px-3 text-xs">✓</Button>
          </div>
        ) : (
          <Button
            variant="ghost"
            className="text-lg font-semibold p-0 h-auto hover:bg-transparent border-slate-200 rounded-full border-0"
            onClick={() => handleEdit("title")}
            title="Edit title"
          >
            {trip.title}
            <ChevronDown className="w-4 h-4 ml-1" />
          </Button>
        )}
      </div>

      <div className="flex items-center gap-4 text-sm">
        {/* Origin */}
        <div className="flex items-center gap-1">
          <MapPin className="w-4 h-4 text-gray-500 rounded p-0.5 border-transparent px-0 py-0 border-0" />
          {editingField === "origin" ? (
            <div className="flex items-center gap-1">
              <Input
                value={tempValues.origin}
                onChange={(e) => setTempValues({ ...tempValues, origin: e.target.value })}
                className="w-24 h-6 text-xs rounded-full bg-white/80 backdrop-blur-sm border-white/30"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSave("origin")
                  if (e.key === "Escape") handleCancel()
                }}
                autoFocus
              />
              <Button size="sm" onClick={() => handleSave("origin")} className="h-6 px-2 text-xs">
                ✓
              </Button>
            </div>
          ) : (
            <Button
              variant="ghost"
              onClick={() => handleEdit("origin")}
              className="text-gray-700 p-0 h-auto hover:bg-transparent border-0"
            >
              {trip.origin || "Add origin"}
            </Button>
          )}
        </div>

        {/* Destination */}
        <div className="flex items-center gap-1">
          <MapPin className="w-4 h-4 text-gray-500 rounded p-0.5 border-transparent border-0 px-0 py-0" />
          {editingField === "destination" ? (
            <div className="flex items-center gap-1">
              <Input
                value={tempValues.destination}
                onChange={(e) => setTempValues({ ...tempValues, destination: e.target.value })}
                className="w-32 h-6 text-xs rounded-full bg-white/80 backdrop-blur-sm border-white/30"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSave("destination")
                  if (e.key === "Escape") handleCancel()
                }}
                autoFocus
              />
              <Button size="sm" onClick={() => handleSave("destination")} className="h-6 px-2 text-xs">
                ✓
              </Button>
            </div>
          ) : (
            <Button
              variant="ghost"
              onClick={() => handleEdit("destination")}
              className="text-gray-700 p-0 h-auto hover:bg-transparent"
            >
              {trip.destinations && trip.destinations.length > 1 
                ? `${trip.destinations.length} destinations` 
                : trip.destination || trip.destinations?.[0] || "Destination"}
            </Button>
          )}
        </div>

        {/* Duration */}
        {editingField === "duration" ? (
          <div className="flex items-center gap-1">
            <Input
              type="number"
              placeholder="Days"
              value={parseInt(/\d+/.exec(tempValues.duration || "")?.[0] || "", 10) as any}
              onChange={(e) => {
                const v = e.target.value
                setTempValues({ ...tempValues, duration: v ? `${v} days` : "" })
              }}
              className="w-24 h-6 text-xs rounded-full bg-white/80 backdrop-blur-sm border-white/30"
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSave("duration")
                if (e.key === "Escape") handleCancel()
              }}
              autoFocus
            />
            <Button size="sm" onClick={() => handleSave("duration")} className="h-6 px-2 text-xs">
              ✓
            </Button>
          </div>
        ) : (
          <Badge
            variant="secondary"
            onClick={() => handleEdit("duration")}
            className="bg-white/80 backdrop-blur-sm text-gray-700 hover:bg-white/90 cursor-pointer border border-white/30 rounded-full"
          >
            <Calendar className="w-3 h-3 mr-1 border-current rounded p-0.5 border-transparent px-0 py-0 border-0" />
            {trip.duration || "Days"}
          </Badge>
        )}

        {/* Travelers */}
        {editingField === "travelers" ? (
          <div className="flex items-center gap-1">
            <Input
              type="number"
              placeholder="Travellers"
              value={parseInt(/\d+/.exec(tempValues.travelers || "")?.[0] || "", 10) as any}
              onChange={(e) => {
                const v = e.target.value
                setTempValues({ ...tempValues, travelers: v ? `${v} travelers` : "" })
              }}
              className="w-28 h-6 text-xs rounded-full bg-white/80 backdrop-blur-sm border-white/30"
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSave("travelers")
                if (e.key === "Escape") handleCancel()
              }}
              autoFocus
            />
            <Button size="sm" onClick={() => handleSave("travelers")} className="h-6 px-2 text-xs">
              ✓
            </Button>
          </div>
        ) : (
          <Badge
            variant="secondary"
            onClick={() => handleEdit("travelers")}
            className="bg-white/80 backdrop-blur-sm text-gray-700 hover:bg-white/90 cursor-pointer border border-white/30 rounded-full"
          >
            <Users className="w-3 h-3 mr-1 border-current rounded p-0.5 border-transparent border-0 px-0 py-0" />
            {trip.travelers || "Travellers"}
          </Badge>
        )}

        {/* Budget */}
        {editingField === "budget" ? (
          <div className="flex items-center gap-1">
            <Input
              type="number"
              placeholder="Budget"
              value={parseInt((tempValues.budget || "").replace(/[^0-9]/g, "") || "", 10) as any}
              onChange={(e) => {
                const v = e.target.value
                setTempValues({ ...tempValues, budget: v })
              }}
              className="w-24 h-6 text-xs rounded-full bg-white/80 backdrop-blur-sm border-white/30"
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSave("budget")
                if (e.key === "Escape") handleCancel()
              }}
              autoFocus
            />
            <Button size="sm" onClick={() => handleSave("budget")} className="h-6 px-2 text-xs">
              ✓
            </Button>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <Badge
              variant="secondary"
              onClick={() => handleEdit("budget")}
              className="bg-white/80 backdrop-blur-sm text-gray-700 hover:bg-white/90 cursor-pointer border border-white/30 rounded-full"
            >
              {trip.budget ? `₹${trip.budget}` : "Budget"}
            </Badge>
            {onReplan && (
              <Button
                variant="outline"
                size="sm"
                className="rounded-full bg-white/80 backdrop-blur-sm border-0 shadow-md hover:shadow-lg transition-shadow"
                onClick={onReplan}
                title="Update plan with current trip details"
              >
                Update plan
              </Button>
            )}
          </div>
        )}
      </div>

      <div className="flex items-center gap-3">
        <div className="relative">
          <Button
            variant="outline"
            size="sm"
            className="rounded-full bg-white/80 backdrop-blur-sm border-0 shadow-md hover:shadow-lg transition-shadow"
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
            <div className="absolute right-0 mt-2 w-72 bg-white border-0 rounded-xl shadow-xl p-3 z-[9999]">
              {inviteLink ? (
                <>
                  <div className="text-xs text-gray-600 mb-2">Share this link to invite collaborators:</div>
                  <div className="flex items-center gap-2">
                    <Input readOnly value={inviteLink} className="h-8 text-xs" onFocus={(e) => e.currentTarget.select()} />
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-8"
                      onClick={async () => {
                        try { await navigator.clipboard.writeText(inviteLink) } catch {}
                      }}
                    >
                      Copy
                    </Button>
                  </div>
                </>
              ) : (
                <div className="text-xs text-gray-600">Generating invite link...</div>
              )}
            </div>
          )}
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="outline"
              size="sm"
              className="rounded-full bg-white/80 backdrop-blur-sm border-0 shadow-md hover:shadow-lg transition-shadow"
            >
              <User className="w-4 h-4 border-current rounded p-0.5 border-0 px-0 py-0" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48 bg-white/95 backdrop-blur-md border-0 shadow-xl rounded-xl z-[10001]">
            {onDeleteTrip && (
              <DropdownMenuItem
                className="flex items-center gap-2 cursor-pointer text-red-600"
                disabled={!!isDeleting}
                onClick={isDeleting ? undefined : onDeleteTrip}
              >
                {isDeleting && <Loader2 className="w-4 h-4 animate-spin" />}
                {isDeleting ? "Deleting…" : "Delete trip"}
              </DropdownMenuItem>
            )}
            <DropdownMenuSeparator />
            <DropdownMenuItem asChild>
              <Link href="/profile" className="flex items-center gap-2 cursor-pointer">
                <User className="w-4 h-4" />
                Profile
              </Link>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem 
              className="flex items-center gap-2 cursor-pointer text-red-600"
              onClick={onSignOut || handleSignOut}
            >
              <LogOut className="w-4 h-4" />
              Logout
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  )
}
