"use client"

import { useState, useCallback, memo, useEffect, useRef } from "react"
import { useRouter } from "next/navigation"
import { supabase } from "@/lib/supabase/client"
import { ChevronDown, MapPin, Calendar, Users, LogOut, Loader2, IndianRupee } from "lucide-react"
import { Button } from "@/components/ui/button"
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
}

function metaChipClassName(editing: boolean, recentlySaved: boolean) {
  return editing
    ? "flex items-center gap-2 rounded-full border border-[#d1d5db] bg-[#f3f4f6] px-3 py-1.5 text-[12px] text-[#111827] shadow-[0_2px_8px_rgba(0,0,0,0.05)] transition-all duration-200 ease-out"
    : `flex items-center gap-2 px-3 text-[12px] font-medium transition-all duration-200 ease-out ${
        recentlySaved
          ? "rounded-full bg-[#f3f4f6] text-[#111827] shadow-[0_2px_8px_rgba(0,0,0,0.05)]"
          : "text-[#6b7280]"
      }`
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
}: TopNavigationProps) {
  const [editingField, setEditingField] = useState<string | null>(null)
  const [recentlySavedField, setRecentlySavedField] = useState<string | null>(null)
  const [tempValues, setTempValues] = useState({
    title: trip.title,
    origin: trip.origin,
    destination: trip.destination,
    duration: trip.duration,
    travelers: trip.travelers,
    budget: trip.budget,
  })
  const router = useRouter()
  const containerRef = useRef<HTMLDivElement | null>(null)

  const handleSignOut = useCallback(async () => {
    try {
      await supabase.auth.signOut()
    } catch (e) {
      // ignore and force navigation anyway
    } finally {
      // Hard navigation to prevent flicker/intermediate renders
      window.location.replace("/auth/login")
    }
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
    setRecentlySavedField(field)
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

  useEffect(() => {
    if (!editingField) {
      return
    }

    const handlePointerDown = (event: MouseEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) {
        handleSave(editingField)
      }
    }

    document.addEventListener("mousedown", handlePointerDown)
    return () => document.removeEventListener("mousedown", handlePointerDown)
  }, [editingField, handleSave])

  useEffect(() => {
    if (!recentlySavedField) {
      return
    }

    const timer = setTimeout(() => setRecentlySavedField(null), 700)
    return () => clearTimeout(timer)
  }, [recentlySavedField])

  return (
    <div ref={containerRef} className="relative z-20 grid h-[64px] grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-5 bg-transparent px-7">
      <div className="flex min-w-0 items-center gap-5">
        <button
          onClick={handleLogoClick}
          className="flex items-center gap-3 rounded-full"
        >
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[linear-gradient(135deg,#000000_0%,#374151_100%)] shadow-[0_2px_8px_rgba(0,0,0,0.2)]">
            <div className="h-2 w-2 rounded-full bg-white" />
          </div>
          <span className="text-[17px] font-bold tracking-[-0.04em] text-black">roameo</span>
        </button>

        <div className="h-5 w-px bg-[#e5e7eb]/80" />

        <div className="min-w-0">
          {editingField === "title" ? (
            <div className="flex items-center gap-2 rounded-full border border-[#d1d5db] bg-[#f3f4f6] px-2 py-1 transition-all duration-200 ease-out">
              <Input
                value={tempValues.title}
                onChange={(e) => setTempValues({ ...tempValues, title: e.target.value })}
                placeholder="Trip title"
                className="h-8 w-64 border-0 bg-transparent px-3 text-center text-[14px] text-[#111827] placeholder:text-[#9ca3af] shadow-none focus-visible:ring-0"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSave("title")
                  if (e.key === "Escape") handleCancel()
                }}
                autoFocus
              />
              <Button size="sm" onClick={() => handleSave("title")} className="h-8 rounded-full bg-black px-4 text-[12px] text-white hover:bg-black/90">Save</Button>
            </div>
          ) : (
            <Button
              variant="ghost"
              className={`min-w-0 gap-1.5 rounded-[12px] px-3 py-1.5 text-[14px] font-medium transition-all duration-200 ease-out hover:bg-transparent ${
                recentlySavedField === "title" ? "bg-[#f3f4f6] text-[#111827] shadow-[0_2px_8px_rgba(0,0,0,0.05)]" : "text-[#111827]"
              }`}
              onClick={() => handleEdit("title")}
            >
              <span className="truncate">{trip.title || "My Trip"}</span>
              <ChevronDown className="h-4 w-4 text-[#9ca3af]" />
            </Button>
          )}
        </div>
      </div>

      <div className="flex items-center justify-center gap-4">
        <div className="flex items-center rounded-full border border-[rgba(255,255,255,0.8)] bg-[rgba(255,255,255,0.72)] px-2 py-1.5 shadow-[0_4px_16px_rgba(0,0,0,0.12)] backdrop-blur-xl">
          <div className={metaChipClassName(editingField === "origin", recentlySavedField === "origin")}>
            <MapPin className="h-3.5 w-3.5" />
          {editingField === "origin" ? (
              <div className="flex items-center gap-2">
              <Input
                value={tempValues.origin}
                onChange={(e) => setTempValues({ ...tempValues, origin: e.target.value })}
                  placeholder="Origin"
                  className="h-6 w-[92px] border-0 bg-transparent px-0 text-center text-[12px] text-[#111827] placeholder:text-[#9ca3af] shadow-none focus-visible:ring-0"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSave("origin")
                  if (e.key === "Escape") handleCancel()
                }}
                autoFocus
              />
                <Button size="sm" onClick={() => handleSave("origin")} className="h-6 shrink-0 rounded-full bg-black px-2.5 text-[10px] text-white hover:bg-black/90">OK</Button>
            </div>
          ) : (
              <span
              onClick={() => handleEdit("origin")}
                className="cursor-pointer"
            >
              {trip.origin || "Origin"}
            </span>
          )}
        </div>

          <div className="h-4 w-px bg-[#d1d5db]/70" />

          <div className={metaChipClassName(editingField === "destination", recentlySavedField === "destination")}>
            <MapPin className="h-3.5 w-3.5" />
          {editingField === "destination" ? (
              <div className="flex items-center gap-2">
              <Input
                value={tempValues.destination}
                onChange={(e) => setTempValues({ ...tempValues, destination: e.target.value })}
                  placeholder="Destination"
                  className="h-6 w-[124px] border-0 bg-transparent px-0 text-center text-[12px] text-[#111827] placeholder:text-[#9ca3af] shadow-none focus-visible:ring-0"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSave("destination")
                  if (e.key === "Escape") handleCancel()
                }}
                autoFocus
              />
                <Button size="sm" onClick={() => handleSave("destination")} className="h-6 shrink-0 rounded-full bg-black px-2.5 text-[10px] text-white hover:bg-black/90">OK</Button>
            </div>
          ) : (
              <span
              onClick={() => handleEdit("destination")}
                className="cursor-pointer"
            >
              {trip.destinations && trip.destinations.length > 1 
                ? `${trip.destinations.length} destinations` 
                : trip.destination || trip.destinations?.[0] || "Destination"}
            </span>
          )}
        </div>

          <div className="h-4 w-px bg-[#d1d5db]/70" />

          <div className={metaChipClassName(editingField === "duration", recentlySavedField === "duration")}>
            <Calendar className="h-3.5 w-3.5" />
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
                  className="h-6 w-[48px] border-0 bg-transparent px-0 text-center text-[12px] text-[#111827] placeholder:text-[#9ca3af] shadow-none focus-visible:ring-0"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSave("duration")
                  if (e.key === "Escape") handleCancel()
                }}
                autoFocus
              />
                <Button size="sm" onClick={() => handleSave("duration")} className="h-6 shrink-0 rounded-full bg-black px-2.5 text-[10px] text-white hover:bg-black/90">OK</Button>
            </div>
          ) : (
              <span
              onClick={() => handleEdit("duration")}
                className="cursor-pointer"
            >
              {trip.duration || "0 days"}
            </span>
          )}
        </div>

          <div className="h-4 w-px bg-[#d1d5db]/70" />

          <div className={metaChipClassName(editingField === "travelers", recentlySavedField === "travelers")}>
            <Users className="h-3.5 w-3.5" />
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
                  className="h-6 w-[40px] border-0 bg-transparent px-0 text-center text-[12px] text-[#111827] placeholder:text-[#9ca3af] shadow-none focus-visible:ring-0"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSave("travelers")
                  if (e.key === "Escape") handleCancel()
                }}
                autoFocus
              />
                <Button size="sm" onClick={() => handleSave("travelers")} className="h-6 shrink-0 rounded-full bg-black px-2.5 text-[10px] text-white hover:bg-black/90">OK</Button>
            </div>
          ) : (
              <span
              onClick={() => handleEdit("travelers")}
                className="cursor-pointer"
            >
                {trip.travelers?.replace(" travelers", "") || "1"}
            </span>
          )}
        </div>

          <div className="h-4 w-px bg-[#d1d5db]/70" />

          <div className={metaChipClassName(editingField === "budget", recentlySavedField === "budget")}>
            <IndianRupee className="h-3.5 w-3.5" />
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
                  className="h-6 w-[78px] border-0 bg-transparent px-0 text-center text-[12px] text-[#111827] placeholder:text-[#9ca3af] shadow-none focus-visible:ring-0"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSave("budget")
                  if (e.key === "Escape") handleCancel()
                }}
                autoFocus
              />
                <Button size="sm" onClick={() => handleSave("budget")} className="h-6 shrink-0 rounded-full bg-black px-2.5 text-[10px] text-white hover:bg-black/90">OK</Button>
            </div>
          ) : (
              <span
              onClick={() => handleEdit("budget")}
                className="cursor-pointer"
            >
              Budget
            </span>
          )}
        </div>
        </div>

        {onPopulateInput && (
          <Button
            size="sm"
            className="h-8 rounded-full bg-black px-4 text-[12px] font-medium text-white shadow-[0_2px_8px_rgba(0,0,0,0.15)] hover:bg-black/90"
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

      <div className="flex items-center justify-end">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              size="sm"
              className="h-[30px] w-[30px] rounded-full bg-[linear-gradient(135deg,#6366f1_0%,#8b5cf6_100%)] p-0 text-[12px] font-semibold text-white shadow-[0_0_0_2px_rgba(255,255,255,0.45),0_0_0_4px_rgba(255,255,255,0.6),0_4px_16px_rgba(99,102,241,0.25)] hover:opacity-95"
            >
              N
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="z-[10001] w-52 rounded-[24px] border border-slate-100/60 bg-white/60 p-2 shadow-[0_20px_80px_rgba(15,23,42,0.08),_0_6px_20px_rgba(15,23,42,0.05)] backdrop-blur-2xl">
            {onDeleteTrip && (
              <DropdownMenuItem
                className="flex cursor-pointer items-center gap-3 rounded-xl p-3 text-red-600 transition-all hover:bg-red-50 focus:bg-red-50"
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
            <DropdownMenuSeparator className="my-2 bg-slate-100 bg-opacity-60" />
            <DropdownMenuItem asChild className="focus:bg-slate-50/80">
              <Link href="/profile" className="flex cursor-pointer items-center gap-3 rounded-xl p-3 transition-all hover:bg-slate-50/80">
                <div className="flex h-6 w-6 items-center justify-center rounded-full bg-[#f1f5f9] text-[10px] font-semibold text-[#4f46e5] shadow-[0_2px_4px_rgba(0,0,0,0.02)]">N</div>
                <span className="font-medium text-slate-700">Profile Menu</span>
              </Link>
            </DropdownMenuItem>
            <DropdownMenuSeparator className="my-2 bg-slate-100 bg-opacity-60" />
            <DropdownMenuItem 
              className="flex cursor-pointer items-center gap-3 rounded-xl p-3 text-red-600 transition-all hover:bg-red-50 focus:bg-red-50"
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
