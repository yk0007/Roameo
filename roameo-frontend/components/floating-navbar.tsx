"use client"

import { usePathname } from "next/navigation"
import { TopNavigation } from "./top-navigation"

interface Trip {
  id: string
  title: string
  origin: string
  destination: string
  duration: string
  travelers: string
  budget: string
}

interface FloatingNavbarProps {
  trip?: Trip
  onTripUpdate?: (trip: Trip) => void
  isRightPanelVisible?: boolean
  onToggleRightPanel?: () => void
  onSaveTrip?: () => void
  onInvite?: () => void
  inviteLink?: string
  onDeleteTrip?: () => void
  isDeleting?: boolean
  onReplan?: () => void
  onSignOut?: () => void
}

export function FloatingNavbar(props: FloatingNavbarProps) {
  const pathname = usePathname()
  
  // Don't show floating navbar on these pages
  const excludedPaths = ['/chat', '/dashboard', '/profile', '/auth/login', '/']
  if (excludedPaths.includes(pathname)) {
    return null
  }

  // Don't show if no trip data
  if (!props.trip) {
    return null
  }

  // Ensure required props have defaults
  const safeProps = {
    trip: props.trip,
    onTripUpdate: props.onTripUpdate || (() => {}),
    isRightPanelVisible: props.isRightPanelVisible || false,
    onToggleRightPanel: props.onToggleRightPanel || (() => {}),
    onSaveTrip: props.onSaveTrip || (() => {}),
    onInvite: props.onInvite || (() => {}),
    inviteLink: props.inviteLink,
    onDeleteTrip: props.onDeleteTrip,
    isDeleting: props.isDeleting,
    onReplan: props.onReplan,
    onSignOut: props.onSignOut,
  }

  return (
    <div className="fixed top-4 left-1/2 transform -translate-x-1/2 z-[10000] w-full max-w-7xl mx-auto px-4 animate-in slide-in-from-top-2 duration-300">
      <div className="bg-white backdrop-blur-xl border border-white/30 rounded-2xl shadow-2xl shadow-black/5 hover:shadow-black/10 transition-all duration-200">
        <TopNavigation {...safeProps} />
      </div>
    </div>
  )
}
