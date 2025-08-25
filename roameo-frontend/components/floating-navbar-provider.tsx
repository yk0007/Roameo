"use client"

import { useState, useEffect } from "react"
import { usePathname } from "next/navigation"
import { FloatingNavbar } from "./floating-navbar"

interface Trip {
  id: string
  title: string
  origin: string
  destination: string
  duration: string
  travelers: string
  budget: string
}

interface FloatingNavbarProviderProps {
  children: React.ReactNode
}

export function FloatingNavbarProvider({ children }: FloatingNavbarProviderProps) {
  const pathname = usePathname()
  const [trip, setTrip] = useState<Trip | null>(null)

  // Example trip data - in a real app this would come from context/state management
  useEffect(() => {
    // Only set trip data for pages that might need it
    // This is a placeholder - in practice this would come from your app's state management
    const shouldShowNavbar = !['/', '/chat', '/dashboard', '/profile', '/auth/login'].includes(pathname)
    
    if (shouldShowNavbar) {
      // This would typically come from your app's global state, URL params, or API
      setTrip({
        id: "example-trip",
        title: "Sample Trip",
        origin: "New York",
        destination: "Paris",
        duration: "7 days",
        travelers: "2 travelers",
        budget: "$3000"
      })
    } else {
      setTrip(null)
    }
  }, [pathname])

  const handleTripUpdate = (updatedTrip: Trip) => {
    setTrip(updatedTrip)
    // In a real app, you'd also persist this to your backend/state management
  }

  const handleSaveTrip = () => {
    console.log("Saving trip:", trip)
    // Implement trip saving logic
  }

  const handleInvite = () => {
    console.log("Creating invite for trip:", trip?.id)
    // Implement invite creation logic
  }

  const handleDeleteTrip = () => {
    console.log("Deleting trip:", trip?.id)
    // Implement trip deletion logic
    setTrip(null)
  }

  const handleReplan = () => {
    console.log("Replanning trip:", trip?.id)
    // Implement replan logic
  }

  const handleSignOut = () => {
    console.log("Signing out")
    // Implement sign out logic
  }

  return (
    <>
      <FloatingNavbar
        trip={trip || undefined}
        onTripUpdate={handleTripUpdate}
        isRightPanelVisible={false}
        onToggleRightPanel={() => {}}
        onSaveTrip={handleSaveTrip}
        onInvite={handleInvite}
        onDeleteTrip={handleDeleteTrip}
        onReplan={handleReplan}
        onSignOut={handleSignOut}
      />
      {children}
    </>
  )
}
