"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { supabase } from "@/lib/supabase/client"
import { Button } from "@/components/ui/button"
import { listTrips } from "@/lib/api"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import Image from "next/image"
import { MapPin, Users, Send, LogOut, Plus, Mic, Clock, ArrowRight, Plane, Camera, Compass, Palmtree, Globe, Map } from "lucide-react"
import DestinationCardArt from "@/components/DestinationCardArt"
import { Cover } from "@/components/ui/cover"

export default function Dashboard() {
  const router = useRouter()
  const [user, setUser] = useState<any>(null)
  const [message, setMessage] = useState("")
  const [loading, setLoading] = useState(true)
  const [trips, setTrips] = useState<Array<any>>([])

  // Load trips from backend with proper authentication
  useEffect(() => {
    let mounted = true
    let isLoading = false
    
    const load = async () => {
      if (isLoading || !user) return
      isLoading = true
      
      try {
        const { trips } = await listTrips()
        if (mounted) setTrips(trips)
      } catch (error) {
        console.error('Failed to load trips:', error)
        if (mounted) setTrips([])
      } finally {
        isLoading = false
      }
    }
    
    if (user) {
      load()
    }
    
    return () => {
      mounted = false
    }
  }, [user])

  useEffect(() => {
    const getUser = async () => {
      const {
        data: { session },
      } = await supabase.auth.getSession()
      if (!session) {
        router.push("/auth/login")
        return
      }
      setUser(session.user)
      setLoading(false)
    }

    getUser()

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event, session) => {
      if (!session) {
        router.push("/auth/login")
      } else {
        setUser(session.user)
      }
    })

    return () => subscription.unsubscribe()
  }, [router])

  const handleSignOut = async () => {
    try {
      // Sign out first, then navigate
      await supabase.auth.signOut()
      router.push("/auth/login")
    } catch (error) {
      console.error('Sign out error:', error)
      // Navigate anyway if sign out fails
      router.push("/auth/login")
    }
  }

  const handleSendMessage = () => {
    const text = message.trim()
    if (!text) return
    // Navigate to chat with initial message; chat page will send it and remove from URL
    router.push(`/chat?message=${encodeURIComponent(text)}`)
    setMessage("")
  }

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading your dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen w-full bg-[#f8fafc] relative scroll-smooth overflow-hidden">

      <header className="flex items-center justify-between px-6 py-3 border-0 bg-white/80 backdrop-blur-md shadow-lg">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-black rounded-full flex items-center justify-center shadow-md">
                <div className="w-2 h-2 bg-white rounded-full"></div>
              </div>
              <span className="text-xl font-bold text-gray-900">roameo</span>
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <span className="text-sm text-muted-foreground">
            Welcome, {user?.user_metadata?.username || user?.user_metadata?.first_name || user?.email?.split("@")[0]}
          </span>
          <Button
            onClick={() => router.push("/profile")}
            variant="ghost"
            size="sm"
            className="text-muted-foreground hover:text-foreground rounded-full bg-white/80 backdrop-blur-sm border-0 shadow-md hover:shadow-lg transition-shadow"
          >
            Profile
          </Button>
          <Button
            onClick={handleSignOut}
            variant="outline"
            size="sm"
            className="flex items-center space-x-2 rounded-full bg-white/80 backdrop-blur-sm border-0 shadow-md hover:shadow-lg transition-shadow"
          >
            <LogOut className="h-4 w-4" />
            <span>Sign Out</span>
          </Button>
        </div>
      </header>

      <main className="flex-1 p-6">
        <div className="max-w-6xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Welcome back!</h1>
            <p className="text-gray-600">Plan your next adventure or continue working on existing trips.</p>
          </div>

          <div className="mb-8">
            <form
              onSubmit={(e) => {
                e.preventDefault()
                handleSendMessage()
              }}
              className="relative"
            >
              <div className="bg-white border rounded-lg shadow-sm p-4">
                <div className="flex items-center gap-3">
                  <Input
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="Where would you like to go? What kind of experience are you looking for?"
                    className="flex-1 text-base"
                  />
                  <Button
                    type="submit"
                    className="bg-black hover:bg-gray-800 text-white"
                    disabled={!message.trim()}
                  >
                    <Send className="w-4 h-4 mr-2" />
                    Start Planning
                  </Button>
                </div>
              </div>
            </form>
          </div>

          <div className="flex items-center justify-between mb-8">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Your Trips</h2>
              <p className="text-gray-600">Manage and explore your planned adventures</p>
            </div>
            <Button className="bg-black hover:bg-gray-800 text-white" onClick={() => router.push("/chat")}>
              <Plus className="w-4 h-4 mr-2" />
              Plan New Trip
            </Button>
          </div>

          {trips.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {trips.map((trip: any, index: number) => {
                // Color palette for the dots
                const colors = [
                  'bg-blue-500',
                  'bg-red-500', 
                  'bg-green-500',
                  'bg-gray-500',
                  'bg-orange-500',
                  'bg-purple-500',
                  'bg-yellow-500',
                  'bg-pink-500'
                ]
                const colorClass = colors[index % colors.length]
                
                return (
                  <Card 
                    key={trip.id} 
                    className="group hover:shadow-xl transition-all duration-300 overflow-hidden bg-white/20 backdrop-blur-md border border-white/30 rounded-3xl hover:-translate-y-1 cursor-pointer"
                    onClick={() => router.push(`/chat?sessionId=${encodeURIComponent(trip.id)}`)}
                  >
                    <div className="p-6">
                      {/* Destination card art */}
                      <div className="relative h-48 bg-gray-100/50 backdrop-blur-sm rounded-2xl mb-4 overflow-hidden">
                        <DestinationCardArt
                          destination={trip.destination || trip.title}
                          variant="stamp"
                          className="rounded-2xl"
                        />
                      </div>
                      
                      {/* Content */}
                      <div className="space-y-3">
                        {/* Title with colored dot */}
                        <div className="flex items-center gap-3">
                          <div className={`w-3 h-3 rounded-full ${colorClass} flex-shrink-0`}></div>
                          <h3 className="font-semibold text-gray-900 text-lg truncate">{trip.title}</h3>
                        </div>
                        
                        {/* Destination with subtle styling */}
                        <div className="flex items-center gap-2 text-gray-600 text-sm">
                          <MapPin className="w-4 h-4" />
                          <span>{trip.destination}</span>
                        </div>
                      </div>
                    </div>
                  </Card>
                )
              })}
            </div>
          ) : (
            <div className="text-center py-16 rounded-2xl bg-white/20 backdrop-blur-md border border-white/30">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-white/30 backdrop-blur-sm flex items-center justify-center">
                <MapPin className="w-8 h-8 text-gray-600" />
              </div>
              <h3 className="text-lg font-semibold text-slate-900 mb-2">No trips yet</h3>
              <p className="text-slate-600 mb-6">Start planning your first adventure with Roameo</p>
              <Button className="bg-black hover:bg-gray-800 text-white rounded-full px-6" onClick={() => router.push("/chat") }>
                <Plus className="w-4 h-4 mr-2" /> Plan New Trip
              </Button>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
